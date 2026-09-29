import os
import time
import json
import csv
import threading
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim

from backend.app.config import BASE_DIR, CHECKPOINTS_DIR, REPORTS_DIR, EXPERIMENTS_DIR
from backend.app.models.model_factory import (
    build_model,
    normalize_model_name,
    save_checkpoint,
    get_model_info
)
from backend.app.services.dataset_loader import create_dataloaders

logger = logging.getLogger(__name__)

class TrainingManager:
    """
    Thread-safe training manager for CPU-first RSNA chest X-ray models.
    Supports real-time progress reporting, safe stopping, checkpointing,
    and persistent history logging.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.stop_requested = threading.Event()
        self.training_thread: Optional[threading.Thread] = None

        self.current_state: Dict[str, Any] = {
            "status": "idle",  # idle, preparing, training, validating, saving, completed, stopped, failed
            "model_name": None,
            "epoch": 0,
            "total_epochs": 0,
            "step": 0,
            "total_steps": 0,
            "train_loss": 0.0,
            "train_acc": 0.0,
            "val_loss": 0.0,
            "val_acc": 0.0,
            "learning_rate": 0.0,
            "elapsed_seconds": 0.0,
            "estimated_remaining_seconds": 0.0,
            "message": "Ready to train.",
            "error": None,
            "history": []
        }

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.current_state)

    def get_history(self, model_name: str) -> List[Dict[str, Any]]:
        norm_name = normalize_model_name(model_name)
        hist_file = REPORTS_DIR / "training" / norm_name / "history.json"
        if hist_file.exists():
            try:
                with open(hist_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def stop_training(self) -> Dict[str, Any]:
        with self.lock:
            if self.current_state["status"] in ("training", "preparing", "validating"):
                self.stop_requested.set()
                self.current_state["message"] = "Stopping requested. Completing current step..."
                return {"message": "Stop signal sent successfully."}
            return {"message": "No active training run to stop."}

    def start_training(
        self,
        model_name: str = "resnet18",
        epochs: int = 5,
        batch_size: int = 8,
        learning_rate: float = 0.0001,
        image_size: int = 224,
        early_stopping_patience: int = 2,
        seed: int = 42,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Validates arguments and launches training in a background daemon thread.
        """
        norm_name = normalize_model_name(model_name)

        # Enforce strict 1 <= epochs <= 100
        if not (1 <= int(epochs) <= 100):
            raise ValueError(f"Epochs must be between 1 and 100. Received: {epochs}")

        with self.lock:
            if self.current_state["status"] in ("training", "preparing", "validating"):
                raise RuntimeError("A training process is already running. Please wait or stop it first.")

            self.stop_requested.clear()
            self.current_state = {
                "status": "preparing",
                "model_name": norm_name,
                "epoch": 0,
                "total_epochs": epochs,
                "step": 0,
                "total_steps": 0,
                "train_loss": 0.0,
                "train_acc": 0.0,
                "val_loss": 0.0,
                "val_acc": 0.0,
                "learning_rate": learning_rate,
                "elapsed_seconds": 0.0,
                "estimated_remaining_seconds": 0.0,
                "message": f"Initializing {norm_name} architecture and data loaders...",
                "error": None,
                "history": []
            }

        self.training_thread = threading.Thread(
            target=self._run_training_loop,
            args=(norm_name, epochs, batch_size, learning_rate, image_size, early_stopping_patience, seed, max_train_samples, max_val_samples),
            daemon=True
        )
        self.training_thread.start()
        return {"status": "started", "model": norm_name, "epochs": epochs}

    def _run_training_loop(
        self,
        model_name: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        image_size: int,
        patience: int,
        seed: int,
        max_train_samples: Optional[int],
        max_val_samples: Optional[int]
    ):
        start_time = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)

        # Experiment metadata tracking
        exp_id = f"EXP-{int(time.time()) % 100000:05d}"
        history_records = []
        best_val_loss = float("inf")
        patience_counter = 0

        ckpt_dir = CHECKPOINTS_DIR / model_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = ckpt_dir / "best.pth"
        last_ckpt_path = ckpt_dir / "last.pth"

        hist_dir = REPORTS_DIR / "training" / model_name
        hist_dir.mkdir(parents=True, exist_ok=True)
        hist_json_path = hist_dir / "history.json"
        hist_csv_path = hist_dir / "history.csv"

        try:
            # 1. Build Model
            model = build_model(model_name=model_name, pretrained=True, num_classes=2)
            model.to(device)

            # 2. Build Dataloaders
            train_loader, val_loader, class_weights = create_dataloaders(
                batch_size=batch_size,
                image_size=image_size,
                num_workers=0,
                max_train_samples=max_train_samples,
                max_val_samples=max_val_samples
            )

            # 3. Setup Loss, Optimizer, Scheduler
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

            total_train_steps = len(train_loader)

            with self.lock:
                self.current_state["total_steps"] = total_train_steps
                self.current_state["status"] = "training"

            epoch_durations = []

            for epoch in range(1, epochs + 1):
                if self.stop_requested.is_set():
                    logger.info("Training stopped early by user request.")
                    break

                epoch_start = time.time()
                model.train()
                running_loss = 0.0
                running_corrects = 0
                total_samples = 0

                with self.lock:
                    self.current_state["status"] = "training"
                    self.current_state["epoch"] = epoch
                    self.current_state["message"] = f"Epoch {epoch}/{epochs} - Training..."

                for step, (inputs, labels) in enumerate(train_loader, start=1):
                    if self.stop_requested.is_set():
                        break

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    preds = torch.argmax(outputs, dim=1)
                    batch_size_curr = inputs.size(0)
                    running_loss += loss.item() * batch_size_curr
                    running_corrects += torch.sum(preds == labels.data).item()
                    total_samples += batch_size_curr

                    # Live update current step
                    current_loss = running_loss / total_samples
                    current_acc = (running_corrects / total_samples) * 100.0
                    elapsed = time.time() - start_time

                    # Estimate remaining time
                    steps_completed = (epoch - 1) * total_train_steps + step
                    total_expected_steps = epochs * total_train_steps
                    if steps_completed > 0:
                        sec_per_step = elapsed / steps_completed
                        rem_seconds = sec_per_step * (total_expected_steps - steps_completed)
                    else:
                        rem_seconds = 0.0

                    with self.lock:
                        self.current_state["step"] = step
                        self.current_state["train_loss"] = round(current_loss, 4)
                        self.current_state["train_acc"] = round(current_acc, 2)
                        self.current_state["elapsed_seconds"] = round(elapsed, 1)
                        self.current_state["estimated_remaining_seconds"] = round(max(0, rem_seconds), 1)

                if self.stop_requested.is_set():
                    break

                train_epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
                train_epoch_acc = (running_corrects / total_samples) * 100.0 if total_samples > 0 else 0.0

                # Validation Phase
                with self.lock:
                    self.current_state["status"] = "validating"
                    self.current_state["message"] = f"Epoch {epoch}/{epochs} - Validating..."

                model.eval()
                val_loss_running = 0.0
                val_corrects_running = 0
                val_samples_count = 0

                with torch.no_grad():
                    for val_inputs, val_labels in val_loader:
                        val_inputs = val_inputs.to(device)
                        val_labels = val_labels.to(device)

                        val_outputs = model(val_inputs)
                        v_loss = criterion(val_outputs, val_labels)

                        v_preds = torch.argmax(val_outputs, dim=1)
                        val_loss_running += v_loss.item() * val_inputs.size(0)
                        val_corrects_running += torch.sum(v_preds == val_labels.data).item()
                        val_samples_count += val_inputs.size(0)

                val_epoch_loss = val_loss_running / val_samples_count if val_samples_count > 0 else 0.0
                val_epoch_acc = (val_corrects_running / val_samples_count) * 100.0 if val_samples_count > 0 else 0.0

                current_lr = optimizer.param_groups[0]["lr"]
                scheduler.step(val_epoch_loss)

                epoch_dur = round(time.time() - epoch_start, 2)
                epoch_durations.append(epoch_dur)

                # Record epoch metrics
                record = {
                    "epoch": epoch,
                    "train_loss": round(train_epoch_loss, 4),
                    "train_accuracy": round(train_epoch_acc, 2),
                    "validation_loss": round(val_epoch_loss, 4),
                    "validation_accuracy": round(val_epoch_acc, 2),
                    "learning_rate": current_lr,
                    "epoch_duration": epoch_dur,
                    "timestamp": datetime.now().isoformat()
                }
                history_records.append(record)

                with self.lock:
                    self.current_state["val_loss"] = round(val_epoch_loss, 4)
                    self.current_state["val_acc"] = round(val_epoch_acc, 2)
                    self.current_state["learning_rate"] = current_lr
                    self.current_state["history"] = list(history_records)

                # Save Checkpoints
                with self.lock:
                    self.current_state["status"] = "saving"

                save_checkpoint(
                    model=model,
                    checkpoint_path=last_ckpt_path,
                    epoch=epoch,
                    metrics={"val_loss": val_epoch_loss, "val_acc": val_epoch_acc},
                    optimizer_state=optimizer.state_dict()
                )

                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    patience_counter = 0
                    save_checkpoint(
                        model=model,
                        checkpoint_path=best_ckpt_path,
                        epoch=epoch,
                        metrics={"val_loss": val_epoch_loss, "val_acc": val_epoch_acc},
                        optimizer_state=optimizer.state_dict()
                    )
                else:
                    patience_counter += 1

                # Write permanent history after each epoch
                self._save_history_files(hist_json_path, hist_csv_path, history_records)

                # Check Early Stopping
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {patience} epochs without validation loss improvement.")
                    break

            final_status = "stopped" if self.stop_requested.is_set() else "completed"
            total_duration = round(time.time() - start_time, 2)

            with self.lock:
                self.current_state["status"] = final_status
                self.current_state["message"] = f"Training {final_status} in {total_duration}s."

            # Save Experiment record
            self._save_experiment_record(
                exp_id=exp_id,
                model_name=model_name,
                epochs_requested=epochs,
                epochs_completed=len(history_records),
                batch_size=batch_size,
                learning_rate=learning_rate,
                image_size=image_size,
                seed=seed,
                device=str(device),
                best_val_loss=best_val_loss if best_val_loss != float("inf") else 0.0,
                final_val_acc=history_records[-1]["validation_accuracy"] if history_records else 0.0,
                duration=total_duration,
                checkpoint_path=str(best_ckpt_path.relative_to(BASE_DIR)),
                status=final_status
            )

        except Exception as e:
            logger.exception("Error during training loop")
            with self.lock:
                self.current_state["status"] = "failed"
                self.current_state["error"] = str(e)
                self.current_state["message"] = f"Training failed: {str(e)}"

    def _save_history_files(self, json_path: Path, csv_path: Path, records: List[Dict[str, Any]]):
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        if records:
            keys = list(records[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(records)

    def _save_experiment_record(self, **kwargs):
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        exp_file = EXPERIMENTS_DIR / "experiments.json"
        experiments = []
        if exp_file.exists():
            try:
                with open(exp_file, "r", encoding="utf-8") as f:
                    experiments = json.load(f)
            except Exception:
                experiments = []
        experiments.insert(0, kwargs)
        with open(exp_file, "w", encoding="utf-8") as f:
            json.dump(experiments, f, indent=2)

training_manager = TrainingManager()
