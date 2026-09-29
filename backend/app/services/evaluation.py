import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from backend.app.config import BASE_DIR, CHECKPOINTS_DIR, REPORTS_DIR
from backend.app.models.model_factory import (
    build_model,
    load_checkpoint,
    normalize_model_name,
    count_parameters,
    get_model_size_mb
)
from backend.app.services.dataset_loader import RSNAPneumoniaDataset
from backend.app.services.dataset_service import dataset_service
from backend.app.services.preprocessing import get_val_transforms

logger = logging.getLogger(__name__)

EVALUATION_DIR = REPORTS_DIR / "evaluation"

class EvaluationService:
    """
    Executes actual model evaluation on the test or validation split.
    Calculates exact classification metrics, ROC curve, PR curve,
    confusion matrix, and latency profile.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_evaluation(self, model_name: str = "resnet18") -> Optional[Dict[str, Any]]:
        norm_name = normalize_model_name(model_name)
        report_file = EVALUATION_DIR / f"{norm_name}_evaluation.json"
        if report_file.exists():
            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def run_evaluation(
        self,
        model_name: str = "resnet18",
        split_name: str = "test",
        max_samples: Optional[int] = 300
    ) -> Dict[str, Any]:
        """
        Runs full test-set evaluation using real model checkpoint.
        """
        norm_name = normalize_model_name(model_name)
        ckpt_dir = CHECKPOINTS_DIR / norm_name
        ckpt_path = ckpt_dir / "best.pth"
        if not ckpt_path.exists():
            ckpt_path = ckpt_dir / "last.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No checkpoint found for {norm_name}. Please train the model first.")

        # 1. Load model
        model = build_model(model_name=norm_name, pretrained=False, num_classes=2)
        load_checkpoint(model, ckpt_path, device=self.device)
        model.eval()

        # 2. Prepare test dataset
        split = dataset_service.get_split()
        patient_ids = split.get(split_name, split["test"])
        dataset = RSNAPneumoniaDataset(
            patient_ids=patient_ids,
            transform=get_val_transforms(224),
            max_samples=max_samples
        )
        if len(dataset) == 0:
            raise RuntimeError(f"No valid test samples found for split: {split_name}")

        y_true = []
        y_pred = []
        y_probs = []
        latencies = []

        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        with torch.no_grad():
            for img, target in loader:
                img = img.to(self.device)
                
                t0 = time.perf_counter()
                logits = model(img)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)

                probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
                pred = int(torch.argmax(logits, dim=1).item())

                y_true.append(int(target.item()))
                y_pred.append(pred)
                y_probs.append(float(probs[1]))  # Probability of pneumonia

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_probs = np.array(y_probs)

        # Basic Metrics
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        sensitivity = rec  # TP / (TP + FN)
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        # ROC Curve & AUC
        try:
            roc_auc = float(roc_auc_score(y_true, y_probs))
            fpr, tpr, roc_thresh = roc_curve(y_true, y_probs)
            # Sample down if too large for clean chart rendering
            step = max(1, len(fpr) // 50)
            roc_points = [
                {"fpr": round(float(f), 4), "tpr": round(float(t), 4), "threshold": round(float(th), 4)}
                for f, t, th in zip(fpr[::step], tpr[::step], roc_thresh[::step])
            ]
        except Exception:
            roc_auc = 0.5
            roc_points = []

        # Precision-Recall Curve & Average Precision
        try:
            pr_auc = float(average_precision_score(y_true, y_probs))
            prec_curve, rec_curve, pr_thresh = precision_recall_curve(y_true, y_probs)
            step_pr = max(1, len(prec_curve) // 50)
            pr_points = [
                {"precision": round(float(p), 4), "recall": round(float(r), 4)}
                for p, r in zip(prec_curve[::step_pr], rec_curve[::step_pr])
            ]
        except Exception:
            pr_auc = 0.0
            pr_points = []

        # Latency & Model Stats
        params = count_parameters(model)
        size_mb = get_model_size_mb(model)
        avg_latency = float(np.mean(latencies))
        min_latency = float(np.min(latencies))
        max_latency = float(np.max(latencies))
        p50_latency = float(np.percentile(latencies, 50))
        p95_latency = float(np.percentile(latencies, 95))

        evaluation_data = {
            "model_name": norm_name,
            "split": split_name,
            "evaluated_samples": len(y_true),
            "metrics": {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "sensitivity": round(sensitivity, 4),
                "specificity": round(specificity, 4),
                "roc_auc": round(roc_auc, 4),
                "pr_auc": round(pr_auc, 4)
            },
            "confusion_matrix": {
                "raw": {
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp)
                },
                "normalized": {
                    "tn": round(float(tn / (tn + fp)), 4) if (tn + fp) > 0 else 0.0,
                    "fp": round(float(fp / (tn + fp)), 4) if (tn + fp) > 0 else 0.0,
                    "fn": round(float(fn / (fn + tp)), 4) if (fn + tp) > 0 else 0.0,
                    "tp": round(float(tp / (fn + tp)), 4) if (fn + tp) > 0 else 0.0
                }
            },
            "roc_curve": roc_points,
            "pr_curve": pr_points,
            "model_specs": {
                "total_parameters": params["total_parameters"],
                "model_size_mb": size_mb,
                "avg_latency_ms": round(avg_latency, 2),
                "min_latency_ms": round(min_latency, 2),
                "max_latency_ms": round(max_latency, 2),
                "p50_latency_ms": round(p50_latency, 2),
                "p95_latency_ms": round(p95_latency, 2),
                "device": str(self.device).upper()
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }

        # Persist report
        EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
        with open(EVALUATION_DIR / f"{norm_name}_evaluation.json", "w", encoding="utf-8") as f:
            json.dump(evaluation_data, f, indent=2)

        return evaluation_data

evaluation_service = EvaluationService()
