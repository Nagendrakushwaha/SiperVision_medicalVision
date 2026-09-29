import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch

from backend.app.config import BASE_DIR, REPORTS_DIR, CHECKPOINTS_DIR
from backend.app.models.model_factory import (
    build_model,
    load_checkpoint,
    normalize_model_name,
    count_parameters,
    get_model_size_mb,
    SUPPORTED_MODELS
)

BENCHMARK_DIR = REPORTS_DIR / "benchmarks"
BENCHMARK_FILE = BENCHMARK_DIR / "benchmarks.json"

class BenchmarkService:
    """
    CPU Performance & Latency Benchmark Service.
    Measures warm-up runs, real forward passes, P50, P95, and throughput.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_benchmarks(self) -> List[Dict[str, Any]]:
        if BENCHMARK_FILE.exists():
            try:
                with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def run_benchmark(
        self,
        model_name: str = "resnet18",
        num_runs: int = 50,
        warmup_runs: int = 5
    ) -> Dict[str, Any]:
        """
        Runs rigorous CPU benchmark with dummy/real synthetic tensors.
        """
        norm_name = normalize_model_name(model_name)
        ckpt_dir = CHECKPOINTS_DIR / norm_name
        ckpt_path = ckpt_dir / "best.pth"
        if not ckpt_path.exists():
            ckpt_path = ckpt_dir / "last.pth"

        # Check if checkpoint exists or if benchmark is run on pretrained architecture
        checkpoint_status = "trained" if ckpt_path.exists() else "pretrained_uncalibrated"

        model = build_model(model_name=norm_name, pretrained=not ckpt_path.exists(), num_classes=2)
        if ckpt_path.exists():
            load_checkpoint(model, ckpt_path, device=self.device)
        model.to(self.device)
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224, device=self.device)

        # Warm-up cycles
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy_input)

        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                t0 = time.perf_counter()
                _ = model(dummy_input)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)

        latencies = np.array(latencies)
        avg_ms = float(np.mean(latencies))
        p50_ms = float(np.percentile(latencies, 50))
        p95_ms = float(np.percentile(latencies, 95))
        min_ms = float(np.min(latencies))
        max_ms = float(np.max(latencies))
        throughput_fps = round(1000.0 / avg_ms, 2) if avg_ms > 0 else 0.0

        params = count_parameters(model)
        size_mb = get_model_size_mb(model)

        record = {
            "model_id": norm_name,
            "model_name": SUPPORTED_MODELS[norm_name]["name"],
            "parameters": params["total_parameters"],
            "model_size_mb": size_mb,
            "device": str(self.device).upper(),
            "runs": num_runs,
            "avg_latency_ms": round(avg_ms, 2),
            "p50_latency_ms": round(p50_ms, 2),
            "p95_latency_ms": round(p95_ms, 2),
            "min_latency_ms": round(min_ms, 2),
            "max_latency_ms": round(max_ms, 2),
            "throughput_fps": throughput_fps,
            "status": checkpoint_status,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }

        # Update persistent file
        existing = self.get_benchmarks()
        # Replace or add
        existing = [b for b in existing if b.get("model_id") != norm_name]
        existing.append(record)

        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
        with open(BENCHMARK_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)

        return record

benchmark_service = BenchmarkService()
