import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.benchmark_service import benchmark_service

def main():
    parser = argparse.ArgumentParser(description="SmartMed Vision CPU Latency Benchmark CLI")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "mobilenet", "efficientnet"], help="Model architecture")
    parser.add_argument("--runs", type=int, default=50, help="Number of benchmark iterations")
    args = parser.parse_args()

    print("=" * 70)
    print(f" SmartMed Vision: CPU Inference Benchmark")
    print(f" Model:  {args.model.upper()}")
    print(f" Device: CPU (AMD Ryzen 5 5500U)")
    print(f" Iterations: {args.runs}")
    print("=" * 70)

    try:
        res = benchmark_service.run_benchmark(model_name=args.model, num_runs=args.runs)
        print(f"\n[BENCHMARK RESULTS]")
        print(f"Model:           {res['model_name']} ({res['status']})")
        print(f"Parameters:      {res['parameters']:,}")
        print(f"Model Size:      {res['model_size_mb']} MB")
        print(f"Average Latency: {res['avg_latency_ms']} ms")
        print(f"P50 Latency:     {res['p50_latency_ms']} ms")
        print(f"P95 Latency:     {res['p95_latency_ms']} ms")
        print(f"Min Latency:     {res['min_latency_ms']} ms")
        print(f"Max Latency:     {res['max_latency_ms']} ms")
        print(f"Throughput:      {res['throughput_fps']} images / second")
        print("=" * 70)
    except Exception as e:
        print(f"[ERROR] Benchmark failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
