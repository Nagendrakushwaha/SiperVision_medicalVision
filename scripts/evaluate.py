import sys
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.evaluation import evaluation_service

def main():
    parser = argparse.ArgumentParser(description="SmartMed Vision Model Evaluation CLI")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "mobilenet", "efficientnet"], help="Model architecture")
    parser.add_argument("--split", type=str, default="test", choices=["test", "validation"], help="Dataset split to evaluate on")
    parser.add_argument("--samples", type=int, default=200, help="Number of samples to evaluate")
    args = parser.parse_args()

    print("=" * 70)
    print(f" SmartMed Vision Evaluation: {args.model.upper()} on {args.split.upper()} split")
    print("=" * 70)

    try:
        results = evaluation_service.run_evaluation(
            model_name=args.model,
            split_name=args.split,
            max_samples=args.samples
        )
        metrics = results["metrics"]
        cm = results["confusion_matrix"]["raw"]
        specs = results["model_specs"]

        print(f"\n[EVALUATION METRICS (N={results['evaluated_samples']} samples)]")
        print(f"Accuracy:    {metrics['accuracy'] * 100:.2f}%")
        print(f"Precision:   {metrics['precision'] * 100:.2f}%")
        print(f"Recall:      {metrics['recall'] * 100:.2f}%")
        print(f"F1 Score:    {metrics['f1_score'] * 100:.2f}%")
        print(f"Sensitivity: {metrics['sensitivity'] * 100:.2f}%")
        print(f"Specificity: {metrics['specificity'] * 100:.2f}%")
        print(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:      {metrics['pr_auc']:.4f}")

        print(f"\n[CONFUSION MATRIX]")
        print(f"                Predicted Normal   Predicted Pneumonia")
        print(f"Actual Normal        {cm['tn']:<18} {cm['fp']}")
        print(f"Actual Pneumonia     {cm['fn']:<18} {cm['tp']}")

        print(f"\n[CPU LATENCY PROFILE]")
        print(f"Average Latency: {specs['avg_latency_ms']} ms")
        print(f"P50 Latency:     {specs['p50_latency_ms']} ms")
        print(f"P95 Latency:     {specs['p95_latency_ms']} ms")
        print(f"Parameters:      {specs['total_parameters']:,}")
        print(f"Model Size:      {specs['model_size_mb']} MB")
        print("=" * 70)

    except Exception as e:
        print(f"[ERROR] Evaluation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
