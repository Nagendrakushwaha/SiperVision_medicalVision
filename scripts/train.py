import sys
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.training_service import training_manager

def main():
    parser = argparse.ArgumentParser(description="SmartMed Vision Model Training CLI")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "mobilenet", "efficientnet"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (1-100)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=224, help="Input image resolution")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit training samples for testing")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Limit validation samples for testing")
    args = parser.parse_args()

    # Reject invalid epochs
    if args.epochs < 1 or args.epochs > 100:
        print(f"[ERROR] Epochs must be an integer between 1 and 100. Received: {args.epochs}")
        sys.exit(1)

    print("=" * 70)
    print(f" SmartMed Vision Training Pipeline")
    print(f" Model:       {args.model}")
    print(f" Epochs:      {args.epochs}")
    print(f" Batch Size:  {args.batch_size}")
    print(f" Learning Rate: {args.lr}")
    print(f" Device:      CPU (AMD Ryzen 5 5500U)")
    print("=" * 70)

    try:
        training_manager.start_training(
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            image_size=args.image_size,
            early_stopping_patience=args.patience,
            seed=args.seed,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples
        )

        last_epoch = 0
        last_step = 0
        while True:
            time.sleep(1)
            status = training_manager.get_status()
            curr_status = status["status"]
            
            if curr_status == "training" or curr_status == "validating":
                ep = status["epoch"]
                tot_ep = status["total_epochs"]
                st = status["step"]
                tot_st = status["total_steps"]
                loss = status["train_loss"]
                acc = status["train_acc"]
                eta = status["estimated_remaining_seconds"]
                print(f"\r[{curr_status.upper()}] Epoch {ep}/{tot_ep} | Step {st}/{tot_st} | Loss: {loss:.4f} | Acc: {acc:.1f}% | ETA: {eta:.0f}s", end="")
            elif curr_status == "saving":
                print("\n[SAVING CHECKPOINT] Updating best.pth and last.pth...")
            elif curr_status in ("completed", "stopped", "failed"):
                print(f"\n\nTraining finished with status: {curr_status.upper()}")
                if status.get("error"):
                    print(f"Error details: {status['error']}")
                else:
                    print(f"Training duration: {status['elapsed_seconds']}s")
                    print(f"Final Train Loss: {status['train_loss']}, Train Acc: {status['train_acc']}%")
                    print(f"Final Val Loss:   {status['val_loss']}, Val Acc:   {status['val_acc']}%")
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Stopping gracefully...")
        training_manager.stop_training()
    except Exception as e:
        print(f"\n[FATAL ERROR] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
