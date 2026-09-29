import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.dataset_service import dataset_service

def main():
    print("=" * 70)
    print(" SmartMed Vision: Dataset Analysis & Distribution Report")
    print("=" * 70)
    
    summary = dataset_service.get_summary()
    split = dataset_service.get_split()

    print(json.dumps(summary, indent=2))
    print("\n[LEAKAGE AUDIT]")
    train_set = set(split["train"])
    val_set = set(split["validation"])
    test_set = set(split["test"])
    
    print(f"Overlap Train & Val:  {len(train_set.intersection(val_set))} patients (MUST BE 0)")
    print(f"Overlap Train & Test: {len(train_set.intersection(test_set))} patients (MUST BE 0)")
    print(f"Overlap Val & Test:   {len(val_set.intersection(test_set))} patients (MUST BE 0)")
    print("=" * 70)

if __name__ == "__main__":
    main()
