import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.dataset_service import analyze_and_build_split

def main():
    print("=" * 70)
    print(" SmartMed Vision: RSNA Dataset Inspection & Patient-Level Splitting")
    print("=" * 70)
    print("Inspecting RSNA Pneumonia Detection Challenge files...")
    
    res = analyze_and_build_split(force_recompute=True)
    summary = res["summary"]
    splits = summary["splits"]

    print("\n[DATASET VERIFICATION COMPLETE]")
    print(f"Dataset Root:             {summary['dataset_root']}")
    print(f"Total DICOM Images:       {summary['total_images']:,}")
    print(f"Total Unique Patients:    {summary['total_patients']:,}")
    print(f"Pneumonia Cases (Target=1): {summary['positive_cases']:,} ({summary['positive_percentage']}%)")
    print(f"Normal Cases (Target=0):    {summary['negative_cases']:,} ({summary['negative_percentage']}%)")
    print(f"Total Bounding Boxes:     {summary['total_bounding_boxes']:,}")
    print(f"Image Dimensions:         {summary['image_dimensions']}")
    print(f"Missing / Corrupt Files:  {summary['missing_files']}")

    print("\n[PATIENT-LEVEL LEAK-FREE SPLITS]")
    print(f"Train Set:      {splits['train']['patients']:,} patients ({splits['train']['positive']:,} pneumonia, {splits['train']['negative']:,} normal)")
    print(f"Validation Set: {splits['validation']['patients']:,} patients ({splits['validation']['positive']:,} pneumonia, {splits['validation']['negative']:,} normal)")
    print(f"Test Set:       {splits['test']['patients']:,} patients ({splits['test']['positive']:,} pneumonia, {splits['test']['negative']:,} normal)")

    print("\nMetadata successfully saved to:")
    print(" - data/metadata/dataset_summary.json")
    print(" - data/metadata/patient_split.json")
    print("=" * 70)

if __name__ == "__main__":
    main()
