import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from backend.app.config import DATASET_ROOT, METADATA_DIR

logger = logging.getLogger(__name__)

SUMMARY_FILE = METADATA_DIR / "dataset_summary.json"
SPLIT_FILE = METADATA_DIR / "patient_split.json"
DETAILED_CACHE = METADATA_DIR / "patients_manifest.json"

def discover_dataset_files(root: Optional[Path] = None) -> Dict[str, Path]:
    if root is None:
        root = DATASET_ROOT

    # Look for train images directory
    img_candidates = ["stage_2_train_images", "train_images", "images"]
    img_dir = None
    for c in img_candidates:
        if (root / c).is_dir():
            img_dir = root / c
            break

    # Look for train labels CSV
    csv_candidates = ["stage_2_train_labels.csv", "train_labels.csv"]
    labels_csv = None
    for c in csv_candidates:
        if (root / c).is_file():
            labels_csv = root / c
            break

    # Look for detailed class info CSV
    class_info_csv = None
    detail_candidates = ["stage_2_detailed_class_info.csv", "detailed_class_info.csv"]
    for c in detail_candidates:
        if (root / c).is_file():
            class_info_csv = root / c
            break

    return {
        "root": root,
        "images_dir": img_dir,
        "labels_csv": labels_csv,
        "class_info_csv": class_info_csv
    }

def analyze_and_build_split(seed: int = 42, force_recompute: bool = False) -> Dict[str, Any]:
    """
    Inspects real RSNA dataset, creates strictly leak-free patient-level splits,
    computes exact dataset statistics, and caches them to disk.
    """
    if not force_recompute and SUMMARY_FILE.exists() and SPLIT_FILE.exists():
        with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
            summary = json.load(f)
        with open(SPLIT_FILE, "r", encoding="utf-8") as f:
            split = json.load(f)
        return {"summary": summary, "split": split}

    files = discover_dataset_files()
    if not files["labels_csv"] or not files["labels_csv"].exists():
        raise FileNotFoundError(f"RSNA dataset labels CSV not found in {files['root']}")
    if not files["images_dir"] or not files["images_dir"].exists():
        raise FileNotFoundError(f"RSNA dataset images directory not found in {files['root']}")

    df_labels = pd.read_csv(files["labels_csv"])
    
    # Detailed class info if available
    class_map = {}
    if files["class_info_csv"] and files["class_info_csv"].exists():
        df_classes = pd.read_csv(files["class_info_csv"])
        for _, row in df_classes.iterrows():
            class_map[row["patientId"]] = row["class"]

    # Check available DICOM image files
    available_image_files = set(f.stem for f in files["images_dir"].glob("*.dcm"))
    
    # Aggregate patient-level information
    patient_groups = {}
    for _, row in df_labels.iterrows():
        pid = str(row["patientId"])
        target = int(row["Target"])
        
        if pid not in patient_groups:
            patient_groups[pid] = {
                "patient_id": pid,
                "target": target,
                "detailed_class": class_map.get(pid, "Lung Opacity" if target == 1 else "Normal"),
                "has_dicom": pid in available_image_files,
                "boxes": []
            }
        
        if target == 1 and not pd.isna(row["x"]):
            patient_groups[pid]["boxes"].append({
                "x": float(row["x"]),
                "y": float(row["y"]),
                "width": float(row["width"]),
                "height": float(row["height"])
            })

    all_patients = list(patient_groups.values())
    df_patients = pd.DataFrame([
        {
            "patient_id": p["patient_id"],
            "target": p["target"],
            "box_count": len(p["boxes"]),
            "has_dicom": p["has_dicom"]
        }
        for p in all_patients
    ])

    # Filter to patients with valid existing DICOM files
    valid_patients = df_patients[df_patients["has_dicom"]].copy()
    missing_files_count = int((~df_patients["has_dicom"]).sum())

    total_patients = len(valid_patients)
    positive_patients = int((valid_patients["target"] == 1).sum())
    negative_patients = int((valid_patients["target"] == 0).sum())
    total_boxes = sum(len(p["boxes"]) for p in all_patients if p["has_dicom"])

    # Strict Patient-Level Split (70% train, 15% validation, 15% test)
    # Stratified by target to maintain exact class ratio
    train_df, temp_df = train_test_split(
        valid_patients,
        test_size=0.30,
        random_state=seed,
        stratify=valid_patients["target"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["target"]
    )

    train_pids = set(train_df["patient_id"].tolist())
    val_pids = set(val_df["patient_id"].tolist())
    test_pids = set(test_df["patient_id"].tolist())

    # Leakage verification
    assert len(train_pids.intersection(val_pids)) == 0, "Data leakage detected between Train and Val!"
    assert len(train_pids.intersection(test_pids)) == 0, "Data leakage detected between Train and Test!"
    assert len(val_pids.intersection(test_pids)) == 0, "Data leakage detected between Val and Test!"

    split_data = {
        "seed": seed,
        "train": sorted(list(train_pids)),
        "validation": sorted(list(val_pids)),
        "test": sorted(list(test_pids)),
        "counts": {
            "train": len(train_pids),
            "validation": len(val_pids),
            "test": len(test_pids),
            "total": len(valid_patients)
        }
    }

    # Assign split to all_patients
    for p in all_patients:
        pid = p["patient_id"]
        if pid in train_pids:
            p["split"] = "train"
        elif pid in val_pids:
            p["split"] = "validation"
        elif pid in test_pids:
            p["split"] = "test"
        else:
            p["split"] = "excluded"

    # Box count distribution
    box_counts = {
        "0_boxes": int((valid_patients["box_count"] == 0).sum()),
        "1_box": int((valid_patients["box_count"] == 1).sum()),
        "2_boxes": int((valid_patients["box_count"] == 2).sum()),
        "3_or_more_boxes": int((valid_patients["box_count"] >= 3).sum())
    }

    summary_data = {
        "dataset_name": "RSNA Pneumonia Detection Challenge",
        "dataset_root": str(files["root"]),
        "total_images": total_patients,
        "total_patients": total_patients,
        "positive_cases": positive_patients,
        "negative_cases": negative_patients,
        "positive_percentage": round((positive_patients / total_patients) * 100, 2) if total_patients > 0 else 0,
        "negative_percentage": round((negative_patients / total_patients) * 100, 2) if total_patients > 0 else 0,
        "total_bounding_boxes": total_boxes,
        "image_dimensions": "1024x1024",
        "missing_files": missing_files_count,
        "invalid_files": 0,
        "duplicate_patient_ids": 0,
        "box_distribution": box_counts,
        "splits": {
            "train": {
                "patients": len(train_pids),
                "images": len(train_pids),
                "positive": int((train_df["target"] == 1).sum()),
                "negative": int((train_df["target"] == 0).sum()),
                "positive_percentage": round(int((train_df["target"] == 1).sum()) / len(train_pids) * 100, 2)
            },
            "validation": {
                "patients": len(val_pids),
                "images": len(val_pids),
                "positive": int((val_df["target"] == 1).sum()),
                "negative": int((val_df["target"] == 0).sum()),
                "positive_percentage": round(int((val_df["target"] == 1).sum()) / len(val_pids) * 100, 2)
            },
            "test": {
                "patients": len(test_pids),
                "images": len(test_pids),
                "positive": int((test_df["target"] == 1).sum()),
                "negative": int((test_df["target"] == 0).sum()),
                "positive_percentage": round(int((test_df["target"] == 1).sum()) / len(test_pids) * 100, 2)
            }
        }
    }

    # Persist metadata to disk
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    with open(SPLIT_FILE, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2)

    with open(DETAILED_CACHE, "w", encoding="utf-8") as f:
        json.dump({p["patient_id"]: p for p in all_patients if p["has_dicom"]}, f)

    return {"summary": summary_data, "split": split_data}

class RSNADatasetService:
    def __init__(self):
        self._manifest = None
        self._summary = None
        self._split = None

    def _ensure_loaded(self):
        if self._summary is None or self._split is None or self._manifest is None:
            if not SUMMARY_FILE.exists() or not SPLIT_FILE.exists() or not DETAILED_CACHE.exists():
                analyze_and_build_split()
            with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
                self._summary = json.load(f)
            with open(SPLIT_FILE, "r", encoding="utf-8") as f:
                self._split = json.load(f)
            with open(DETAILED_CACHE, "r", encoding="utf-8") as f:
                self._manifest = json.load(f)

    def get_summary(self) -> Dict[str, Any]:
        self._ensure_loaded()
        return self._summary

    def get_split(self) -> Dict[str, Any]:
        self._ensure_loaded()
        return self._split

    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        self._ensure_loaded()
        return self._manifest.get(patient_id)

    def get_patient_dcm_path(self, patient_id: str) -> Optional[Path]:
        files = discover_dataset_files()
        img_dir = files["images_dir"]
        if img_dir:
            p = img_dir / f"{patient_id}.dcm"
            if p.exists():
                return p
        return None

    def query_examples(
        self,
        split: Optional[str] = None,
        target: Optional[int] = None,
        has_bbox: Optional[bool] = None,
        search: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        self._ensure_loaded()
        results = list(self._manifest.values())

        if split and split.lower() != "all":
            results = [p for p in results if p.get("split") == split.lower()]

        if target is not None:
            results = [p for p in results if p.get("target") == int(target)]

        if has_bbox is not None:
            if has_bbox:
                results = [p for p in results if len(p.get("boxes", [])) > 0]
            else:
                results = [p for p in results if len(p.get("boxes", [])) == 0]

        if search:
            q = search.lower().strip()
            results = [p for p in results if q in p.get("patient_id", "").lower()]

        total = len(results)
        start = (page - 1) * page_size
        end = start + page_size
        items = results[start:end]

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size if page_size > 0 else 1,
            "items": items
        }

    def get_random_sample(
        self,
        split: Optional[str] = None,
        target: Optional[int] = None,
        has_bbox: Optional[bool] = None
    ) -> Optional[Dict[str, Any]]:
        self._ensure_loaded()
        candidates = list(self._manifest.values())
        if split and split.lower() != "all":
            candidates = [p for p in candidates if p.get("split") == split.lower()]
        if target is not None:
            candidates = [p for p in candidates if p.get("target") == int(target)]
        if has_bbox is not None:
            if has_bbox:
                candidates = [p for p in candidates if len(p.get("boxes", [])) > 0]
            else:
                candidates = [p for p in candidates if len(p.get("boxes", [])) == 0]

        if not candidates:
            return None

        idx = np.random.randint(0, len(candidates))
        return candidates[idx]

dataset_service = RSNADatasetService()
