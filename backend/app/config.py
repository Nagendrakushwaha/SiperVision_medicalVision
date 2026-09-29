import os
import yaml
from pathlib import Path

# Base project path (root of SmartVision)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"

def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

CONFIG = load_config()

def get_dataset_root() -> Path:
    """
    Find the RSNA dataset root automatically:
    1. Check config.yaml 'dataset.root'
    2. Check config.yaml 'dataset.alt_root' (e.g. F:/SmartMedVision/data/rsna)
    3. Auto-discover inside workspace (e.g. rsna-pneumonia-detection-challenge)
    """
    cfg_root = CONFIG.get("dataset", {}).get("root")
    if cfg_root:
        p = Path(cfg_root)
        if not p.is_absolute():
            p = BASE_DIR / p
        if p.exists() and (p / "stage_2_train_labels.csv").exists() or (p / "stage_2_train_images").exists():
            return p

    alt_root = CONFIG.get("dataset", {}).get("alt_root")
    if alt_root:
        p = Path(alt_root)
        if p.exists():
            return p

    # Candidate directories
    candidates = [
        BASE_DIR / "rsna-pneumonia-detection-challenge",
        BASE_DIR / "data" / "rsna",
        BASE_DIR / "data" / "raw",
        Path("F:/SmartMedVision/data/rsna"),
    ]
    for c in candidates:
        if c.exists() and (c / "stage_2_train_labels.csv").exists():
            return c

    # Fallback to local rsna folder or relative path
    return BASE_DIR / "rsna-pneumonia-detection-challenge"

DATASET_ROOT = get_dataset_root()
CHECKPOINTS_DIR = BASE_DIR / CONFIG.get("paths", {}).get("checkpoints_dir", "models/checkpoints")
REPORTS_DIR = BASE_DIR / CONFIG.get("paths", {}).get("reports_dir", "reports")
EXPERIMENTS_DIR = BASE_DIR / CONFIG.get("paths", {}).get("experiments_dir", "experiments")
METADATA_DIR = BASE_DIR / CONFIG.get("dataset", {}).get("metadata_dir", "data/metadata")

# Ensure required directories exist
for d in [CHECKPOINTS_DIR, REPORTS_DIR, EXPERIMENTS_DIR, METADATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)
