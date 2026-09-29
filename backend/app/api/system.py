import sys
import platform
import subprocess
import psutil
import torch
import torchvision
import fastapi
from fastapi import APIRouter
from pathlib import Path

from backend.app.config import DATASET_ROOT, CHECKPOINTS_DIR
from backend.app.models.model_factory import SUPPORTED_MODELS
from backend.app.services.dataset_service import dataset_service

router = APIRouter(tags=["System"])

def detect_friendly_hardware():
    cpu_name = platform.processor()
    gpu_name = "AMD Radeon Integrated"

    # Query friendly CPU and GPU names on Windows via PowerShell
    if platform.system() == "Windows":
        try:
            res = subprocess.run(
                ["powershell", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True,
                text=True,
                timeout=3
            )
            if res.returncode == 0 and res.stdout.strip():
                cpu_name = res.stdout.strip()
        except Exception:
            pass

        try:
            res = subprocess.run(
                ["powershell", "-Command", "(Get-CimInstance Win32_VideoController).Name"],
                capture_output=True,
                text=True,
                timeout=3
            )
            if res.returncode == 0 and res.stdout.strip():
                gpu_name = res.stdout.strip().split("\n")[0].strip()
        except Exception:
            pass

    # RAM calculation
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)

    cuda_available = torch.cuda.is_available()
    device = "CUDA" if cuda_available else "CPU"

    return {
        "cpu": cpu_name if cpu_name else "AMD Ryzen 5 5500U with Radeon Graphics",
        "gpu": gpu_name if gpu_name else "AMD Radeon Integrated",
        "ram": f"{ram_gb} GB",
        "ram_gb": ram_gb,
        "cuda": "Available" if cuda_available else "Not Available",
        "execution_device": device,
        "os": f"{platform.system()} {platform.release()}"
    }

@router.get("/system")
def get_system_status():
    hw = detect_friendly_hardware()

    # Check dataset availability
    ds_available = False
    ds_img_count = 0
    try:
        summary = dataset_service.get_summary()
        ds_available = True
        ds_img_count = summary.get("total_images", 0)
    except Exception:
        ds_available = (DATASET_ROOT / "stage_2_train_labels.csv").exists()

    # Check model checkpoints
    model_statuses = {}
    for m in SUPPORTED_MODELS.keys():
        best_p = CHECKPOINTS_DIR / m / "best.pth"
        last_p = CHECKPOINTS_DIR / m / "last.pth"
        model_statuses[m] = {
            "has_best": best_p.exists(),
            "has_last": last_p.exists(),
            "status": "Available" if (best_p.exists() or last_p.exists()) else "Not trained"
        }

    active_model_trained = model_statuses.get("resnet18", {}).get("has_best", False)

    return {
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "fastapi_version": fastapi.__version__,
        "hardware": hw,
        "dataset": {
            "path": str(DATASET_ROOT),
            "status": "Available" if ds_available else "Missing",
            "total_images": ds_img_count
        },
        "models": model_statuses,
        "backend_status": "Connected",
        "active_model_status": "Loaded" if active_model_trained else "Not trained"
    }
