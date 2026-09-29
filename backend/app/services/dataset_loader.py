from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from PIL import Image

from backend.app.services.dataset_service import dataset_service
from backend.app.services.dicom_service import dicom_to_pil
from backend.app.services.preprocessing import get_train_transforms, get_val_transforms

class RSNAPneumoniaDataset(Dataset):
    """
    PyTorch Dataset for RSNA Pneumonia Detection Challenge.
    Loads DICOM images lazily by patientId to prevent RAM exhaustion.
    Optimized for CPU execution (Ryzen 5 5500U).
    """
    def __init__(
        self,
        patient_ids: List[str],
        transform=None,
        max_samples: Optional[int] = None
    ):
        self.transform = transform
        self.samples = []
        
        for pid in patient_ids:
            p_info = dataset_service.get_patient(pid)
            if p_info:
                dcm_path = dataset_service.get_patient_dcm_path(pid)
                if dcm_path and dcm_path.exists():
                    self.samples.append({
                        "patient_id": pid,
                        "target": int(p_info["target"]),
                        "path": dcm_path
                    })

        if max_samples and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.samples[idx]
        try:
            pil_img, _ = dicom_to_pil(item["path"])
        except Exception:
            # Fallback black image if corrupt
            pil_img = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            img_tensor = self.transform(pil_img)
        else:
            tf = get_val_transforms(224)
            img_tensor = tf(pil_img)

        return img_tensor, item["target"]

    def get_targets(self) -> List[int]:
        return [s["target"] for s in self.samples]

def compute_class_weights(targets: List[int]) -> torch.Tensor:
    """
    Computes inverse frequency class weights:
    weight[c] = total_samples / (num_classes * count[c])
    """
    targets_np = np.array(targets)
    count_0 = np.sum(targets_np == 0)
    count_1 = np.sum(targets_np == 1)
    total = len(targets_np)
    
    if count_0 == 0 or count_1 == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)

    w0 = total / (2.0 * count_0)
    w1 = total / (2.0 * count_1)
    return torch.tensor([w0, w1], dtype=torch.float32)

def create_dataloaders(
    batch_size: int = 8,
    image_size: int = 224,
    num_workers: int = 0,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    Creates train and validation DataLoaders using the pre-computed patient split.
    """
    split = dataset_service.get_split()
    train_pids = split["train"]
    val_pids = split["validation"]

    train_dataset = RSNAPneumoniaDataset(
        patient_ids=train_pids,
        transform=get_train_transforms(image_size),
        max_samples=max_train_samples
    )
    val_dataset = RSNAPneumoniaDataset(
        patient_ids=val_pids,
        transform=get_val_transforms(image_size),
        max_samples=max_val_samples
    )

    class_weights = compute_class_weights(train_dataset.get_targets())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader, class_weights
