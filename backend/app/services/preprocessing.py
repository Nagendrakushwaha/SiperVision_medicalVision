import io
from pathlib import Path
from typing import Tuple, Union, Optional
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# ImageNet standard normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transforms(image_size: int = 224):
    """
    Medically conservative data augmentation:
    - Small rotation (up to 7 degrees)
    - Small translation (up to 5%)
    - Controlled scaling (0.95 - 1.05)
    - Mild brightness and contrast (±10%)
    - Resize to target image_size
    - Convert to Tensor and ImageNet normalization
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomAffine(
            degrees=(-7, 7),
            translate=(0.04, 0.04),
            scale=(0.96, 1.04)
        ),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def get_val_transforms(image_size: int = 224):
    """
    Deterministic validation/inference transform:
    - Resize to target image_size
    - Convert to Tensor
    - ImageNet normalization
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def preprocess_image_for_inference(
    image_source: Union[Image.Image, np.ndarray, bytes, Path, str],
    image_size: int = 224
) -> Tuple[torch.Tensor, Image.Image]:
    """
    Converts any image source (DICOM, PNG, JPG, PIL, or numpy array)
    into:
    1. model input tensor of shape [1, 3, image_size, image_size]
    2. clean PIL RGB image for web visualization
    """
    from backend.app.services.dicom_service import dicom_to_pil

    if isinstance(image_source, Image.Image):
        pil_img = image_source.convert("RGB")
    elif isinstance(image_source, np.ndarray):
        if image_source.dtype != np.uint8:
            arr_norm = ((image_source - image_source.min()) / (image_source.max() - image_source.min() + 1e-7) * 255.0).astype(np.uint8)
        else:
            arr_norm = image_source
        pil_img = Image.fromarray(arr_norm).convert("RGB")
    elif isinstance(image_source, (str, Path)):
        p = Path(image_source)
        if p.suffix.lower() == ".dcm":
            pil_img, _ = dicom_to_pil(p)
        else:
            pil_img = Image.open(p).convert("RGB")
    elif isinstance(image_source, bytes):
        # Determine if it's DICOM or standard image
        # DICOM usually has 'DICM' at byte offset 128 or pydicom can parse it
        try:
            pil_img, _ = dicom_to_pil(image_source)
        except Exception:
            try:
                pil_img = Image.open(io.BytesIO(image_source)).convert("RGB")
            except Exception as e:
                raise ValueError(f"Unsupported or corrupted image file format: {str(e)}")
    else:
        raise TypeError(f"Unsupported image source type: {type(image_source)}")

    val_tf = get_val_transforms(image_size)
    tensor = val_tf(pil_img).unsqueeze(0)  # Shape: [1, 3, image_size, image_size]
    return tensor, pil_img
