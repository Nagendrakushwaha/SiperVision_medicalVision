import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import torch
import torch.nn as nn

from backend.app.models.resnet18 import SmartMedResNet18
from backend.app.models.mobilenet import SmartMedMobileNetV3
from backend.app.models.efficientnet import SmartMedEfficientNetB0

SUPPORTED_MODELS = {
    "resnet18": {
        "class": SmartMedResNet18,
        "name": "ResNet18",
        "description": "Residual Network 18 layers (Default standard for medical imaging)",
        "default": True
    },
    "mobilenet": {
        "class": SmartMedMobileNetV3,
        "name": "MobileNetV3-Small",
        "description": "Ultra-lightweight edge architecture with hard-swish activation",
        "default": False
    },
    "efficientnet": {
        "class": SmartMedEfficientNetB0,
        "name": "EfficientNet-B0",
        "description": "Compound scaled convolutional network with MBConv blocks",
        "default": False
    }
}

def normalize_model_name(model_name: str) -> str:
    key = model_name.lower().replace("-", "").replace("_", "")
    if "resnet" in key:
        return "resnet18"
    elif "mobile" in key:
        return "mobilenet"
    elif "efficient" in key:
        return "efficientnet"
    raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(SUPPORTED_MODELS.keys())}")

def build_model(model_name: str = "resnet18", pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    key = normalize_model_name(model_name)
    model_cls = SUPPORTED_MODELS[key]["class"]
    return model_cls(pretrained=pretrained, num_classes=num_classes)

def get_model_target_layer(model_name: str, model: nn.Module) -> nn.Module:
    if hasattr(model, "get_target_layer"):
        return model.get_target_layer()
    key = normalize_model_name(model_name)
    if key == "resnet18":
        return model.backbone.layer4[-1]
    elif key in ("mobilenet", "efficientnet"):
        return model.backbone.features[-1]
    raise ValueError(f"Unknown target layer for model: {model_name}")

def count_parameters(model: nn.Module) -> Dict[str, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params
    }

def get_model_size_mb(model: nn.Module) -> float:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return round((param_size + buffer_size) / (1024 * 1024), 2)

def get_model_info(model_name: str, model: Optional[nn.Module] = None) -> Dict[str, Any]:
    key = normalize_model_name(model_name)
    info = {
        "id": key,
        "name": SUPPORTED_MODELS[key]["name"],
        "description": SUPPORTED_MODELS[key]["description"],
        "default": SUPPORTED_MODELS[key]["default"],
    }
    if model is not None:
        params = count_parameters(model)
        info.update(params)
        info["size_mb"] = get_model_size_mb(model)
    return info

def save_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    epoch: int,
    metrics: Dict[str, Any],
    optimizer_state: Optional[Dict[str, Any]] = None,
    scheduler_state: Optional[Dict[str, Any]] = None
):
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "optimizer_state_dict": optimizer_state,
        "scheduler_state_dict": scheduler_state
    }
    torch.save(state, str(path))

def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    device: torch.device
) -> Dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")
    checkpoint = torch.load(str(path), map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return checkpoint
