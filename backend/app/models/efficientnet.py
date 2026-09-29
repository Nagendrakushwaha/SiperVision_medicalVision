import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class SmartMedEfficientNetB0(nn.Module):
    """
    EfficientNet-B0 adapted for Explainable Binary Pneumonia Detection.
    - Backbone: ImageNet Pretrained EfficientNet-B0
    - Output: 2 classes
    - Target Layer for Grad-CAM: features[-1]
    """
    def __init__(self, pretrained: bool = True, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        # EfficientNet classifier is Sequential(Dropout, Linear)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[0] = nn.Dropout(p=dropout)
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        self.target_layer = self.backbone.features[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_target_layer(self) -> nn.Module:
        return self.target_layer
