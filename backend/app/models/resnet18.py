import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class SmartMedResNet18(nn.Module):
    """
    ResNet-18 adapted for Explainable Binary Pneumonia Detection.
    - Backbone: ImageNet Pretrained ResNet-18
    - Output: 2 classes [0: Normal / No Lung Opacity, 1: Pneumonia / Lung Opacity]
    - Target Layer for Grad-CAM: layer4[-1]
    """
    def __init__(self, pretrained: bool = True, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        self.target_layer = self.backbone.layer4[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_target_layer(self) -> nn.Module:
        return self.target_layer
