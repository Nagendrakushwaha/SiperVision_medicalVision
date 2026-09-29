import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class SmartMedMobileNetV3(nn.Module):
    """
    MobileNetV3-Small optimized for lightweight edge / CPU chest X-ray screening.
    - Backbone: ImageNet Pretrained MobileNetV3-Small
    - Output: 2 classes
    - Target Layer for Grad-CAM: features[-1]
    """
    def __init__(self, pretrained: bool = True, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)
        # MobileNetV3 classifier is Sequential(Linear, Hardswish, Dropout, Linear)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[2] = nn.Dropout(p=dropout)
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)
        self.target_layer = self.backbone.features[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_target_layer(self) -> nn.Module:
        return self.target_layer
