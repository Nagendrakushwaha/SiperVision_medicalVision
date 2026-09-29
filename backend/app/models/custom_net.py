import torch
import torch.nn as nn
from typing import Optional

def get_activation_fn(name: str) -> nn.Module:
    name = (name or "relu").lower().replace("-", "_")
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif name == "gelu":
        return nn.GELU()
    elif name in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    elif name == "tanh":
        return nn.Tanh()
    return nn.ReLU(inplace=True)

class SmartMedCustomNet(nn.Module):
    """
    Configurable Deep Neural Network architecture supporting 1 to 100 hidden layers.
    Combines a multi-stage convolutional feature extractor stem with a deep MLP head.
    """
    def __init__(
        self,
        num_classes: int = 2,
        hidden_layers: int = 2,
        hidden_units: int = 256,
        activation: str = "relu",
        dropout: float = 0.2,
        pretrained: bool = False
    ):
        super().__init__()
        self.num_hidden_layers = max(1, min(100, int(hidden_layers)))
        self.hidden_units = int(hidden_units)
        self.activation_name = activation
        act_fn = get_activation_fn(activation)
        
        # Convolutional Feature Stem (downsamples 224x224 to feature vectors)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            act_fn,
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            act_fn,
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            act_fn,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.target_layer = self.stem[8]  # Target layer for Grad-CAM
        
        # Deep Neural Network Head with 1 to 100 Hidden Layers
        layers = []
        # First projection: 128 -> hidden_units
        layers.append(nn.Linear(128, self.hidden_units))
        layers.append(nn.BatchNorm1d(self.hidden_units))
        layers.append(get_activation_fn(activation))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
            
        # Subsequent hidden layers (up to 100 layers)
        for _ in range(self.num_hidden_layers - 1):
            layers.append(nn.Linear(self.hidden_units, self.hidden_units))
            layers.append(nn.BatchNorm1d(self.hidden_units))
            layers.append(get_activation_fn(activation))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
                
        # Final classification output
        layers.append(nn.Linear(self.hidden_units, num_classes))
        self.mlp_head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        feat_flat = torch.flatten(feat, 1)
        return self.mlp_head(feat_flat)

    def get_target_layer(self) -> nn.Module:
        return self.target_layer
