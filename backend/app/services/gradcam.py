import io
import base64
from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.app.models.model_factory import get_model_target_layer

class GradCAM:
    """
    Vanilla and Robust Grad-CAM implementation for Chest X-Ray Explainability.
    Works seamlessly across ResNet18, MobileNetV3, and EfficientNet-B0.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple where the first element is the gradient wrt output
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Computes 2D Grad-CAM heatmap normalized to [0, 1] for input_tensor [1, 3, H, W].
        """
        self.model.eval()
        self.model.zero_grad()

        input_tensor = input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture activations or gradients for Grad-CAM.")

        # Global average pooling of gradients: [1, C, H_feat, W_feat] -> [1, C, 1, 1]
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Weighted combination of forward activation maps
        cam = torch.sum(pooled_gradients * self.activations, dim=1, keepdim=True)

        # Apply ReLU to retain only positive influences
        cam = F.relu(cam)

        # Upsample to match original input size
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

def compute_gradcam_visualizations(
    model: nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    original_pil: Image.Image,
    target_class: Optional[int] = None,
    alpha: float = 0.45,
    colormap_name: str = "jet"
) -> Dict[str, Any]:
    """
    Computes Grad-CAM and returns:
    - heatmap array
    - overlay PIL Image
    - heatmap PIL Image
    - base64 strings for direct frontend display
    """
    target_layer = get_model_target_layer(model_name, model)
    cam_tool = GradCAM(model, target_layer)
    try:
        heatmap_2d = cam_tool.generate_heatmap(input_tensor, target_class=target_class)
    finally:
        cam_tool.remove_hooks()

    # Resize heatmap to match original PIL image size exactly
    orig_w, orig_h = original_pil.size
    heatmap_pil = Image.fromarray((heatmap_2d * 255).astype(np.uint8)).resize((orig_w, orig_h), resample=Image.BILINEAR)
    heatmap_resized = np.array(heatmap_pil) / 255.0

    # Apply colormap (jet)
    try:
        import matplotlib
        cmap = matplotlib.colormaps[colormap_name]
    except Exception:
        cmap = cm.get_cmap(colormap_name)
    color_heatmap = cmap(heatmap_resized)[:, :, :3]  # [H, W, 3] in [0, 1]
    color_heatmap_uint8 = (color_heatmap * 255).astype(np.uint8)
    heatmap_color_pil = Image.fromarray(color_heatmap_uint8)

    # Blend with original grayscale/RGB image
    orig_arr = np.array(original_pil.convert("RGB")).astype(np.float32)
    overlay_arr = (1 - alpha) * orig_arr + alpha * (color_heatmap * 255.0)
    overlay_arr = np.clip(overlay_arr, 0, 255).astype(np.uint8)
    overlay_pil = Image.fromarray(overlay_arr)

    # Encode to Base64
    def to_b64(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

    return {
        "heatmap_base64": to_b64(heatmap_color_pil),
        "overlay_base64": to_b64(overlay_pil),
        "original_base64": to_b64(original_pil),
        "target_class": int(target_class) if target_class is not None else None,
        "disclaimer": "Grad-CAM highlights image regions that contributed to the model prediction. It is an interpretability technique and does not establish clinical correctness."
    }
