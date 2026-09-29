import io
import base64
from pathlib import Path
from typing import Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from backend.app.models.model_factory import (
    build_model,
    normalize_model_name,
    SUPPORTED_MODELS,
    load_checkpoint
)
from backend.app.config import CHECKPOINTS_DIR, BASE_DIR
from backend.app.services.dataset_service import dataset_service
from backend.app.services.dicom_service import dicom_to_pil
from backend.app.services.preprocessing import get_val_transforms

def _mat_to_base64_png(mat: np.ndarray) -> str:
    """Encodes BGR or RGB or Grayscale numpy matrix to base64 data URI."""
    if mat.ndim == 2:
        success, encoded = cv2.imencode(".png", mat)
    else:
        # Convert RGB to BGR for cv2 encoding
        bgr = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(".png", bgr)
    if not success:
        raise RuntimeError("Failed to encode image to PNG.")
    b64_str = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"

class VisualProcessingService:
    """
    Generates real-time visual feature decomposition showing:
    1. Input Radiograph
    2. CLAHE Density Enhancement
    3. Sobel Edge Gradient Magnitude
    4. Canny Edge Contours Overlaid on Anatomy
    5. Real Convolutional Feature Maps from Early Layers
    6. Deep Saliency / Attention Mapping
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cached_models = {}

    def _get_model(self, model_name: str) -> torch.nn.Module:
        norm_name = normalize_model_name(model_name)
        if norm_name not in self._cached_models:
            model = build_model(norm_name, pretrained=False, num_classes=2)
            ckpt_path = CHECKPOINTS_DIR / norm_name / "best.pth"
            if ckpt_path.exists():
                try:
                    load_checkpoint(model, ckpt_path, device=self.device)
                except Exception:
                    pass
            model.to(self.device)
            model.eval()
            self._cached_models[norm_name] = model
        return self._cached_models[norm_name]

    def extract_early_features(self, model_name: str, tensor: torch.Tensor) -> np.ndarray:
        """
        Extracts real activation maps from the model's first convolutional layer
        and compiles the top 8 filter responses into a 4x2 grid image.
        """
        norm_name = normalize_model_name(model_name)
        model = self._get_model(norm_name)

        with torch.no_grad():
            tensor = tensor.to(self.device)
            if norm_name == "resnet18":
                # ResNet-18 conv1 + bn1 + relu
                x = model.backbone.conv1(tensor)
                x = model.backbone.bn1(x)
                activations = F.relu(x)  # [1, 64, 112, 112]
            elif norm_name in ("mobilenet", "efficientnet"):
                # MobileNet/EfficientNet features[0]
                activations = model.backbone.features[0](tensor)  # [1, C, H, W]

        # Extract top 8 channels with highest variance (most active edge/texture filters)
        act = activations.squeeze(0).cpu().numpy()  # [C, H, W]
        variances = np.var(act, axis=(1, 2))
        top_indices = np.argsort(variances)[::-1][:8]

        maps = []
        for idx in top_indices:
            channel = act[idx]
            # Normalize channel to 0-255
            c_norm = (channel - channel.min()) / (channel.max() - channel.min() + 1e-7)
            c_uint8 = (c_norm * 255.0).astype(np.uint8)
            # Resize for crisp display
            c_resized = cv2.resize(c_uint8, (140, 140), interpolation=cv2.INTER_LINEAR)
            # Colorize with MAGMA or VIRIDIS for striking visibility
            color_map = cv2.applyColorMap(c_resized, cv2.COLORMAP_VIRIDIS)
            # Convert BGR from colormap to RGB
            color_rgb = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
            maps.append(color_rgb)

        # Pad to 8 if fewer
        while len(maps) < 8:
            maps.append(np.zeros((140, 140, 3), dtype=np.uint8))

        # Stitch into a 2 rows x 4 cols grid
        row1 = np.hstack(maps[:4])
        row2 = np.hstack(maps[4:8])
        grid = np.vstack([row1, row2])
        return grid

    def get_pipeline(
        self,
        model_name: str = "resnet18",
        patient_id: Optional[str] = None,
        target: Optional[int] = None
    ) -> Dict[str, Any]:
        norm_name = normalize_model_name(model_name)

        # 1. Resolve Patient
        if not patient_id:
            sample = dataset_service.get_random_sample(target=target, has_bbox=(target == 1))
            if not sample:
                sample = dataset_service.get_random_sample()
            patient_id = sample["patient_id"]
        
        patient_info = dataset_service.get_patient(patient_id)
        if not patient_info:
            raise FileNotFoundError(f"Patient ID {patient_id} not found in dataset manifest.")

        dcm_path = dataset_service.get_patient_dcm_path(patient_id)
        if not dcm_path or not dcm_path.exists():
            raise FileNotFoundError(f"DICOM file for patient {patient_id} not found on disk.")

        pil_img, _ = dicom_to_pil(dcm_path)
        img_resized = pil_img.resize((224, 224), Image.BILINEAR)
        rgb_arr = np.array(img_resized)
        gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)

        # 2. Stage 1: CLAHE Contrast & Tissue Enhancement
        clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
        clahe_enhanced = clahe.apply(gray)
        clahe_rgb = cv2.cvtColor(clahe_enhanced, cv2.COLOR_GRAY2RGB)

        # 3. Stage 2: Sobel Edge Gradient Magnitude (Horizontal & Vertical spatial derivatives)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        magnitude_norm = np.clip(magnitude / (magnitude.max() + 1e-7) * 255.0, 0, 255).astype(np.uint8)
        
        # Colorize Sobel magnitude with cyan-focused palette
        sobel_colored = cv2.applyColorMap(magnitude_norm, cv2.COLORMAP_OCEAN)
        sobel_colored_rgb = cv2.cvtColor(sobel_colored, cv2.COLOR_BGR2RGB)

        # 4. Stage 3: Canny Anatomical Contours Overlaid on Anatomy
        canny_edges = cv2.Canny(clahe_enhanced, 40, 120)
        canny_overlay = rgb_arr.copy()
        # Highlight edges in bright luminous cyan (#06B6D4)
        canny_overlay[canny_edges > 0] = [6, 182, 212]

        # 5. Stage 4: Early Convolutional Neural Network Feature Maps
        tensor = get_val_transforms(224)(pil_img).unsqueeze(0)
        early_features_grid = self.extract_early_features(norm_name, tensor)

        # 6. Stage 5: Deep Saliency & Attribution (Grad-CAM or deep activations)
        model = self._get_model(norm_name)
        with torch.no_grad():
            tensor_dev = tensor.to(self.device)
            logits = model(tensor_dev)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
            pred_class = int(np.argmax(probs))
            pneumonia_prob = float(probs[1])

        # Deep layer heatmap
        try:
            from backend.app.services.explainability import generate_gradcam
            from backend.app.models.model_factory import get_model_target_layer
            target_layer = get_model_target_layer(norm_name, model)
            cam_heatmap, cam_overlay = generate_gradcam(model, target_layer, tensor_dev, pil_img)
            heatmap_b64 = _mat_to_base64_png(np.array(cam_heatmap))
            overlay_b64 = _mat_to_base64_png(np.array(cam_overlay))
        except Exception:
            heatmap_b64 = None
            overlay_b64 = None

        return {
            "patient_id": patient_id,
            "target": patient_info.get("target"),
            "label": "Pneumonia (Positive)" if patient_info.get("target") == 1 else "Normal / No Opacity",
            "detailed_class": patient_info.get("detailed_class"),
            "boxes": patient_info.get("boxes", []),
            "model_name": norm_name,
            "model_display_name": SUPPORTED_MODELS[norm_name]["name"],
            "prediction": {
                "predicted_class": pred_class,
                "label": "Pneumonia" if pred_class == 1 else "Normal",
                "pneumonia_probability": round(pneumonia_prob, 4),
                "normal_probability": round(float(probs[0]), 4)
            },
            "images": {
                "original": _mat_to_base64_png(rgb_arr),
                "clahe": _mat_to_base64_png(clahe_rgb),
                "sobel_edges": _mat_to_base64_png(sobel_colored_rgb),
                "canny_overlay": _mat_to_base64_png(canny_overlay),
                "conv_features": _mat_to_base64_png(early_features_grid),
                "saliency_heatmap": heatmap_b64,
                "saliency_overlay": overlay_b64
            },
            "stages": [
                {
                    "step": 1,
                    "title": "Input DICOM Radiograph",
                    "subtitle": "224×224 Tensor Input with Zero-Centered ImageNet Normalization",
                    "desc": "The raw radiograph undergoes VOI-LUT windowing and normalization to standardize pixel intensities across diverse hospital imaging sensors."
                },
                {
                    "step": 2,
                    "title": "Tissue Contrast Enhancement (CLAHE)",
                    "subtitle": "Contrast-Limited Adaptive Histogram Equalization",
                    "desc": "Local contrast amplification reveals subtle ground-glass opacities, alveolar infiltrates, and vascular markings that are otherwise faint in raw exposures."
                },
                {
                    "step": 3,
                    "title": "Spatial Edge & Contour Detection",
                    "subtitle": "Sobel Gradient Magnitude ∇I = (∂I/∂x, ∂I/∂y)",
                    "desc": "Detects high-frequency structural boundaries: diaphragm domes, costophrenic angles, bronchial margins, and fluid consolidation perimeters."
                },
                {
                    "step": 4,
                    "title": "Anatomical Boundary Overlay",
                    "subtitle": "Canny Edge Extraction Mapped onto Lung Parenchyma",
                    "desc": "Visualizes the structural boundaries that convolutional kernels scan to distinguish normal anatomical airspaces from pathological consolidation boundaries."
                },
                {
                    "step": 5,
                    "title": "Early Convolutional Feature Activations",
                    "subtitle": f"{SUPPORTED_MODELS[norm_name]['name']} Conv1 Neural Filter Responses",
                    "desc": "Actual activation maps from the network's first convolutional layer. Each tile shows an individual learned directional edge, ridge, or frequency filter firing on the radiograph."
                },
                {
                    "step": 6,
                    "title": "Deep Saliency & Pathology Focus",
                    "subtitle": "High-Level Attribution & Receptive Field Aggregation",
                    "desc": "Deep layers pool intermediate edge features into semantic representations, focusing receptive fields on dense fluid consolidations indicative of pneumonia."
                }
            ]
        }

visual_processing_service = VisualProcessingService()
