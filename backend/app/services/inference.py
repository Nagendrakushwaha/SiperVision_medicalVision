import time
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Union, List
from PIL import Image
import torch
import torch.nn.functional as F

from backend.app.config import BASE_DIR, CHECKPOINTS_DIR, REPORTS_DIR
from backend.app.models.model_factory import (
    build_model,
    load_checkpoint,
    normalize_model_name,
    get_model_info
)
from backend.app.services.preprocessing import preprocess_image_for_inference
from backend.app.services.gradcam import compute_gradcam_visualizations
from backend.app.services.dataset_service import dataset_service
from backend.app.services.localization import draw_bounding_boxes, format_rsna_boxes

PREDICTION_HISTORY_FILE = REPORTS_DIR / "predictions" / "prediction_history.json"

class InferenceEngine:
    """
    Inference Engine for SmartMed Vision.
    Executes actual forward passes, measures CPU latency, generates Grad-CAM,
    and maintains local prediction audit history.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded_model = None
        self.current_model_name = None
        self.active_checkpoint_path = None

    def get_active_checkpoint(self, model_name: str = "resnet18") -> Optional[Path]:
        norm_name = normalize_model_name(model_name)
        ckpt_dir = CHECKPOINTS_DIR / norm_name
        best_ckpt = ckpt_dir / "best.pth"
        last_ckpt = ckpt_dir / "last.pth"
        if best_ckpt.exists():
            return best_ckpt
        if last_ckpt.exists():
            return last_ckpt
        return None

    def ensure_model_loaded(self, model_name: str = "resnet18") -> Tuple[torch.nn.Module, str]:
        norm_name = normalize_model_name(model_name)
        ckpt_path = self.get_active_checkpoint(norm_name)
        if not ckpt_path or not ckpt_path.exists():
            raise FileNotFoundError(
                f"No trained model available for {norm_name}. Please train a model before running inference."
            )

        if self.loaded_model is None or self.current_model_name != norm_name or self.active_checkpoint_path != ckpt_path:
            model = build_model(model_name=norm_name, pretrained=False, num_classes=2)
            load_checkpoint(model, ckpt_path, device=self.device)
            self.loaded_model = model
            self.current_model_name = norm_name
            self.active_checkpoint_path = ckpt_path

        return self.loaded_model, str(ckpt_path.relative_to(BASE_DIR))

    def predict(
        self,
        image_source: Union[Image.Image, bytes, Path, str],
        model_name: str = "resnet18",
        include_gradcam: bool = True,
        patient_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs actual model inference on provided chest X-ray.
        Calculates exact softmax probabilities and CPU latency.
        """
        model, ckpt_rel_path = self.ensure_model_loaded(model_name)

        # Preprocess image
        input_tensor, pil_img = preprocess_image_for_inference(image_source)
        input_tensor = input_tensor.to(self.device)

        # Measure exact inference time
        model.eval()
        start_t = time.perf_counter()
        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)

        p_normal = float(probs[0])
        p_pneumonia = float(probs[1])

        pred_class_idx = int(torch.argmax(logits, dim=1).item())
        pred_label = "Pneumonia" if pred_class_idx == 1 else "Normal / No Lung Opacity"

        result = {
            "prediction": pred_label,
            "target_class": pred_class_idx,
            "pneumonia_probability": round(p_pneumonia * 100.0, 2),
            "normal_probability": round(p_normal * 100.0, 2),
            "confidence": round(float(probs[pred_class_idx]) * 100.0, 2),
            "latency_ms": latency_ms,
            "model_name": self.current_model_name,
            "checkpoint": ckpt_rel_path,
            "image_size": f"{pil_img.size[0]}x{pil_img.size[1]}",
            "device": str(self.device).upper(),
            "timestamp": datetime.now().isoformat()
        }

        # Explainability: Grad-CAM
        if include_gradcam:
            try:
                cam_res = compute_gradcam_visualizations(
                    model=model,
                    model_name=self.current_model_name,
                    input_tensor=input_tensor.clone(),
                    original_pil=pil_img,
                    target_class=pred_class_idx
                )
                result["gradcam"] = cam_res
            except Exception as e:
                result["gradcam_error"] = str(e)

        # Ground-truth annotations if this image corresponds to an RSNA patient
        if patient_id:
            patient_info = dataset_service.get_patient(patient_id)
            if patient_info:
                boxes = patient_info.get("boxes", [])
                result["ground_truth"] = {
                    "patient_id": patient_id,
                    "target": patient_info.get("target"),
                    "label": "Pneumonia" if patient_info.get("target") == 1 else "Normal",
                    "box_count": len(boxes),
                    "boxes": boxes
                }
                if boxes:
                    _, bbox_b64 = draw_bounding_boxes(pil_img, boxes)
                    result["ground_truth"]["bounding_box_image_base64"] = bbox_b64

        # Record in local prediction history
        self._record_prediction(result, patient_id)

        return result

    def _record_prediction(self, res: Dict[str, Any], patient_id: Optional[str]):
        PREDICTION_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        history = []
        if PREDICTION_HISTORY_FILE.exists():
            try:
                with open(PREDICTION_HISTORY_FILE, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []

        identifier = patient_id if patient_id else f"IMG-{datetime.now().strftime('%H%M%S')}"

        record = {
            "id": f"PRED-{int(time.time() * 1000) % 1000000:06d}",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "model": res["model_name"],
            "prediction": res["prediction"],
            "probability": f"{res['confidence']}%",
            "pneumonia_prob": res["pneumonia_probability"],
            "latency_ms": res["latency_ms"],
            "image_identifier": identifier,
            "checkpoint": res["checkpoint"]
        }
        history.insert(0, record)
        # Keep last 100 predictions
        history = history[:100]

        with open(PREDICTION_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def get_prediction_history(self) -> List[Dict[str, Any]]:
        if PREDICTION_HISTORY_FILE.exists():
            try:
                with open(PREDICTION_HISTORY_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def clear_prediction_history(self):
        if PREDICTION_HISTORY_FILE.exists():
            with open(PREDICTION_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)

inference_engine = InferenceEngine()
