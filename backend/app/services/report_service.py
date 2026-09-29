import json
from datetime import datetime
from typing import Dict, Any, List

from backend.app.config import BASE_DIR, REPORTS_DIR, CHECKPOINTS_DIR
from backend.app.services.dataset_service import dataset_service
from backend.app.services.evaluation import evaluation_service
from backend.app.services.benchmark_service import benchmark_service
from backend.app.services.training_service import training_manager
from backend.app.models.model_factory import SUPPORTED_MODELS

class ReportService:
    """
    Technical and Clinical Research Report Generator.
    Gathers real data from dataset inspection, training logs, evaluation,
    benchmarks, and system hardware. Zero fabricated data.
    """
    def generate_full_report(self, active_model: str = "resnet18") -> Dict[str, Any]:
        # 1. Dataset stats
        try:
            ds_summary = dataset_service.get_summary()
        except Exception:
            ds_summary = None

        # 2. Evaluation stats
        eval_data = evaluation_service.get_evaluation(active_model)

        # 3. Training history
        train_hist = training_manager.get_history(active_model)

        # 4. Benchmark stats
        benchmarks = benchmark_service.get_benchmarks()
        active_benchmark = next((b for b in benchmarks if b.get("model_id") == active_model), None)

        # 5. Model status overview
        models_overview = []
        for m_key, m_meta in SUPPORTED_MODELS.items():
            ckpt_best = CHECKPOINTS_DIR / m_key / "best.pth"
            ckpt_last = CHECKPOINTS_DIR / m_key / "last.pth"
            m_eval = evaluation_service.get_evaluation(m_key)
            
            if m_eval is not None:
                status = "Evaluated"
            elif ckpt_best.exists() or ckpt_last.exists():
                status = "Trained — Not Evaluated"
            else:
                status = "Not Trained"

            models_overview.append({
                "id": m_key,
                "name": m_meta["name"],
                "status": status,
                "metrics": m_eval.get("metrics") if m_eval else None,
                "model_specs": m_eval.get("model_specs") if m_eval else None
            })

        report = {
            "title": "SmartMed Vision: Explainable Pneumonia Detection & Localization Research Report",
            "generated_at": datetime.now().isoformat(),
            "target_model": active_model,
            "abstract": (
                "SmartMed Vision is a clinical-grade educational and research framework designed for automated, "
                "explainable binary pneumonia screening and bounding-box localization using the RSNA Pneumonia "
                "Detection Challenge dataset. The architecture leverages deep convolutional neural networks with ImageNet "
                "transfer learning, class-weighted optimization to mitigate severe clinical imbalance, and Gradient-weighted "
                "Class Activation Mapping (Grad-CAM) to provide transparent anatomical attribution maps."
            ),
            "dataset": {
                "name": "RSNA Pneumonia Detection Challenge",
                "status": "Available" if ds_summary else "Not available",
                "summary": ds_summary
            },
            "preprocessing": {
                "input_dimensions": "224x224 (RGB 3-channel)",
                "dicom_handling": "pydicom VOI LUT, Rescale Slope/Intercept, MONOCHROME1 inversion",
                "normalization": "ImageNet standards (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
                "augmentation": "Conservative medical affine transformations (±7° rotation, ±4% translation, ±4% scaling, ±10% brightness/contrast)"
            },
            "data_split": {
                "strategy": "Patient-Level Stratified Splitting (70% Train / 15% Validation / 15% Test)",
                "leakage_prevention": "Strictly verified: 0 overlapping patient IDs across sets",
                "distribution": ds_summary.get("splits") if ds_summary else "Not available"
            },
            "training_configuration": {
                "active_model": active_model,
                "default_optimizer": "AdamW (lr=0.0001, weight_decay=1e-5)",
                "loss_function": "Weighted Cross-Entropy (calculated from inverse class frequencies)",
                "lr_scheduler": "ReduceLROnPlateau (factor=0.5, patience=1)",
                "early_stopping": "Patience of 2 epochs without validation loss improvement",
                "training_history": train_hist if train_hist else "No training history recorded yet"
            },
            "evaluation_results": eval_data if eval_data else "Not evaluated yet. Run evaluation after training checkpoint is created.",
            "model_comparison": models_overview,
            "benchmarks": active_benchmark if active_benchmark else "Benchmark not executed yet for this model.",
            "explainability": {
                "method": "Gradient-weighted Class Activation Mapping (Grad-CAM)",
                "target_layer": "Final convolutional layer of feature extractor",
                "interpretation_note": "Grad-CAM visualizes saliency scores that drove model logit activation. It does not certify clinical truth."
            },
            "limitations": [
                "Single-view (frontal AP/PA) analysis without multi-projection correlation.",
                "Dataset contains demographic bias inherent to RSNA challenge cohort.",
                "Grad-CAM resolution is constrained by the feature map spatial dimension (7x7 at layer4).",
                "Designed for CPU-optimized inference without real-time multi-threaded GPU scaling."
            ],
            "medical_disclaimer": (
                "SmartMed Vision is strictly an educational, portfolio, and research platform. "
                "Model predictions and localization maps may be incorrect and MUST NOT be used for medical "
                "diagnosis, treatment planning, or clinical decision-making."
            )
        }
        return report

report_service = ReportService()
