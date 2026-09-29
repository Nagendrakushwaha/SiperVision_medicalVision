import sys
from pathlib import Path
import pytest
import numpy as np
from PIL import Image
import torch

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.dicom_service import read_dicom, dicom_to_pil
from backend.app.services.preprocessing import preprocess_image_for_inference, get_train_transforms, get_val_transforms
from backend.app.services.dataset_service import dataset_service, discover_dataset_files
from backend.app.services.localization import draw_bounding_boxes
from backend.app.models.model_factory import (
    build_model,
    get_model_target_layer,
    count_parameters,
    normalize_model_name
)
from backend.app.services.gradcam import compute_gradcam_visualizations
from backend.app.schemas.schemas import TrainingRequest

def test_dicom_loading_and_metadata():
    files = discover_dataset_files()
    assert files["images_dir"] is not None and files["images_dir"].exists()
    dcm_files = list(files["images_dir"].glob("*.dcm"))
    assert len(dcm_files) > 0, "No DICOM files found in dataset"

    sample_dcm = dcm_files[0]
    arr, meta = read_dicom(sample_dcm)

    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint8
    assert arr.shape == (1024, 1024)
    assert "modality" in meta
    assert "photometric_interpretation" in meta
    assert "patient_name" not in meta  # Non-identifying metadata

def test_preprocessing_pipeline():
    dummy_img = Image.new("RGB", (512, 512), color=(128, 128, 128))
    tensor, pil_out = preprocess_image_for_inference(dummy_img, image_size=224)

    assert tensor.shape == (1, 3, 224, 224)
    assert isinstance(tensor, torch.Tensor)
    assert pil_out.size == (512, 512)

def test_patient_level_leakage_prevention():
    split = dataset_service.get_split()
    train_patients = set(split["train"])
    val_patients = set(split["validation"])
    test_patients = set(split["test"])

    # Strict Patient-Level Leakage Assertions
    assert len(train_patients.intersection(val_patients)) == 0, "Leakage: Overlap between Train and Validation!"
    assert len(train_patients.intersection(test_patients)) == 0, "Leakage: Overlap between Train and Test!"
    assert len(val_patients.intersection(test_patients)) == 0, "Leakage: Overlap between Validation and Test!"
    assert len(train_patients) > 0 and len(val_patients) > 0 and len(test_patients) > 0

def test_bounding_box_localization():
    img = Image.new("RGB", (1024, 1024), color=(50, 50, 50))
    boxes = [{"x": 100.0, "y": 150.0, "width": 200.0, "height": 300.0}]
    annotated_img, b64_str = draw_bounding_boxes(img, boxes)

    assert annotated_img.size == (1024, 1024)
    assert b64_str.startswith("data:image/png;base64,")

def test_model_factory_and_layer_discovery():
    for m_name in ["resnet18", "mobilenet", "efficientnet"]:
        model = build_model(m_name, pretrained=False)
        params = count_parameters(model)
        target_layer = get_model_target_layer(m_name, model)

        assert params["total_parameters"] > 1_000_000
        assert target_layer is not None
        assert isinstance(target_layer, torch.nn.Module)

def test_epoch_limit_validation():
    # Valid epoch: 5
    req_valid = TrainingRequest(model_name="resnet18", epochs=5)
    assert req_valid.epochs == 5

    # Reject epoch 0
    with pytest.raises(Exception):
        TrainingRequest(model_name="resnet18", epochs=0)

    # Reject epoch -5
    with pytest.raises(Exception):
        TrainingRequest(model_name="resnet18", epochs=-5)

    # Reject epoch 101
    with pytest.raises(Exception):
        TrainingRequest(model_name="resnet18", epochs=101)

def test_gradcam_computation():
    model = build_model("resnet18", pretrained=False)
    input_tensor = torch.randn(1, 3, 224, 224)
    original_pil = Image.new("RGB", (224, 224), (100, 100, 100))

    cam_res = compute_gradcam_visualizations(
        model=model,
        model_name="resnet18",
        input_tensor=input_tensor,
        original_pil=original_pil,
        target_class=1
    )

    assert "heatmap_base64" in cam_res
    assert "overlay_base64" in cam_res
    assert "disclaimer" in cam_res
    assert cam_res["overlay_base64"].startswith("data:image/png;base64,")

def test_api_endpoints():
    from fastapi.testclient import TestClient
    from backend.app.main import app

    client = TestClient(app)
    
    # 1. Health
    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"

    # 2. System Status
    res_sys = client.get("/api/system")
    assert res_sys.status_code == 200
    assert "AMD" in res_sys.json()["hardware"]["cpu"] or "Ryzen" in res_sys.json()["hardware"]["cpu"]
    assert res_sys.json()["dataset"]["status"] == "Available"

    # 3. Dataset Summary
    res_ds = client.get("/api/dataset/summary")
    assert res_ds.status_code == 200
    assert res_ds.json()["total_images"] == 26684
    assert res_ds.json()["positive_cases"] == 6012

    # 4. Models List
    res_models = client.get("/api/models")
    assert res_models.status_code == 200
    assert len(res_models.json()["models"]) == 3
