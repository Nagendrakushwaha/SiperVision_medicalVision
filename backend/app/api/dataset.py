from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from backend.app.services.dataset_service import dataset_service
from backend.app.services.dicom_service import dicom_to_base64_png
from backend.app.services.localization import draw_bounding_boxes

router = APIRouter(prefix="/dataset", tags=["Dataset"])

@router.get("/summary")
def get_dataset_summary():
    try:
        return dataset_service.get_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset summary: {str(e)}")

@router.get("/split")
def get_patient_split():
    try:
        return dataset_service.get_split()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load patient split: {str(e)}")

@router.get("/examples")
def get_dataset_examples(
    split: Optional[str] = Query(None, description="train, validation, test, or all"),
    target: Optional[int] = Query(None, description="0 (Normal) or 1 (Pneumonia)"),
    has_bbox: Optional[bool] = Query(None, description="Filter images with bounding boxes"),
    search: Optional[str] = Query(None, description="Patient ID search"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
):
    try:
        return dataset_service.query_examples(
            split=split,
            target=target,
            has_bbox=has_bbox,
            search=search,
            page=page,
            page_size=page_size
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/random")
def get_random_example(
    split: Optional[str] = Query(None),
    target: Optional[int] = Query(None),
    has_bbox: Optional[bool] = Query(None)
):
    sample = dataset_service.get_random_sample(split=split, target=target, has_bbox=has_bbox)
    if not sample:
        raise HTTPException(status_code=404, detail="No matching sample found in dataset.")
    return sample

@router.get("/patient/{patient_id}")
def get_patient_details(patient_id: str):
    patient_info = dataset_service.get_patient(patient_id)
    if not patient_info:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found in RSNA manifest.")

    dcm_path = dataset_service.get_patient_dcm_path(patient_id)
    if not dcm_path or not dcm_path.exists():
        raise HTTPException(status_code=404, detail=f"DICOM file for patient {patient_id} not found on disk.")

    try:
        img_b64, metadata = dicom_to_base64_png(dcm_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read DICOM image: {str(e)}")

    # If boxes exist, draw annotated overlay
    boxes = patient_info.get("boxes", [])
    bbox_img_b64 = None
    if boxes:
        from backend.app.services.dicom_service import dicom_to_pil
        pil_img, _ = dicom_to_pil(dcm_path)
        _, bbox_img_b64 = draw_bounding_boxes(pil_img, boxes)

    return {
        "patient_id": patient_id,
        "target": patient_info.get("target"),
        "label": "Pneumonia" if patient_info.get("target") == 1 else "Normal",
        "detailed_class": patient_info.get("detailed_class"),
        "split": patient_info.get("split"),
        "boxes": boxes,
        "box_count": len(boxes),
        "metadata": metadata,
        "image_base64": img_b64,
        "annotated_image_base64": bbox_img_b64
    }
