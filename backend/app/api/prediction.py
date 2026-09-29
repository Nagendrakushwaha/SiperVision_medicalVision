from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Body
from PIL import Image
import io

from backend.app.services.inference import inference_engine
from backend.app.services.dataset_service import dataset_service
from backend.app.services.localization import draw_bounding_boxes
from backend.app.schemas.schemas import PredictFromPatientRequest

router = APIRouter(prefix="", tags=["Prediction & Explainability"])

@router.post("/predict")
async def predict_image(
    file: Optional[UploadFile] = File(None),
    patient_id: Optional[str] = Form(None),
    model_name: str = Form("resnet18"),
    include_gradcam: bool = Form(True)
):
    """
    Inference endpoint supporting DICOM (.dcm), PNG, JPG, JPEG upload,
    OR direct patient ID from the RSNA dataset.
    """
    try:
        if file is not None:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            res = inference_engine.predict(
                image_source=content,
                model_name=model_name,
                include_gradcam=include_gradcam,
                patient_id=patient_id
            )
            res["filename"] = file.filename
            return res
        elif patient_id:
            dcm_path = dataset_service.get_patient_dcm_path(patient_id)
            if not dcm_path or not dcm_path.exists():
                raise HTTPException(status_code=404, detail=f"DICOM file for patient {patient_id} not found.")
            res = inference_engine.predict(
                image_source=dcm_path,
                model_name=model_name,
                include_gradcam=include_gradcam,
                patient_id=patient_id
            )
            return res
        else:
            raise HTTPException(status_code=400, detail="Must provide either an uploaded image file or a patient_id.")

    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/predict/patient")
def predict_patient_json(req: PredictFromPatientRequest):
    dcm_path = dataset_service.get_patient_dcm_path(req.patient_id)
    if not dcm_path or not dcm_path.exists():
        raise HTTPException(status_code=404, detail=f"DICOM file for patient {req.patient_id} not found.")
    try:
        return inference_engine.predict(
            image_source=dcm_path,
            model_name=req.model_name,
            include_gradcam=req.include_gradcam,
            patient_id=req.patient_id
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/prediction/history")
def get_prediction_history():
    return {"history": inference_engine.get_prediction_history()}

@router.post("/prediction/clear")
def clear_prediction_history():
    inference_engine.clear_prediction_history()
    return {"message": "Prediction history cleared successfully."}
