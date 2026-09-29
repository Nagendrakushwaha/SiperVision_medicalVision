from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from backend.app.schemas.schemas import TrainingRequest
from backend.app.services.training_service import training_manager

router = APIRouter(prefix="/training", tags=["Training"])

@router.post("/start")
def start_training(req: TrainingRequest):
    # Strict validation of epochs 1-100
    if req.epochs < 1 or req.epochs > 100:
        raise HTTPException(
            status_code=400,
            detail=f"Epochs must be an integer between 1 and 100. Received: {req.epochs}"
        )

    try:
        res = training_manager.start_training(
            model_name=req.model_name,
            epochs=req.epochs,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            image_size=req.image_size,
            early_stopping_patience=req.early_stopping_patience,
            seed=req.seed,
            max_train_samples=req.max_train_samples,
            max_val_samples=req.max_val_samples
        )
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@router.post("/stop")
def stop_training():
    return training_manager.stop_training()

@router.get("/status")
def get_training_status():
    return training_manager.get_status()

@router.get("/history")
def get_training_history(model_name: str = Query("resnet18")):
    history = training_manager.get_history(model_name)
    return {
        "model_name": model_name,
        "history": history,
        "has_history": len(history) > 0
    }

@router.get("/visualize-processing")
def get_visual_processing(
    model_name: str = Query("resnet18"),
    patient_id: Optional[str] = Query(None),
    target: Optional[int] = Query(None)
):
    try:
        from backend.app.services.visual_processing_service import visual_processing_service
        return visual_processing_service.get_pipeline(
            model_name=model_name,
            patient_id=patient_id,
            target=target
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visual processing failed: {str(e)}")

