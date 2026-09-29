from typing import Optional
from fastapi import APIRouter, HTTPException
from backend.app.config import CHECKPOINTS_DIR
from backend.app.models.model_factory import (
    SUPPORTED_MODELS,
    build_model,
    get_model_info,
    normalize_model_name
)
from backend.app.services.evaluation import evaluation_service
from backend.app.schemas.schemas import SetActiveModelRequest

router = APIRouter(prefix="/models", tags=["Models"])

ACTIVE_MODEL = "resnet18"

@router.get("")
def list_models():
    results = []
    for key, meta in SUPPORTED_MODELS.items():
        ckpt_dir = CHECKPOINTS_DIR / key
        has_best = (ckpt_dir / "best.pth").exists()
        has_last = (ckpt_dir / "last.pth").exists()
        
        # Check evaluation status
        eval_record = evaluation_service.get_evaluation(key)
        if eval_record is not None:
            status = "Evaluated"
        elif has_best or has_last:
            status = "Trained — Not Evaluated"
        else:
            status = "Not Trained"

        # Instantiate in eval mode to get exact parameter count
        try:
            m = build_model(key, pretrained=False)
            info = get_model_info(key, m)
        except Exception:
            info = {"name": meta["name"], "id": key, "description": meta["description"]}

        info["status"] = status
        info["has_checkpoint"] = has_best or has_last
        info["is_active"] = (key == ACTIVE_MODEL)
        info["metrics"] = eval_record.get("metrics") if eval_record else None
        results.append(info)

    return {"models": results, "active_model": ACTIVE_MODEL}

@router.get("/{model_name}")
def get_model(model_name: str):
    try:
        norm_name = normalize_model_name(model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    meta = SUPPORTED_MODELS[norm_name]
    ckpt_dir = CHECKPOINTS_DIR / norm_name
    has_best = (ckpt_dir / "best.pth").exists()
    has_last = (ckpt_dir / "last.pth").exists()
    eval_record = evaluation_service.get_evaluation(norm_name)

    m = build_model(norm_name, pretrained=False)
    info = get_model_info(norm_name, m)
    info["status"] = "Evaluated" if eval_record else ("Trained — Not Evaluated" if (has_best or has_last) else "Not Trained")
    info["has_checkpoint"] = has_best or has_last
    info["is_active"] = (norm_name == ACTIVE_MODEL)
    info["evaluation"] = eval_record

    return info

@router.post("/active")
def set_active_model(req: SetActiveModelRequest):
    global ACTIVE_MODEL
    try:
        norm = normalize_model_name(req.model_name)
        ACTIVE_MODEL = norm
        return {"status": "success", "active_model": ACTIVE_MODEL}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
