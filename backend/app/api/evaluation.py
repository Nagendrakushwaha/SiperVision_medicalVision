from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from backend.app.services.evaluation import evaluation_service
from backend.app.schemas.schemas import EvaluationRequest

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])

@router.get("")
def get_evaluation(model_name: str = Query("resnet18")):
    data = evaluation_service.get_evaluation(model_name)
    if not data:
        return {
            "status": "not_evaluated",
            "model_name": model_name,
            "message": f"No evaluation results available for {model_name}. Please run evaluation after training."
        }
    return {"status": "available", "data": data}

@router.post("/run")
def run_evaluation(req: EvaluationRequest):
    try:
        results = evaluation_service.run_evaluation(
            model_name=req.model_name,
            split_name=req.split_name,
            max_samples=req.max_samples
        )
        return {"status": "success", "results": results}
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
