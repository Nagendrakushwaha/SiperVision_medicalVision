import json
from fastapi import APIRouter
from backend.app.config import EXPERIMENTS_DIR

router = APIRouter(prefix="/experiments", tags=["Experiments"])

@router.get("")
def list_experiments():
    exp_file = EXPERIMENTS_DIR / "experiments.json"
    if exp_file.exists():
        try:
            with open(exp_file, "r", encoding="utf-8") as f:
                return {"experiments": json.load(f)}
        except Exception:
            return {"experiments": []}
    return {"experiments": []}
