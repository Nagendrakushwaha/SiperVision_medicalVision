from fastapi import APIRouter, Query
from backend.app.services.report_service import report_service

router = APIRouter(prefix="/reports", tags=["Reports"])

@router.get("")
def get_report(model_name: str = Query("resnet18")):
    return report_service.generate_full_report(active_model=model_name)
