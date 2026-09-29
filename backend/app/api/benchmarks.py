from fastapi import APIRouter, HTTPException
from backend.app.services.benchmark_service import benchmark_service
from backend.app.schemas.schemas import BenchmarkRequest

router = APIRouter(prefix="/benchmarks", tags=["Benchmarks"])

@router.get("")
def list_benchmarks():
    return {"benchmarks": benchmark_service.get_benchmarks()}

@router.post("")
def run_benchmark(req: BenchmarkRequest):
    try:
        res = benchmark_service.run_benchmark(
            model_name=req.model_name,
            num_runs=req.num_runs
        )
        return {"status": "success", "benchmark": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark error: {str(e)}")
