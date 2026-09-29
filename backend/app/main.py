import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.health import router as health_router
from backend.app.api.system import router as system_router
from backend.app.api.dataset import router as dataset_router
from backend.app.api.models import router as models_router
from backend.app.api.training import router as training_router
from backend.app.api.prediction import router as prediction_router
from backend.app.api.evaluation import router as evaluation_router
from backend.app.api.experiments import router as experiments_router
from backend.app.api.benchmarks import router as benchmarks_router
from backend.app.api.reports import router as reports_router
from backend.app.schemas.schemas import TrainingRequest, BenchmarkRequest
from backend.app.services.training_service import training_manager
from backend.app.services.benchmark_service import benchmark_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("smartmed-vision")

app = FastAPI(
    title="SmartMed Vision API",
    description="Explainable Pneumonia Detection & Localization Platform using RSNA Dataset",
    version="1.0.0"
)

# Enable CORS for React/Vite development server (ports 5173, 3000, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Modular Routers under /api
app.include_router(health_router, prefix="/api")
app.include_router(system_router, prefix="/api")
app.include_router(dataset_router, prefix="/api")
app.include_router(models_router, prefix="/api")
app.include_router(training_router, prefix="/api")
app.include_router(prediction_router, prefix="/api")
app.include_router(evaluation_router, prefix="/api")
app.include_router(experiments_router, prefix="/api")
app.include_router(benchmarks_router, prefix="/api")
app.include_router(reports_router, prefix="/api")

# Top-level alias endpoints for direct specification compliance:
# POST /api/train -> starts training
@app.post("/api/train")
def train_alias(req: TrainingRequest):
    return training_manager.start_training(
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

# POST /api/benchmark -> runs benchmark
@app.post("/api/benchmark")
def benchmark_alias(req: BenchmarkRequest):
    return benchmark_service.run_benchmark(
        model_name=req.model_name,
        num_runs=req.num_runs
    )

@app.get("/")
def root():
    return {
        "title": "SmartMed Vision API",
        "description": "Explainable Pneumonia Detection & Localization Platform",
        "docs": "/docs",
        "health": "/api/health",
        "disclaimer": "SmartMed Vision is an educational/research platform. Model predictions must not be used for clinical diagnosis."
    }
