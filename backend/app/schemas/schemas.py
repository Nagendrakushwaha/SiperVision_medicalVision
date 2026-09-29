from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator

class TrainingRequest(BaseModel):
    model_name: str = Field(default="resnet18", description="Target model: resnet18, mobilenet, efficientnet")
    epochs: int = Field(default=5, ge=1, le=100, description="Epoch count must be between 1 and 100")
    batch_size: int = Field(default=8, ge=1, le=64)
    learning_rate: float = Field(default=0.0001, gt=0.0)
    image_size: int = Field(default=224, ge=128, le=512)
    early_stopping_patience: int = Field(default=2, ge=1, le=20)
    seed: int = Field(default=42)
    max_train_samples: Optional[int] = Field(default=None, ge=1)
    max_val_samples: Optional[int] = Field(default=None, ge=1)

    @field_validator("epochs")
    @classmethod
    def validate_epochs(cls, v: int) -> int:
        if v < 1 or v > 100:
            raise ValueError(f"Epochs must be an integer between 1 and 100. Received: {v}")
        return v

class EvaluationRequest(BaseModel):
    model_name: str = Field(default="resnet18")
    split_name: str = Field(default="test")
    max_samples: Optional[int] = Field(default=300, ge=10, le=5000)

class BenchmarkRequest(BaseModel):
    model_name: str = Field(default="resnet18")
    num_runs: int = Field(default=50, ge=5, le=500)

class PredictFromPatientRequest(BaseModel):
    patient_id: str
    model_name: str = "resnet18"
    include_gradcam: bool = True

class SetActiveModelRequest(BaseModel):
    model_name: str
