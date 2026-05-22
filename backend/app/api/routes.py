from fastapi import APIRouter, HTTPException

from app.schemas.prediction import (
    ImagePredictionRequest,
    MultimodalPredictionRequest,
    PredictionResponse,
    TabularPredictionRequest,
    TextPredictionRequest,
)
from app.services.inference import InferenceService


router = APIRouter(prefix="/api/v1", tags=["prediction"])
service = InferenceService()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def ready() -> dict[str, object]:
    return service.readiness_report()


@router.post("/predict/tabular", response_model=PredictionResponse)
def predict_tabular(payload: TabularPredictionRequest) -> PredictionResponse:
    return service.predict_tabular(payload)


@router.post("/predict/text", response_model=PredictionResponse)
def predict_text(payload: TextPredictionRequest) -> PredictionResponse:
    return service.predict_text(payload)


@router.post("/predict/image", response_model=PredictionResponse)
def predict_image(payload: ImagePredictionRequest) -> PredictionResponse:
    return service.predict_image(payload)


@router.post("/predict/multimodal", response_model=PredictionResponse)
def predict_multimodal(payload: MultimodalPredictionRequest) -> PredictionResponse:
    if not payload.has_any_modality():
        raise HTTPException(
            status_code=400,
            detail="At least one modality must be provided.",
        )
    return service.predict_multimodal(payload)
