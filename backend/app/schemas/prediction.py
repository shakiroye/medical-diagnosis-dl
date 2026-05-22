from typing import Any

from pydantic import BaseModel, Field


class TabularPredictionRequest(BaseModel):
    patient_id: str | None = None
    features: dict[str, float | int | str | bool] = Field(default_factory=dict)


class TextPredictionRequest(BaseModel):
    patient_id: str | None = None
    text: str
    language: str = "fr"


class ImagePredictionRequest(BaseModel):
    patient_id: str | None = None
    image_base64: str
    image_type: str = "chest_xray"


class MultimodalPredictionRequest(BaseModel):
    patient_id: str | None = None
    tabular: TabularPredictionRequest | None = None
    text: TextPredictionRequest | None = None
    image: ImagePredictionRequest | None = None

    def has_any_modality(self) -> bool:
        return any([self.tabular, self.text, self.image])


class PredictionResponse(BaseModel):
    modality: str
    label: str
    confidence: float
    model_name: str
    explanation: str
    metadata: dict[str, Any] = Field(default_factory=dict)
