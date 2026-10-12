"""Validated public API response contracts."""
from typing import Literal
from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    breed: str
    confidence: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    breed: str
    species: Literal["cat", "dog"]
    confidence: float = Field(ge=0, le=1)
    top_3: list[PredictionItem] = Field(min_length=3, max_length=3)
    decision: Literal["accept", "abstain"]
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["healthy"]
    model_version: str
    num_classes: int
