from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EvaluationResponse(BaseModel):
    accuracy: float = Field(..., ge=0.0, le=100.0)
    deviation_position: Literal["START", "MIDDLE", "END", "PERFECT"]
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    deviation_index: int | None = Field(default=None, ge=0)

    reference_text: str | None = None
    user_text: str | None = None

    mismatch_indices: list[int] = Field(default_factory=list)
    dtw_distance: float = Field(..., ge=0.0)
    alignment_path_length: int = Field(..., ge=0)
    processing_time_ms: float = Field(..., ge=0.0)


class HealthResponse(BaseModel):
    status: Literal["ok"]
