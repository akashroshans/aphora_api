from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EvaluationResponse(BaseModel):
    # ── Combined result ──
    combined_accuracy: float = Field(..., ge=0.0, le=100.0, description="Weighted blend of audio and text accuracy")
    feedback_message: str = Field(..., description="Human-readable feedback for the patient")

    # ── Audio-level (WavLM + DTW) ──
    accuracy: float = Field(..., ge=0.0, le=100.0, description="Audio-only accuracy percentage")
    audio_similarity: float = Field(..., ge=0.0, le=1.0, description="Raw audio similarity score (0-1)")
    deviation_position: Literal["START", "MIDDLE", "END", "PERFECT"]
    deviation_index: int | None = Field(default=None, ge=0)
    mismatch_indices: list[int] = Field(default_factory=list)
    dtw_distance: float = Field(..., ge=0.0)
    alignment_path_length: int = Field(..., ge=0)

    # ── Text-level (Wav2Vec2) ──
    reference_text: str | None = None
    user_text: str | None = None
    text_similarity: float | None = Field(default=None, ge=0.0, le=1.0, description="1 - CER, clamped to [0,1]")
    character_error_rate: float | None = Field(default=None, ge=0.0, description="Character Error Rate")
    word_error_rate: float | None = Field(default=None, ge=0.0, description="Word Error Rate")

    # ── Meta ──
    processing_time_ms: float = Field(..., ge=0.0)


class HealthResponse(BaseModel):
    status: Literal["ok"]
