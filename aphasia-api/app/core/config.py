from __future__ import annotations

from functools import lru_cache
from os import getenv

from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_name: str = Field(default="Aphasia Rehabilitation Speech Evaluation API")
    app_version: str = Field(default="2.0.0")
    log_level: str = Field(default=getenv("LOG_LEVEL", "INFO"))

    target_sample_rate: int = Field(default=16000)
    max_file_size_mb: int = Field(default=10)
    min_audio_duration_sec: float = Field(default=0.3)

    # WavLM – audio embedding extraction for similarity scoring
    wavlm_model_name: str = Field(default=getenv("WAVLM_MODEL_NAME", "microsoft/wavlm-base-plus"))

    # Wav2Vec2 – CTC-based speech-to-text transcription
    wav2vec2_model_name: str = Field(default=getenv("WAV2VEC2_MODEL_NAME", "facebook/wav2vec2-large-960h"))

    # Scoring parameters
    similarity_scale_factor: float = Field(default=600.0)
    deviation_threshold: float = Field(default=1.0)

    # Combined scoring weights (must sum to 1.0)
    audio_weight: float = Field(default=0.70)
    text_weight: float = Field(default=0.30)

    # Audio preprocessing
    trim_silence: bool = Field(default=True)
    trim_top_db: int = Field(default=25)

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
