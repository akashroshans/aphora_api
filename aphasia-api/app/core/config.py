from __future__ import annotations

from functools import lru_cache
from os import getenv

from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_name: str = Field(default="Aphasia Rehabilitation Speech Evaluation API")
    app_version: str = Field(default="1.0.0")
    log_level: str = Field(default=getenv("LOG_LEVEL", "INFO"))

    target_sample_rate: int = Field(default=16000)
    max_file_size_mb: int = Field(default=10)

    wavlm_model_name: str = Field(default=getenv("WAVLM_MODEL_NAME", "microsoft/wavlm-base-plus"))
    whisper_model_name: str = Field(default=getenv("WHISPER_MODEL_NAME", "base"))

    similarity_scale_factor: float = Field(default=600.0)
    deviation_threshold: float = Field(default=1.0)

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
