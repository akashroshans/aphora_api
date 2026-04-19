from __future__ import annotations

import re
from tempfile import NamedTemporaryFile
from threading import Lock

import numpy as np
import torch
from loguru import logger
from scipy.io import wavfile

from app.core.config import get_settings

_punctuation_pattern = re.compile(r"[^\w\s]")


class WhisperService:
    def __init__(self) -> None:
        self._enabled = False
        self._model = None

        settings = get_settings()
        try:
            import whisper  # type: ignore

            logger.info("Loading Whisper model '{}' (optional service)", settings.whisper_model_name)
            self._model = whisper.load_model(settings.whisper_model_name)
            self._enabled = True
            logger.info("Whisper model loaded")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Whisper unavailable. Continuing without transcription. Reason: {}", exc)

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.lower().strip()
        text = _punctuation_pattern.sub("", text)
        return re.sub(r"\s+", " ", text)

    def transcribe_audio(self, waveform: np.ndarray, sample_rate: int) -> str | None:
        if not self._enabled or self._model is None:
            return None

        if waveform.size == 0:
            return None

        clipped = np.clip(waveform, -1.0, 1.0)
        pcm = (clipped * 32767).astype(np.int16)

        with NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            wavfile.write(temp_file.name, sample_rate, pcm)
            result = self._model.transcribe(temp_file.name, fp16=torch.cuda.is_available())

        text = str(result.get("text", "")).strip()
        if not text:
            return ""
        return self._clean_text(text)


_whisper_service: WhisperService | None = None
_whisper_lock = Lock()


def get_whisper_service() -> WhisperService:
    global _whisper_service
    if _whisper_service is None:
        with _whisper_lock:
            if _whisper_service is None:
                _whisper_service = WhisperService()
    return _whisper_service
