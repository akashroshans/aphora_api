from __future__ import annotations

import re
from dataclasses import dataclass
from threading import Lock

import numpy as np
import torch
from loguru import logger
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from app.core.config import get_settings

_punctuation_pattern = re.compile(r"[^\w\s]")


@dataclass(slots=True)
class Wav2Vec2Runtime:
    processor: Wav2Vec2Processor
    model: Wav2Vec2ForCTC
    device: torch.device


class Wav2Vec2Service:
    """Speech-to-text transcription using Facebook's Wav2Vec2 CTC model.

    Replaces the previous Whisper-based service with a model that:
    - Runs entirely in-memory (no temp files)
    - Uses the same HuggingFace Transformers framework as WavLM
    - Produces character-level output useful for aphasia analysis
    - Supports GPU acceleration automatically
    """

    def __init__(self) -> None:
        settings = get_settings()
        model_name = settings.wav2vec2_model_name
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            "Loading Wav2Vec2 model '{}' on device '{}'. This may take a while...",
            model_name,
            device,
        )

        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        model.eval()
        model.to(device)

        self.runtime = Wav2Vec2Runtime(processor=processor, model=model, device=device)
        logger.info("Wav2Vec2 model loaded successfully")

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize transcription output: lowercase, strip punctuation, collapse whitespace."""
        text = text.lower().strip()
        text = _punctuation_pattern.sub("", text)
        return re.sub(r"\s+", " ", text).strip()

    def transcribe(self, waveform: np.ndarray, sampling_rate: int) -> str | None:
        """Transcribe a mono audio waveform to text.

        Args:
            waveform: 1-D float32 numpy array, amplitude-normalised to [-1, 1].
            sampling_rate: Sample rate of the waveform (must be 16 kHz).

        Returns:
            Cleaned transcription string, or None on failure.
        """
        if waveform.size == 0:
            return None

        try:
            inputs = self.runtime.processor(
                waveform,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                padding="longest",
            )
            input_values = inputs.input_values.to(self.runtime.device)

            with torch.no_grad():
                logits = self.runtime.model(input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.runtime.processor.batch_decode(predicted_ids)[0]

            if not transcription or not transcription.strip():
                return ""

            return self._clean_text(transcription)

        except Exception as exc:  # noqa: BLE001
            logger.warning("Wav2Vec2 transcription failed: {}", exc)
            return None


# ---------------------------------------------------------------------------
# Lazy singleton – thread-safe, loaded once on first use
# ---------------------------------------------------------------------------
_wav2vec2_service: Wav2Vec2Service | None = None
_wav2vec2_lock = Lock()


def get_wav2vec2_service() -> Wav2Vec2Service:
    global _wav2vec2_service
    if _wav2vec2_service is None:
        with _wav2vec2_lock:
            if _wav2vec2_service is None:
                _wav2vec2_service = Wav2Vec2Service()
    return _wav2vec2_service
