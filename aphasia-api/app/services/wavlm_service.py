from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import numpy as np
import torch
from loguru import logger
from transformers import AutoFeatureExtractor, WavLMModel

from app.core.config import get_settings


@dataclass(slots=True)
class WavLMRuntime:
    extractor: AutoFeatureExtractor
    model: WavLMModel
    device: torch.device


class WavLMService:
    def __init__(self) -> None:
        settings = get_settings()
        model_name = settings.wavlm_model_name
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading WavLM model '{}' on device '{}'. This may take a while...", model_name, device)

        extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = WavLMModel.from_pretrained(model_name)
        model.eval()
        model.to(device)

        self.runtime = WavLMRuntime(extractor=extractor, model=model, device=device)
        logger.info("WavLM model loaded successfully")

    def extract_embeddings(self, waveform: np.ndarray, sampling_rate: int) -> np.ndarray:
        if waveform.size == 0:
            raise ValueError("Empty waveform cannot be embedded.")

        inputs = self.runtime.extractor(
            waveform,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {name: tensor.to(self.runtime.device) for name, tensor in inputs.items()}

        with torch.no_grad():
            outputs = self.runtime.model(**inputs)

        embeddings = outputs.last_hidden_state.squeeze(0).detach().cpu().numpy()
        if embeddings.ndim != 2:
            raise ValueError("Unexpected embedding shape produced by WavLM.")
        return embeddings.astype(np.float32, copy=False)


_wavlm_service: WavLMService | None = None
_wavlm_lock = Lock()


def get_wavlm_service() -> WavLMService:
    global _wavlm_service
    if _wavlm_service is None:
        with _wavlm_lock:
            if _wavlm_service is None:
                _wavlm_service = WavLMService()
    return _wavlm_service
