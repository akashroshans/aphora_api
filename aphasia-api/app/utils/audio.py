from __future__ import annotations

from io import BytesIO

import librosa
import numpy as np


def load_and_preprocess_audio(audio_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    if not audio_bytes:
        raise ValueError("Audio content is empty.")

    try:
        waveform, _ = librosa.load(BytesIO(audio_bytes), sr=target_sr, mono=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Invalid or unsupported audio format.") from exc

    if waveform.size == 0:
        raise ValueError("Decoded audio is empty.")

    waveform = waveform.astype(np.float32, copy=False)
    max_abs = float(np.max(np.abs(waveform)))
    if max_abs > 0:
        waveform = waveform / max_abs

    if not np.isfinite(waveform).all():
        raise ValueError("Audio contains invalid numeric values.")

    return waveform
