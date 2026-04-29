from __future__ import annotations

from io import BytesIO

import librosa
import numpy as np
from loguru import logger


def load_and_preprocess_audio(
    audio_bytes: bytes,
    target_sr: int = 16000,
    trim_silence: bool = True,
    trim_top_db: int = 25,
    min_duration_sec: float = 0.3,
) -> np.ndarray:
    """Load raw audio bytes, resample, normalise, and optionally trim silence.

    Args:
        audio_bytes: Raw bytes of an audio file (WAV, MP3, FLAC, OGG, etc.).
        target_sr: Target sample rate in Hz.
        trim_silence: Whether to trim leading/trailing silence.
        trim_top_db: Threshold in dB below reference to consider as silence.
        min_duration_sec: Minimum audio duration after processing.

    Returns:
        1-D float32 numpy array, amplitude-normalised to [-1, 1].

    Raises:
        ValueError: On empty/invalid audio or audio too short.
    """
    if not audio_bytes:
        raise ValueError("Audio content is empty.")

    try:
        waveform, _ = librosa.load(BytesIO(audio_bytes), sr=target_sr, mono=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Invalid or unsupported audio format.") from exc

    if waveform.size == 0:
        raise ValueError("Decoded audio is empty.")

    waveform = waveform.astype(np.float32, copy=False)

    # ── Trim leading/trailing silence ──
    if trim_silence:
        trimmed, _ = librosa.effects.trim(waveform, top_db=trim_top_db)
        if trimmed.size > 0:
            waveform = trimmed
            logger.debug("Trimmed silence: {} → {} samples", waveform.size, trimmed.size)

    # ── Minimum duration check ──
    duration_sec = waveform.size / target_sr
    if duration_sec < min_duration_sec:
        raise ValueError(
            f"Audio too short ({duration_sec:.2f}s). "
            f"Minimum required: {min_duration_sec}s."
        )

    # ── Peak normalisation ──
    max_abs = float(np.max(np.abs(waveform)))
    if max_abs > 0:
        waveform = waveform / max_abs

    if not np.isfinite(waveform).all():
        raise ValueError("Audio contains invalid numeric values.")

    return waveform
