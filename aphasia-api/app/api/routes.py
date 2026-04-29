from __future__ import annotations

from time import perf_counter

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from loguru import logger

from app.core.config import get_settings
from app.models.response import EvaluationResponse, HealthResponse
from app.services.dtw_service import get_dtw_service
from app.services.scoring_service import get_scoring_service
from app.services.wav2vec2_service import get_wav2vec2_service
from app.services.wavlm_service import get_wavlm_service
from app.utils.audio import load_and_preprocess_audio

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_speech(
    reference_audio: UploadFile = File(..., description="Reference (target) audio file"),
    user_audio: UploadFile = File(..., description="Patient's speech attempt"),
    reference_text: str | None = Form(
        default=None,
        description="Optional: therapist-provided expected text. "
        "If omitted, the reference audio is transcribed via Wav2Vec2.",
    ),
) -> EvaluationResponse:
    """Evaluate a patient's speech attempt against a reference pronunciation.

    Performs both **audio-level** (WavLM embeddings + DTW) and **text-level**
    (Wav2Vec2 CTC transcription + CER/WER) analysis, returning a combined
    accuracy score and human-readable feedback.
    """
    settings = get_settings()
    started_at = perf_counter()

    try:
        # ── 1. Read & validate uploads ──
        reference_bytes = await reference_audio.read()
        user_bytes = await user_audio.read()

        if not reference_bytes or not user_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both reference_audio and user_audio are required and cannot be empty.",
            )

        if len(reference_bytes) > settings.max_file_size_bytes or len(user_bytes) > settings.max_file_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Audio file exceeds max size of {settings.max_file_size_mb} MB.",
            )

        # ── 2. Preprocess audio ──
        ref_waveform = load_and_preprocess_audio(
            reference_bytes,
            target_sr=settings.target_sample_rate,
            trim_silence=settings.trim_silence,
            trim_top_db=settings.trim_top_db,
            min_duration_sec=settings.min_audio_duration_sec,
        )
        usr_waveform = load_and_preprocess_audio(
            user_bytes,
            target_sr=settings.target_sample_rate,
            trim_silence=settings.trim_silence,
            trim_top_db=settings.trim_top_db,
            min_duration_sec=settings.min_audio_duration_sec,
        )

        # ── 3. Get services ──
        wavlm_service = get_wavlm_service()
        dtw_service = get_dtw_service()
        scoring_service = get_scoring_service()
        wav2vec2_service = get_wav2vec2_service()

        # ── 4. Audio-level analysis (WavLM + DTW) ──
        ref_embeddings = wavlm_service.extract_embeddings(ref_waveform, sampling_rate=settings.target_sample_rate)
        usr_embeddings = wavlm_service.extract_embeddings(usr_waveform, sampling_rate=settings.target_sample_rate)

        dtw_result = dtw_service.compute_alignment(ref_embeddings, usr_embeddings)

        # ── 5. Text-level analysis (Wav2Vec2) ──
        # Use therapist-provided text if available, otherwise transcribe reference
        if reference_text is not None and reference_text.strip():
            ref_text = reference_text.strip().lower()
        else:
            ref_text = wav2vec2_service.transcribe(ref_waveform, settings.target_sample_rate)

        usr_text = wav2vec2_service.transcribe(usr_waveform, settings.target_sample_rate)

        # ── 6. Combined scoring ──
        scoring_result = scoring_service.score(
            dtw_distance=dtw_result.distance,
            per_frame_distances=dtw_result.per_frame_distances,
            alignment_path=dtw_result.path,
            user_sequence_length=usr_embeddings.shape[0],
            scale_factor=settings.similarity_scale_factor,
            deviation_threshold=settings.deviation_threshold,
            reference_text=ref_text,
            user_text=usr_text,
            audio_weight=settings.audio_weight,
            text_weight=settings.text_weight,
        )

        elapsed_ms = (perf_counter() - started_at) * 1000
        logger.info(
            "Evaluation complete in {:.1f}ms | combined={:.1f}% audio={:.1f}% cer={}",
            elapsed_ms,
            scoring_result.combined_accuracy,
            scoring_result.accuracy,
            scoring_result.character_error_rate,
        )

        return EvaluationResponse(
            combined_accuracy=scoring_result.combined_accuracy,
            feedback_message=scoring_result.feedback_message,
            accuracy=scoring_result.accuracy,
            audio_similarity=scoring_result.similarity_score,
            deviation_position=scoring_result.deviation_position,
            deviation_index=scoring_result.deviation_index,
            mismatch_indices=scoring_result.mismatch_indices,
            dtw_distance=dtw_result.distance,
            alignment_path_length=len(dtw_result.path),
            reference_text=ref_text,
            user_text=usr_text,
            text_similarity=scoring_result.text_similarity,
            character_error_rate=scoring_result.character_error_rate,
            word_error_rate=scoring_result.word_error_rate,
            processing_time_ms=elapsed_ms,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        logger.exception("Audio validation/processing error")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected evaluation error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while evaluating speech.",
        ) from exc
