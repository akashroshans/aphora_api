from __future__ import annotations

from time import perf_counter

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from loguru import logger

from app.core.config import get_settings
from app.models.response import EvaluationResponse, HealthResponse
from app.services.dtw_service import get_dtw_service
from app.services.scoring_service import get_scoring_service
from app.services.wavlm_service import get_wavlm_service
from app.services.whisper_service import get_whisper_service
from app.utils.audio import load_and_preprocess_audio

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_speech(
    reference_audio: UploadFile = File(...),
    user_audio: UploadFile = File(...),
) -> EvaluationResponse:
    settings = get_settings()
    started_at = perf_counter()

    try:
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

        ref_waveform = load_and_preprocess_audio(reference_bytes, target_sr=settings.target_sample_rate)
        usr_waveform = load_and_preprocess_audio(user_bytes, target_sr=settings.target_sample_rate)

        wavlm_service = get_wavlm_service()
        dtw_service = get_dtw_service()
        scoring_service = get_scoring_service()
        whisper_service = get_whisper_service()

        ref_embeddings = wavlm_service.extract_embeddings(ref_waveform, sampling_rate=settings.target_sample_rate)
        usr_embeddings = wavlm_service.extract_embeddings(usr_waveform, sampling_rate=settings.target_sample_rate)

        dtw_result = dtw_service.compute_alignment(ref_embeddings, usr_embeddings)

        scoring_result = scoring_service.score(
            dtw_distance=dtw_result.distance,
            per_frame_distances=dtw_result.per_frame_distances,
            alignment_path=dtw_result.path,
            user_sequence_length=usr_embeddings.shape[0],
            scale_factor=settings.similarity_scale_factor,
            deviation_threshold=settings.deviation_threshold,
        )

        ref_text = whisper_service.transcribe_audio(ref_waveform, settings.target_sample_rate)
        usr_text = whisper_service.transcribe_audio(usr_waveform, settings.target_sample_rate)

        elapsed_ms = (perf_counter() - started_at) * 1000

        return EvaluationResponse(
            accuracy=scoring_result.accuracy,
            deviation_position=scoring_result.deviation_position,
            similarity_score=scoring_result.similarity_score,
            deviation_index=scoring_result.deviation_index,
            reference_text=ref_text,
            user_text=usr_text,
            mismatch_indices=scoring_result.mismatch_indices,
            dtw_distance=dtw_result.distance,
            alignment_path_length=len(dtw_result.path),
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
