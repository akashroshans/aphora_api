from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np

from app.utils.text_metrics import character_error_rate, word_error_rate


@dataclass(slots=True)
class ScoringResult:
    # Audio-level (WavLM + DTW)
    similarity_score: float
    accuracy: float
    deviation_position: str
    deviation_index: int | None
    mismatch_indices: list[int]

    # Text-level (Wav2Vec2 transcriptions)
    character_error_rate: float | None
    word_error_rate: float | None
    text_similarity: float | None

    # Combined
    combined_accuracy: float
    feedback_message: str


class ScoringService:
    @staticmethod
    def distance_to_similarity(distance: float, scale_factor: float) -> float:
        if scale_factor <= 0:
            raise ValueError("scale_factor must be greater than zero.")
        similarity = exp(-distance / scale_factor)
        return float(np.clip(similarity, 0.0, 1.0))

    @staticmethod
    def _resolve_threshold(per_frame_distances: np.ndarray, base_threshold: float) -> float:
        if base_threshold > 0:
            return float(base_threshold)

        # Adaptive fallback threshold if config threshold is not positive.
        mean = float(np.mean(per_frame_distances))
        std = float(np.std(per_frame_distances))
        return mean + std

    def _detect_deviation(
        self,
        per_frame_distances: np.ndarray,
        alignment_path: list[tuple[int, int]],
        user_sequence_length: int,
        threshold: float,
    ) -> tuple[str, int | None, list[int]]:
        if per_frame_distances.size == 0 or user_sequence_length <= 0:
            return "PERFECT", None, []

        mismatch_mask = per_frame_distances > threshold
        mismatch_alignment_indices = np.where(mismatch_mask)[0]

        if mismatch_alignment_indices.size == 0:
            return "PERFECT", None, []

        mismatch_indices = sorted({alignment_path[i][1] for i in mismatch_alignment_indices.tolist()})
        first_idx = mismatch_indices[0]

        relative_pos = first_idx / max(user_sequence_length - 1, 1)
        if relative_pos < 0.33:
            position = "START"
        elif relative_pos < 0.66:
            position = "MIDDLE"
        else:
            position = "END"

        return position, first_idx, mismatch_indices

    @staticmethod
    def _compute_text_metrics(
        reference_text: str | None,
        user_text: str | None,
    ) -> tuple[float | None, float | None, float | None]:
        """Compute CER, WER, and text similarity from transcriptions.

        Returns:
            Tuple of (cer, wer, text_similarity). All None if texts unavailable.
        """
        if not reference_text or not user_text:
            return None, None, None

        cer = character_error_rate(reference_text, user_text)
        wer = word_error_rate(reference_text, user_text)

        # Text similarity is 1 - CER, clamped to [0, 1]
        text_sim = max(0.0, min(1.0, 1.0 - cer))

        return round(cer, 4), round(wer, 4), round(text_sim, 4)

    @staticmethod
    def _compute_combined_accuracy(
        audio_similarity: float,
        text_similarity: float | None,
        audio_weight: float,
        text_weight: float,
    ) -> float:
        """Weighted combination of audio and text similarity.

        Falls back to audio-only when text is unavailable.
        """
        if text_similarity is None:
            return round(audio_similarity * 100.0, 2)

        combined = (audio_weight * audio_similarity) + (text_weight * text_similarity)
        return round(combined * 100.0, 2)

    @staticmethod
    def _generate_feedback(
        combined_accuracy: float,
        deviation_position: str,
        reference_text: str | None,
        user_text: str | None,
    ) -> str:
        """Generate a human-readable feedback message for the patient."""
        # Overall performance bracket
        if combined_accuracy >= 95:
            praise = "Excellent work!"
        elif combined_accuracy >= 85:
            praise = "Great job!"
        elif combined_accuracy >= 70:
            praise = "Good effort!"
        elif combined_accuracy >= 50:
            praise = "Keep practicing!"
        else:
            praise = "Don't give up — every attempt helps!"

        score_part = f"You scored {combined_accuracy:.0f}%."

        # Position-based guidance
        if deviation_position == "PERFECT":
            position_hint = "Your pronunciation was spot-on!"
        elif deviation_position == "START":
            position_hint = "The main difficulty was at the beginning of the phrase. Try focusing on the opening sounds."
        elif deviation_position == "MIDDLE":
            position_hint = "The main difficulty was in the middle of the phrase. Try slowing down through the middle section."
        else:  # END
            position_hint = "The main difficulty was at the end of the phrase. Try maintaining focus through the final sounds."

        # Word-level diff hint (if transcriptions are available)
        word_hint = ""
        if reference_text and user_text and deviation_position != "PERFECT":
            ref_words = reference_text.split()
            usr_words = user_text.split()
            # Find first differing word
            for i, ref_w in enumerate(ref_words):
                if i >= len(usr_words) or ref_w != usr_words[i]:
                    word_hint = f" Try focusing on the word '{ref_w}'."
                    break

        return f"{praise} {score_part} {position_hint}{word_hint}"

    def score(
        self,
        dtw_distance: float,
        per_frame_distances: np.ndarray,
        alignment_path: list[tuple[int, int]],
        user_sequence_length: int,
        scale_factor: float,
        deviation_threshold: float,
        reference_text: str | None = None,
        user_text: str | None = None,
        audio_weight: float = 0.70,
        text_weight: float = 0.30,
    ) -> ScoringResult:
        # ── Audio-level scoring (WavLM + DTW) ──
        similarity = self.distance_to_similarity(dtw_distance, scale_factor)
        accuracy = round(similarity * 100.0, 2)

        threshold = self._resolve_threshold(per_frame_distances, deviation_threshold)
        deviation_position, deviation_index, mismatch_indices = self._detect_deviation(
            per_frame_distances=per_frame_distances,
            alignment_path=alignment_path,
            user_sequence_length=user_sequence_length,
            threshold=threshold,
        )

        # ── Text-level scoring (Wav2Vec2 transcriptions) ──
        cer, wer, text_sim = self._compute_text_metrics(reference_text, user_text)

        # ── Combined ──
        combined_accuracy = self._compute_combined_accuracy(
            audio_similarity=similarity,
            text_similarity=text_sim,
            audio_weight=audio_weight,
            text_weight=text_weight,
        )

        feedback = self._generate_feedback(
            combined_accuracy=combined_accuracy,
            deviation_position=deviation_position,
            reference_text=reference_text,
            user_text=user_text,
        )

        return ScoringResult(
            similarity_score=round(similarity, 4),
            accuracy=accuracy,
            deviation_position=deviation_position,
            deviation_index=deviation_index,
            mismatch_indices=mismatch_indices,
            character_error_rate=cer,
            word_error_rate=wer,
            text_similarity=text_sim,
            combined_accuracy=combined_accuracy,
            feedback_message=feedback,
        )


_scoring_service = ScoringService()


def get_scoring_service() -> ScoringService:
    return _scoring_service
