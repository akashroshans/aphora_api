from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np


@dataclass(slots=True)
class ScoringResult:
    similarity_score: float
    accuracy: float
    deviation_position: str
    deviation_index: int | None
    mismatch_indices: list[int]


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

    def score(
        self,
        dtw_distance: float,
        per_frame_distances: np.ndarray,
        alignment_path: list[tuple[int, int]],
        user_sequence_length: int,
        scale_factor: float,
        deviation_threshold: float,
    ) -> ScoringResult:
        similarity = self.distance_to_similarity(dtw_distance, scale_factor)
        accuracy = round(similarity * 100.0, 2)

        threshold = self._resolve_threshold(per_frame_distances, deviation_threshold)
        deviation_position, deviation_index, mismatch_indices = self._detect_deviation(
            per_frame_distances=per_frame_distances,
            alignment_path=alignment_path,
            user_sequence_length=user_sequence_length,
            threshold=threshold,
        )

        return ScoringResult(
            similarity_score=round(similarity, 4),
            accuracy=accuracy,
            deviation_position=deviation_position,
            deviation_index=deviation_index,
            mismatch_indices=mismatch_indices,
        )


_scoring_service = ScoringService()


def get_scoring_service() -> ScoringService:
    return _scoring_service
