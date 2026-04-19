from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


@dataclass(slots=True)
class DTWResult:
    distance: float
    path: list[tuple[int, int]]
    per_frame_distances: np.ndarray


class DTWService:
    def compute_alignment(self, reference_embeddings: np.ndarray, user_embeddings: np.ndarray) -> DTWResult:
        if reference_embeddings.ndim != 2 or user_embeddings.ndim != 2:
            raise ValueError("Embedding inputs must both be 2D arrays.")

        distance, path = fastdtw(reference_embeddings, user_embeddings, dist=euclidean)

        per_frame_distances = np.array(
            [np.linalg.norm(reference_embeddings[i] - user_embeddings[j]) for i, j in path],
            dtype=np.float32,
        )

        return DTWResult(distance=float(distance), path=path, per_frame_distances=per_frame_distances)


_dtw_service = DTWService()


def get_dtw_service() -> DTWService:
    return _dtw_service
