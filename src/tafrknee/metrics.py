"""Objective-space robustness and trade-off metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


def objective_distance(left: ArrayLike, right: ArrayLike, order: float | int = 2) -> float:
    return float(
        np.linalg.norm(np.asarray(left, dtype=float) - np.asarray(right, dtype=float), ord=order)
    )


def anchor_distance(
    objectives: ArrayLike,
    anchor_objectives: ArrayLike,
    *,
    order: float | int = 2,
    distinct_tolerance: float = 1e-12,
) -> float:
    """Distance to the nearest distinct anchor objective vector."""

    point = np.asarray(objectives, dtype=float).reshape(-1)
    anchors = np.asarray(anchor_objectives, dtype=float)
    unique: list[np.ndarray] = []
    for anchor in anchors:
        if all(np.linalg.norm(anchor - kept, ord=order) > distinct_tolerance for kept in unique):
            unique.append(anchor)
    distances = np.linalg.norm(np.asarray(unique) - point, ord=order, axis=1)
    return float(np.min(distances))


@dataclass(frozen=True)
class Tradeoff:
    improvement: float
    deterioration: float
    ratio: float
    active: bool
    improved_objectives: int
    deteriorated_objectives: int


def tradeoff(
    nominal: ArrayLike,
    alternative: ArrayLike,
    *,
    min_improvement: float = 1e-6,
    min_deterioration: float = 1e-6,
    safeguard: float = 1e-12,
) -> Tradeoff:
    """Compute minimization-oriented exit improvement and deterioration."""

    delta = np.asarray(alternative, dtype=float) - np.asarray(nominal, dtype=float)
    improvements = np.maximum(-delta, 0.0)
    deteriorations = np.maximum(delta, 0.0)
    improvement = float(np.sum(improvements))
    deterioration = float(np.sum(deteriorations))
    improved_count = int(np.count_nonzero(improvements > min_improvement))
    deteriorated_count = int(np.count_nonzero(deteriorations > min_deterioration))
    active = (
        improvement >= min_improvement
        and deterioration >= min_deterioration
        and improved_count > 0
        and deteriorated_count > 0
    )
    return Tradeoff(
        improvement=improvement,
        deterioration=deterioration,
        ratio=deterioration / (improvement + safeguard),
        active=active,
        improved_objectives=improved_count,
        deteriorated_objectives=deteriorated_count,
    )
