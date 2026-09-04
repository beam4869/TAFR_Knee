"""Immutable data structures returned by the public API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _readonly_vector(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=float).reshape(-1).copy()
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    array.setflags(write=False)
    return array


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


@dataclass(frozen=True)
class SolveResult:
    """Result of one scalarized lower-level optimization."""

    decision: Any
    objectives: NDArray[np.float64]
    status: str = "optimal"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "objectives", _readonly_vector(self.objectives, name="objectives"))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": _jsonable(self.decision),
            "objectives": self.objectives.tolist(),
            "status": self.status,
            "metadata": _jsonable(self.metadata),
        }


@dataclass(frozen=True)
class WeightAudit:
    """Diagnostics and certification state for one nominal preference weight."""

    weight: NDArray[np.float64]
    solution: SolveResult
    normalized_objectives: NDArray[np.float64]
    robustness: float
    stability_radius: float
    exit_tradeoff: float | None
    anchor_distance: float
    non_extreme: bool
    tradeoff_active: bool
    has_exit: bool
    certified: bool
    perturbation_count: int
    exit_count: int
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "weight", _readonly_vector(self.weight, name="weight"))
        object.__setattr__(
            self,
            "normalized_objectives",
            _readonly_vector(self.normalized_objectives, name="normalized_objectives"),
        )
        object.__setattr__(self, "reasons", tuple(self.reasons))

    @property
    def decision(self) -> Any:
        return self.solution.decision

    @property
    def objectives(self) -> NDArray[np.float64]:
        return self.solution.objectives

    def to_dict(self) -> dict[str, Any]:
        return {
            "weight": self.weight.tolist(),
            "decision": _jsonable(self.decision),
            "objectives": self.objectives.tolist(),
            "normalized_objectives": self.normalized_objectives.tolist(),
            "robustness": self.robustness,
            "stability_radius": self.stability_radius,
            "exit_tradeoff": self.exit_tradeoff,
            "anchor_distance": self.anchor_distance,
            "non_extreme": self.non_extreme,
            "tradeoff_active": self.tradeoff_active,
            "has_exit": self.has_exit,
            "certified": self.certified,
            "perturbation_count": self.perturbation_count,
            "exit_count": self.exit_count,
            "reasons": list(self.reasons),
            "solve_metadata": _jsonable(self.solution.metadata),
        }


@dataclass(frozen=True)
class SelectionResult:
    """Complete result of a knee selection or discovery run."""

    selected: WeightAudit | None
    knees: tuple[WeightAudit, ...]
    audits: tuple[WeightAudit, ...]
    anchors: tuple[SolveResult, ...]
    ideal: NDArray[np.float64]
    reference: NDArray[np.float64]
    mode: str
    solver_calls: int
    cache_hits: int
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "knees", tuple(self.knees))
        object.__setattr__(self, "audits", tuple(self.audits))
        object.__setattr__(self, "anchors", tuple(self.anchors))
        object.__setattr__(self, "ideal", _readonly_vector(self.ideal, name="ideal"))
        object.__setattr__(self, "reference", _readonly_vector(self.reference, name="reference"))

    @property
    def found(self) -> bool:
        return self.selected is not None

    def to_dict(self, *, include_audits: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "found": self.found,
            "selected": None if self.selected is None else self.selected.to_dict(),
            "knees": [knee.to_dict() for knee in self.knees],
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "ideal": self.ideal.tolist(),
            "reference": self.reference.tolist(),
            "mode": self.mode,
            "solver_calls": self.solver_calls,
            "cache_hits": self.cache_hits,
            "message": self.message,
        }
        if include_audits:
            result["audits"] = [audit.to_dict() for audit in self.audits]
        return result
