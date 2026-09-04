"""High-level TAFR-Knee public API."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from ._engine import EvaluationEngine
from .audit import audit_prepared_weight
from .config import KneeConfig, SelectionMode
from .exceptions import ConfigurationError
from .normalization import FrozenNormalizer, fit_normalizer
from .problem import MultiObjectiveProblem
from .reduction import LinearReduction, identity_reduction
from .types import SelectionResult, WeightAudit
from .weights import (
    generate_candidate_weights,
    validate_candidate_weights,
    validate_weight,
)


def audit_weight(
    problem: MultiObjectiveProblem,
    weight: ArrayLike,
    *,
    config: KneeConfig | None = None,
    ideal: ArrayLike | None = None,
    reference: ArrayLike | None = None,
    reduction: LinearReduction | None = None,
) -> WeightAudit:
    """Run the full TAFR audit at one nominal preference weight."""

    normalizer, anchors, reducer, settings = _prepare(
        problem, config=config, ideal=ideal, reference=reference, reduction=reduction
    )
    nominal_weight = validate_weight(weight, reducer.n_reduced)
    if np.any(nominal_weight < settings.min_weight - 1e-12):
        raise ConfigurationError("weight violates config.min_weight")
    engine = EvaluationEngine(
        problem,
        normalizer,
        reducer,
        cache_decimals=settings.cache_decimals,
        n_jobs=settings.n_jobs,
    )
    normalized_anchors = np.vstack([normalizer.transform(anchor.objectives) for anchor in anchors])
    return audit_prepared_weight(engine, nominal_weight, normalized_anchors, settings)


def select_knee(
    problem: MultiObjectiveProblem,
    *,
    config: KneeConfig | None = None,
    candidate_weights: ArrayLike | None = None,
    mode: SelectionMode = "fixed_radius",
    ideal: ArrayLike | None = None,
    reference: ArrayLike | None = None,
    reduction: LinearReduction | None = None,
) -> SelectionResult:
    """Select one certified non-extreme preference-stable knee.

    Returns ``selected=None`` with diagnostics if no candidate is certified;
    it never silently substitutes an uncertified point.
    """

    return _search(
        problem,
        config=config,
        candidate_weights=candidate_weights,
        mode=mode,
        ideal=ideal,
        reference=reference,
        reduction=reduction,
        max_knees=1,
    )


def discover_knees(
    problem: MultiObjectiveProblem,
    *,
    config: KneeConfig | None = None,
    candidate_weights: ArrayLike | None = None,
    mode: SelectionMode = "fixed_radius",
    ideal: ArrayLike | None = None,
    reference: ArrayLike | None = None,
    reduction: LinearReduction | None = None,
    max_knees: int | None = None,
) -> SelectionResult:
    """Return ranked, objective-distinct certified knees and all audits."""

    if max_knees is not None and max_knees < 1:
        raise ConfigurationError("max_knees must be positive or None")
    return _search(
        problem,
        config=config,
        candidate_weights=candidate_weights,
        mode=mode,
        ideal=ideal,
        reference=reference,
        reduction=reduction,
        max_knees=max_knees,
    )


def _search(
    problem: MultiObjectiveProblem,
    *,
    config: KneeConfig | None,
    candidate_weights: ArrayLike | None,
    mode: SelectionMode,
    ideal: ArrayLike | None,
    reference: ArrayLike | None,
    reduction: LinearReduction | None,
    max_knees: int | None,
) -> SelectionResult:
    if mode not in ("fixed_radius", "max_stability"):
        raise ConfigurationError("mode must be 'fixed_radius' or 'max_stability'")
    normalizer, anchors, reducer, settings = _prepare(
        problem, config=config, ideal=ideal, reference=reference, reduction=reduction
    )
    if candidate_weights is None:
        candidates = generate_candidate_weights(
            reducer.n_reduced,
            resolution=settings.candidate_resolution,
            min_weight=settings.min_weight,
            max_candidates=settings.max_candidates,
            seed=settings.random_seed,
        )
    else:
        candidates = validate_candidate_weights(
            candidate_weights,
            n_objectives=reducer.n_reduced,
            min_weight=settings.min_weight,
        )

    engine = EvaluationEngine(
        problem,
        normalizer,
        reducer,
        cache_decimals=settings.cache_decimals,
        n_jobs=settings.n_jobs,
    )
    normalized_anchors = np.vstack([normalizer.transform(anchor.objectives) for anchor in anchors])
    audits = tuple(
        audit_prepared_weight(engine, weight, normalized_anchors, settings) for weight in candidates
    )
    certified = [audit for audit in audits if audit.certified]
    ranked = sorted(certified, key=lambda audit: _rank_key(audit, mode))
    knees = _distinct_knees(ranked, settings.uniqueness_tolerance, max_knees)
    selected = knees[0] if knees else None
    message = (
        "Selected a certified non-extreme preference-stable knee."
        if selected is not None
        else (
            "No certified non-extreme preference-stable knee was found. "
            "Inspect audits and thresholds."
        )
    )
    return SelectionResult(
        selected=selected,
        knees=tuple(knees),
        audits=audits,
        anchors=anchors,
        ideal=normalizer.ideal,
        reference=normalizer.reference,
        mode=mode,
        solver_calls=engine.solver_calls + len(anchors),
        cache_hits=engine.cache_hits,
        message=message,
    )


def _prepare(
    problem: MultiObjectiveProblem,
    *,
    config: KneeConfig | None,
    ideal: ArrayLike | None,
    reference: ArrayLike | None,
    reduction: LinearReduction | None,
) -> tuple[FrozenNormalizer, tuple, LinearReduction, KneeConfig]:
    if not isinstance(problem, MultiObjectiveProblem):
        raise ConfigurationError("problem does not implement the MultiObjectiveProblem protocol")
    normalizer, anchors = fit_normalizer(problem, ideal=ideal, reference=reference)
    reducer = reduction or identity_reduction(problem.n_objectives)
    if reducer.n_original != problem.n_objectives:
        raise ConfigurationError("reduction does not match the problem objective count")
    settings = (config or KneeConfig()).validate(reducer.n_reduced)
    return normalizer, anchors, reducer, settings


def _rank_key(audit: WeightAudit, mode: SelectionMode) -> tuple:
    exit_tradeoff = audit.exit_tradeoff if audit.exit_tradeoff is not None else -np.inf
    stable_weight_key = tuple(np.round(audit.weight, 14).tolist())
    if mode == "fixed_radius":
        return (
            audit.robustness,
            -audit.stability_radius,
            -exit_tradeoff,
            stable_weight_key,
        )
    return (
        -audit.stability_radius,
        audit.robustness,
        -exit_tradeoff,
        stable_weight_key,
    )


def _distinct_knees(
    ranked: Sequence[WeightAudit],
    tolerance: float,
    limit: int | None,
) -> list[WeightAudit]:
    selected: list[WeightAudit] = []
    for candidate in ranked:
        if all(
            np.linalg.norm(candidate.normalized_objectives - existing.normalized_objectives)
            > tolerance
            for existing in selected
        ):
            selected.append(candidate)
            if limit is not None and len(selected) >= limit:
                break
    return selected
