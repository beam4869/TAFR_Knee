"""Finite-radius robustness, plateau, and exit certification."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._engine import EvaluationEngine
from .config import KneeConfig
from .metrics import anchor_distance, objective_distance, tradeoff
from .types import WeightAudit
from .weights import (
    hit_and_run_samples,
    max_simplex_radius,
    perturbation_weights,
    validate_weight,
)


def audit_prepared_weight(
    engine: EvaluationEngine,
    weight: NDArray[np.float64],
    normalized_anchors: NDArray[np.float64],
    config: KneeConfig,
) -> WeightAudit:
    """Audit one weight using an already fitted normalization and solve cache."""

    nominal_weight = validate_weight(weight, engine.reduction.n_reduced)
    nominal_solution = engine.solve(nominal_weight)
    nominal = engine.normalizer.transform(nominal_solution.objectives)
    d_ext = anchor_distance(
        nominal,
        normalized_anchors,
        order=config.distance_norm,
        distinct_tolerance=config.tie_tolerance,
    )
    non_extreme = d_ext >= config.extreme_threshold

    robustness, perturbation_count = _radius_robustness(
        engine, nominal_weight, nominal, config.radius, config, seed_offset=0
    )
    stability_radius = _stability_radius(engine, nominal_weight, nominal, config)
    exit_tradeoff, has_exit, active, exit_count = _exit_tradeoff(
        engine, nominal_weight, nominal, stability_radius, config
    )

    reasons: list[str] = []
    if not non_extreme:
        reasons.append("too_close_to_anchor")
    if not has_exit:
        reasons.append("no_stability_region_exit")
    elif not active:
        reasons.append("no_tradeoff_active_exit")
    if active and exit_tradeoff is not None and exit_tradeoff < config.kappa_min:
        reasons.append("exit_tradeoff_below_threshold")
    certified = (
        non_extreme
        and has_exit
        and active
        and exit_tradeoff is not None
        and exit_tradeoff >= config.kappa_min
    )
    return WeightAudit(
        weight=nominal_weight,
        solution=nominal_solution,
        normalized_objectives=nominal,
        robustness=robustness,
        stability_radius=stability_radius,
        exit_tradeoff=exit_tradeoff,
        anchor_distance=d_ext,
        non_extreme=non_extreme,
        tradeoff_active=active,
        has_exit=has_exit,
        certified=certified,
        perturbation_count=perturbation_count,
        exit_count=exit_count,
        reasons=tuple(reasons),
    )


def _radius_robustness(
    engine: EvaluationEngine,
    weight: NDArray[np.float64],
    nominal: NDArray[np.float64],
    radius: float,
    config: KneeConfig,
    *,
    seed_offset: int,
) -> tuple[float, int]:
    audit_weights = perturbation_weights(
        weight,
        radius,
        strategy=config.audit_strategy,
        interior_samples=config.interior_samples,
        seed=config.random_seed + seed_offset,
    )
    results = engine.solve_many(audit_weights)
    distances = [
        objective_distance(
            engine.normalizer.transform(result.objectives),
            nominal,
            config.distance_norm,
        )
        for result in results
    ]
    evaluated = len(results)
    if config.audit_strategy == "adaptive" and config.adaptive_rounds:
        best_index = int(np.argmax(distances))
        best_weight = audit_weights[best_index]
        best_distance = distances[best_index]
        for round_index in range(config.adaptive_rounds):
            proposals = hit_and_run_samples(
                weight,
                radius,
                config.adaptive_samples,
                seed=config.random_seed + seed_offset + 1_000 * (round_index + 1),
                burn_in=0,
                start=best_weight,
            )
            if not proposals.size:
                continue
            proposal_results = engine.solve_many(proposals)
            proposal_distances = [
                objective_distance(
                    engine.normalizer.transform(result.objectives),
                    nominal,
                    config.distance_norm,
                )
                for result in proposal_results
            ]
            evaluated += len(proposal_results)
            proposal_best = int(np.argmax(proposal_distances))
            if proposal_distances[proposal_best] > best_distance:
                best_distance = proposal_distances[proposal_best]
                best_weight = proposals[proposal_best]
        distances.append(best_distance)
    return float(max(distances, default=0.0)), evaluated


def _stability_radius(
    engine: EvaluationEngine,
    weight: NDArray[np.float64],
    nominal: NDArray[np.float64],
    config: KneeConfig,
) -> float:
    low = 0.0
    high = max_simplex_radius(weight)
    high_robustness, _ = _radius_robustness(
        engine, weight, nominal, high, config, seed_offset=10_000
    )
    if high_robustness <= config.objective_tolerance:
        return high
    for iteration in range(config.stability_iterations):
        if high - low <= config.stability_tolerance:
            break
        midpoint = (low + high) / 2.0
        robustness, _ = _radius_robustness(
            engine,
            weight,
            nominal,
            midpoint,
            config,
            seed_offset=20_000 + iteration,
        )
        if robustness <= config.objective_tolerance:
            low = midpoint
        else:
            high = midpoint
    return low


def _exit_tradeoff(
    engine: EvaluationEngine,
    weight: NDArray[np.float64],
    nominal: NDArray[np.float64],
    stability_radius: float,
    config: KneeConfig,
) -> tuple[float | None, bool, bool, int]:
    maximum = max_simplex_radius(weight)
    ratios: list[float] = []
    has_exit = False
    evaluated = 0
    seen: set[tuple[float, ...]] = set()
    for layer in range(1, config.exit_layers + 1):
        radius = min(maximum, stability_radius + layer * config.exit_step)
        if radius <= stability_radius + config.tie_tolerance:
            continue
        exit_weights = perturbation_weights(
            weight,
            radius,
            strategy=config.audit_strategy,
            interior_samples=config.interior_samples,
            seed=config.random_seed + 30_000 + layer,
        )
        for exit_weight, result in zip(exit_weights, engine.solve_many(exit_weights), strict=True):
            key = tuple(np.round(exit_weight, config.cache_decimals).tolist())
            if key in seen:
                continue
            seen.add(key)
            alternative = engine.normalizer.transform(result.objectives)
            distance = objective_distance(alternative, nominal, config.distance_norm)
            if distance <= config.objective_tolerance:
                continue
            has_exit = True
            evaluated += 1
            value = tradeoff(
                nominal,
                alternative,
                min_improvement=config.min_improvement,
                min_deterioration=config.min_deterioration,
            )
            if value.active:
                ratios.append(value.ratio)
        if ratios:
            break
        if np.isclose(radius, maximum):
            break
    return (min(ratios) if ratios else None, has_exit, bool(ratios), evaluated)
