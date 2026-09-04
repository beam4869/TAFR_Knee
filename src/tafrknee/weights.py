"""Simplex candidate generation and finite perturbation geometry."""

from __future__ import annotations

from itertools import product
from math import ceil, comb, log2

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import qmc

from .exceptions import ConfigurationError


def validate_weight(weight: ArrayLike, n_objectives: int | None = None) -> NDArray[np.float64]:
    """Return a normalized simplex weight after strict validation."""

    values = np.asarray(weight, dtype=float).reshape(-1)
    if n_objectives is not None and values.size != n_objectives:
        raise ConfigurationError(f"weight has length {values.size}; expected {n_objectives}")
    if values.size < 2 or not np.all(np.isfinite(values)) or np.any(values < -1e-12):
        raise ConfigurationError("weights must be a finite non-negative vector of length >= 2")
    total = float(np.sum(values))
    if not np.isclose(total, 1.0, atol=1e-10, rtol=0):
        raise ConfigurationError(f"weights must sum to one, not {total:.16g}")
    values = np.maximum(values, 0.0)
    values /= np.sum(values)
    return values


def simplex_lattice(
    n_objectives: int,
    resolution: int,
    *,
    min_weight: float = 0.0,
) -> NDArray[np.float64]:
    """Generate the complete Das-Dennis simplex lattice."""

    if n_objectives < 2 or resolution < 1:
        raise ConfigurationError("n_objectives must be >= 2 and resolution must be >= 1")
    if min_weight < 0 or n_objectives * min_weight >= 1:
        raise ConfigurationError("min_weight must satisfy 0 <= m * min_weight < 1")

    compositions: list[tuple[int, ...]] = []

    def visit(prefix: tuple[int, ...], remaining: int, dimensions: int) -> None:
        if dimensions == 1:
            compositions.append((*prefix, remaining))
            return
        for value in range(remaining + 1):
            visit((*prefix, value), remaining - value, dimensions - 1)

    visit((), resolution, n_objectives)
    base = np.asarray(compositions, dtype=float) / resolution
    return min_weight + (1.0 - n_objectives * min_weight) * base


def generate_candidate_weights(
    n_objectives: int,
    *,
    resolution: int = 20,
    min_weight: float = 0.0,
    max_candidates: int = 5_000,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Generate a lattice, or a deterministic Sobol-simplex design if it is too large."""

    count = comb(resolution + n_objectives - 1, n_objectives - 1)
    if count <= max_candidates:
        return simplex_lattice(n_objectives, resolution, min_weight=min_weight)
    if max_candidates < 1:
        raise ConfigurationError("max_candidates must be positive")
    exponent = max(1, ceil(log2(max_candidates)))
    unit = qmc.Sobol(d=n_objectives, scramble=True, seed=seed).random_base2(exponent)
    unit = np.clip(unit[:max_candidates], np.finfo(float).eps, 1.0)
    simplex = -np.log(unit)
    simplex /= np.sum(simplex, axis=1, keepdims=True)
    simplex = min_weight + (1.0 - n_objectives * min_weight) * simplex
    centroid = np.full(n_objectives, 1.0 / n_objectives)
    simplex[0] = centroid
    return simplex


def clipped_perturbation_vertices(weight: ArrayLike, radius: float) -> NDArray[np.float64]:
    """Enumerate every vertex of ``{u in simplex: ||u-w||_inf <= radius}``.

    With one equality constraint, every vertex of the clipped box has all but
    at most one coordinate at a bound. Enumerating the free coordinate and all
    bound assignments is therefore complete, including near simplex edges.
    """

    center = validate_weight(weight)
    if radius < 0:
        raise ConfigurationError("radius must be non-negative")
    if radius == 0:
        return center[None, :]
    lower = np.maximum(0.0, center - radius)
    upper = np.minimum(1.0, center + radius)
    vertices: list[NDArray[np.float64]] = []
    n_objectives = center.size
    for free in range(n_objectives):
        fixed = [index for index in range(n_objectives) if index != free]
        for choices in product((0, 1), repeat=n_objectives - 1):
            candidate = np.empty(n_objectives, dtype=float)
            for index, choice in zip(fixed, choices, strict=True):
                candidate[index] = upper[index] if choice else lower[index]
            candidate[free] = 1.0 - float(np.sum(candidate[fixed]))
            if lower[free] - 1e-12 <= candidate[free] <= upper[free] + 1e-12:
                candidate[free] = np.clip(candidate[free], lower[free], upper[free])
                vertices.append(candidate)
    return _unique_rows(vertices or [center])


def pairwise_transfer_weights(weight: ArrayLike, radius: float) -> NDArray[np.float64]:
    """Generate feasible directed pairwise weight transfers up to ``radius``."""

    center = validate_weight(weight)
    if radius < 0:
        raise ConfigurationError("radius must be non-negative")
    points: list[NDArray[np.float64]] = [center]
    for receiver in range(center.size):
        for donor in range(center.size):
            if receiver == donor:
                continue
            transfer = min(radius, 1.0 - center[receiver], center[donor])
            if transfer > 1e-15:
                candidate = center.copy()
                candidate[receiver] += transfer
                candidate[donor] -= transfer
                points.append(candidate)
    return _unique_rows(points)


def hit_and_run_samples(
    weight: ArrayLike,
    radius: float,
    n_samples: int,
    *,
    seed: int = 0,
    burn_in: int | None = None,
    start: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Draw deterministic-seed interior samples from a clipped simplex box."""

    center = validate_weight(weight)
    if radius < 0 or n_samples < 0:
        raise ConfigurationError("radius and n_samples must be non-negative")
    if n_samples == 0 or radius == 0:
        return np.empty((0, center.size), dtype=float)
    lower = np.maximum(0.0, center - radius)
    upper = np.minimum(1.0, center + radius)
    rng = np.random.default_rng(seed)
    point = center.copy() if start is None else validate_weight(start, center.size)
    if np.any(point < lower - 1e-12) or np.any(point > upper + 1e-12):
        raise ConfigurationError("hit-and-run start lies outside the clipped neighborhood")
    samples: list[NDArray[np.float64]] = []
    burn = max(5, 2 * center.size) if burn_in is None else burn_in
    total = burn + n_samples
    for iteration in range(total):
        direction = rng.normal(size=center.size)
        direction -= np.mean(direction)
        norm = np.linalg.norm(direction)
        if norm <= 1e-15:
            continue
        direction /= norm
        low_t = -np.inf
        high_t = np.inf
        for value, component, lo, hi in zip(point, direction, lower, upper, strict=True):
            if component > 1e-15:
                low_t = max(low_t, (lo - value) / component)
                high_t = min(high_t, (hi - value) / component)
            elif component < -1e-15:
                low_t = max(low_t, (hi - value) / component)
                high_t = min(high_t, (lo - value) / component)
        if not np.isfinite(low_t) or not np.isfinite(high_t) or high_t < low_t:
            continue
        point = point + rng.uniform(low_t, high_t) * direction
        point = np.maximum(point, 0.0)
        point /= np.sum(point)
        if iteration >= burn:
            samples.append(point.copy())
    return np.asarray(samples[:n_samples], dtype=float)


def perturbation_weights(
    weight: ArrayLike,
    radius: float,
    *,
    strategy: str = "hybrid",
    interior_samples: int = 16,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Generate finite-radius audit weights using the requested hierarchy."""

    center = validate_weight(weight)
    if strategy == "pairwise":
        points = pairwise_transfer_weights(center, radius)
    elif strategy == "vertices":
        points = clipped_perturbation_vertices(center, radius)
    elif strategy in {"hybrid", "adaptive"}:
        chunks = [
            clipped_perturbation_vertices(center, radius),
            pairwise_transfer_weights(center, radius),
        ]
        samples = hit_and_run_samples(center, radius, interior_samples, seed=seed)
        if samples.size:
            chunks.append(samples)
        points = np.vstack(chunks)
    else:
        raise ConfigurationError("unknown perturbation strategy")
    return _unique_rows([center, *points])


def max_simplex_radius(weight: ArrayLike) -> float:
    """Maximum infinity-norm distance from ``weight`` to the simplex."""

    center = validate_weight(weight)
    distances = [np.max(np.abs(np.eye(center.size)[i] - center)) for i in range(center.size)]
    return float(max(distances))


def validate_candidate_weights(
    weights: ArrayLike,
    *,
    n_objectives: int,
    min_weight: float = 0.0,
) -> NDArray[np.float64]:
    values = np.asarray(weights, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != n_objectives or values.shape[0] < 1:
        raise ConfigurationError("candidate_weights has an invalid shape")
    checked = np.vstack([validate_weight(row, n_objectives) for row in values])
    if np.any(checked < min_weight - 1e-12):
        raise ConfigurationError("candidate weight violates min_weight")
    return _unique_rows(checked)


def _unique_rows(rows: list[ArrayLike] | NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(rows, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    rounded = np.round(values, decimals=14)
    _, indices = np.unique(rounded, axis=0, return_index=True)
    return values[np.sort(indices)]
