from math import comb

import numpy as np
import pytest

from tafrknee import clipped_perturbation_vertices, simplex_lattice
from tafrknee.weights import (
    generate_candidate_weights,
    pairwise_transfer_weights,
    perturbation_weights,
)


def test_simplex_lattice_is_complete_and_feasible() -> None:
    weights = simplex_lattice(3, 4)
    assert weights.shape == (comb(6, 2), 3)
    np.testing.assert_allclose(np.sum(weights, axis=1), 1.0)
    assert np.all(weights >= 0)


def test_min_weight_affinely_shrinks_lattice() -> None:
    weights = simplex_lattice(3, 3, min_weight=0.1)
    assert np.min(weights) == pytest.approx(0.1)
    np.testing.assert_allclose(np.sum(weights, axis=1), 1.0)


@pytest.mark.parametrize(
    ("n_objectives", "expected"),
    [(4, comb(4, 2)), (5, 5 * comb(4, 2))],
)
def test_full_interior_vertex_counts(n_objectives: int, expected: int) -> None:
    center = np.full(n_objectives, 1.0 / n_objectives)
    vertices = clipped_perturbation_vertices(center, 0.05)
    assert vertices.shape == (expected, n_objectives)
    np.testing.assert_allclose(np.sum(vertices, axis=1), 1.0, atol=1e-14)
    np.testing.assert_allclose(np.max(np.abs(vertices - center), axis=1), 0.05)


def test_clipped_vertices_obey_actual_boundary_geometry() -> None:
    center = np.array([0.0, 0.25, 0.75])
    vertices = clipped_perturbation_vertices(center, 0.2)
    assert len(vertices) >= 3
    assert np.all(vertices >= -1e-14)
    assert np.all(vertices <= 1.0 + 1e-14)
    assert np.all(np.abs(vertices - center) <= 0.2 + 1e-14)
    np.testing.assert_allclose(np.sum(vertices, axis=1), 1.0)


def test_pairwise_and_hybrid_audits_are_deterministic() -> None:
    center = np.array([0.2, 0.3, 0.5])
    pairwise = pairwise_transfer_weights(center, 0.1)
    first = perturbation_weights(center, 0.1, strategy="hybrid", interior_samples=10, seed=7)
    second = perturbation_weights(center, 0.1, strategy="hybrid", interior_samples=10, seed=7)
    assert len(first) > len(pairwise)
    np.testing.assert_allclose(first, second)


def test_adaptive_initial_design_is_feasible() -> None:
    center = np.array([0.2, 0.3, 0.5])
    points = perturbation_weights(center, 0.1, strategy="adaptive", interior_samples=5, seed=2)
    assert points.shape[1] == 3
    assert np.all(points >= 0)
    assert np.all(np.abs(points - center) <= 0.1 + 1e-12)


def test_sobol_fallback_respects_candidate_limit() -> None:
    weights = generate_candidate_weights(7, resolution=20, max_candidates=64, seed=3)
    assert weights.shape == (64, 7)
    np.testing.assert_allclose(weights[0], np.full(7, 1 / 7))
    np.testing.assert_allclose(np.sum(weights, axis=1), 1.0)
