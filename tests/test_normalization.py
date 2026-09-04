import numpy as np
import pytest

from tafrknee import ConfigurationError, FrozenNormalizer, TabularProblem, fit_normalizer


def test_payoff_matrix_fits_and_freezes_normalization() -> None:
    problem = TabularProblem([[0.0, 10.0], [4.0, 4.0], [10.0, 0.0]])
    normalizer, anchors = fit_normalizer(problem)
    np.testing.assert_allclose(normalizer.ideal, [0.0, 0.0])
    np.testing.assert_allclose(normalizer.reference, [10.0, 10.0])
    np.testing.assert_allclose(normalizer.transform([4.0, 4.0]), [0.4, 0.4])
    assert len(anchors) == 2
    with pytest.raises(ValueError):
        normalizer.ideal[0] = 2.0


def test_physical_scalar_weights_preserve_normalized_weighted_sum() -> None:
    normalizer = FrozenNormalizer([10.0, 100.0], [20.0, 300.0])
    coefficients = normalizer.physical_scalar_weights([0.25, 0.75])
    left = coefficients @ np.array([15.0, 200.0])
    right = np.array([0.25, 0.75]) @ normalizer.transform([15.0, 200.0])
    offset = coefficients @ normalizer.ideal
    assert left - offset == pytest.approx(right)


def test_constant_objective_requires_removal_or_engineering_scale() -> None:
    problem = TabularProblem([[0.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ConfigurationError, match="constant objectives"):
        fit_normalizer(problem)
