import numpy as np
import pytest

from tafrknee import ConfigurationError, LinearReduction


def test_group_reduction_maps_weights_back_to_original_objectives() -> None:
    reduction = LinearReduction.from_groups([[0, 1], [2, 3]], n_objectives=4)
    np.testing.assert_allclose(reduction.matrix, [[0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5]])
    np.testing.assert_allclose(
        reduction.original_weights([0.25, 0.75]), [0.125, 0.125, 0.375, 0.375]
    )
    np.testing.assert_allclose(reduction.transform([0.2, 0.4, 0.6, 0.8]), [0.3, 0.7])


def test_groups_must_be_a_partition() -> None:
    with pytest.raises(ConfigurationError, match="overlap"):
        LinearReduction.from_groups([[0, 1], [1, 2]], n_objectives=3)
