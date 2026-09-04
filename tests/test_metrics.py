import numpy as np
import pytest

from tafrknee.metrics import anchor_distance, tradeoff


def test_anchor_distance_is_zero_at_an_anchor() -> None:
    anchors = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert anchor_distance([0.0, 1.0], anchors) == pytest.approx(0.0)


def test_duplicate_anchors_do_not_change_distance() -> None:
    anchors = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    expected = np.sqrt(0.5)
    assert anchor_distance([0.5, 0.5], anchors) == pytest.approx(expected)


def test_tradeoff_active_ratio_for_plateau_exit() -> None:
    value = tradeoff([0.4, 0.4], [0.0, 1.0])
    assert value.active
    assert value.improvement == pytest.approx(0.4)
    assert value.deterioration == pytest.approx(0.6)
    assert value.ratio == pytest.approx(1.5)


def test_one_sided_change_is_not_tradeoff_active() -> None:
    value = tradeoff([0.4, 0.4], [0.3, 0.4])
    assert not value.active
    assert value.deteriorated_objectives == 0
