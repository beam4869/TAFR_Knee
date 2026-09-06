"""Derivative correctness and isolation from the official reproduction."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "external/snee/functions.py").exists(),
    reason="Pinned SNEE submodule is required")

def make_upstream(*args, **kwargs):
    from experiments.methods.snee_adapter import make_upstream as upstream
    return upstream(*args, **kwargs)


@pytest.mark.parametrize("normalized", [False, True])
def test_grv2_corrected_hessian_matches_gradient_difference(normalized):
    f, *_ = make_upstream("GRV2", normalized=normalized, corrected_grv2_hessian=True)
    for x in (np.array([.3, 1.2]), np.array([-.4, 2.3])):
        for i in range(2):
            step = 1e-5
            finite_difference = np.column_stack([
                (f.prob.grad_f_dict[i]((x + step * e)[:, None]) -
                 f.prob.grad_f_dict[i]((x - step * e)[:, None])).ravel() / (2 * step)
                for e in np.eye(2)])
            np.testing.assert_allclose(f.prob.hess_f_dict[i](x[:, None]),
                                       finite_difference, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(f.prob.hess_f_dict[i](x), finite_difference,
                                       rtol=1e-7, atol=1e-7)


def test_default_upstream_hessian_is_preserved():
    x = np.array([[.3], [1.2]])
    original, *_ = make_upstream("GRV2")
    corrected, *_ = make_upstream("GRV2", corrected_grv2_hessian=True)
    restored, *_ = make_upstream("GRV2")
    np.testing.assert_array_equal(original.prob.hess_f_dict[0](x),
                                  restored.prob.hess_f_dict[0](x))
    assert not np.allclose(original.prob.hess_f_dict[0](x),
                           corrected.prob.hess_f_dict[0](x))
