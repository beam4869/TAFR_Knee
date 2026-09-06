"""Experimental finite-table evidence; does not change the production API.

The certificate covers a supplied, frozen, normalized weighted-sum table only,
under the existing oracle's declared LP and tie tolerances. It is not evidence
about an unseen continuous front or a heuristic lower-level solver.
"""

from dataclasses import asdict

import numpy as np

from experiments.metrics.exact_tabular_oracle import ExactTabularOracle
from tafrknee.metrics import tradeoff
from tafrknee.weights import max_simplex_radius, validate_weight


def radius_evidence(epsilon, *, lower_bound=None, upper_bound=None):
    """A passing sample cannot certify a supremum; a violating sample refutes it."""
    values = [epsilon, *[x for x in (lower_bound, upper_bound) if x is not None]]
    if not all(np.isfinite(x) and x >= 0 for x in values):
        raise ValueError("Evidence bounds and tolerance must be finite and nonnegative")
    if lower_bound is not None and upper_bound is not None and lower_bound > upper_bound:
        raise ValueError("Inconsistent evidence bounds")
    if lower_bound is not None and lower_bound > epsilon:
        return "refuted"
    if upper_bound is not None and upper_bound <= epsilon:
        return "verified"
    return "unverified"


class CheckedTableOracle(ExactTabularOracle):
    """Fail closed on an LP error instead of silently omitting an output cell."""

    def _lp(self, *args, **kwargs):
        result = super()._lp(*args, **kwargs)
        if result.status not in (0, 2):  # optimal or proved infeasible
            raise RuntimeError(f"Unresolved cell LP ({result.status}): {result.message}")
        return result


def certify_table_weight(oracle, weight, config):
    """Require non-extremeness, a fixed-radius bound, and every first-shell exit.

    The first nonempty shell is at rho_epsilon + j*exit_step, j=1..exit_layers,
    clipped to the simplex. It includes every reachable distinct objective output
    outside the epsilon ball. An inactive exit fails this experimental criterion;
    it is never discarded in favor of a later active exit. This is a deliberately
    stronger research variant, not an equivalence to the original knee definition.
    """
    if not isinstance(oracle, CheckedTableOracle):
        raise TypeError("Certification requires the fail-closed table oracle")
    config.validate(oracle.m)
    if config.distance_norm != 2 or config.min_weight != 0:
        raise ValueError("This oracle supports Euclidean distances and min_weight=0 only")
    w = validate_weight(weight, oracle.m)
    y = oracle.y
    nominal = oracle.select(w)
    anchors = y[[oracle.select(a) for a in np.eye(oracle.m)]]
    distance = float(np.min(np.linalg.norm(anchors - y[nominal], axis=1)))
    result = dict(weight=w, nominal_index=nominal, objectives=y[nominal],
                  anchor_distance=distance, non_extreme=distance >= config.extreme_threshold,
                  radius_status="unverified", R_exact=None, exact_stability_radius=None,
                  exit_radius=None, exits=[], exit_tradeoff=None, all_exits_active=False,
                  accepted=False, reasons=[], evidence_scope="supplied finite weighted-sum table",
                  lp_tolerance=oracle.tolerance, tie_tolerance=1e-12)
    if not result["non_extreme"]:
        result["reasons"].append("too_close_to_anchor")
        return result
    exact = oracle.audit(w, config.radius)
    result.update(R_exact=exact["robustness"], worst_weight=exact["worst_weight"],
                  worst_index=exact["worst_index"], reachable_count=len(exact["reachable"]))
    result["radius_status"] = radius_evidence(config.objective_tolerance,
                                             lower_bound=exact["robustness"],
                                             upper_bound=exact["robustness"])
    if result["radius_status"] != "verified":
        result["reasons"].append("fixed_radius_bound_exceeded")
        return result
    rho = oracle.stability_radius_fast(w, config.objective_tolerance)
    result["exact_stability_radius"] = rho
    maximum = max_simplex_radius(w)
    for layer in range(1, config.exit_layers + 1):
        radius = min(maximum, rho + layer * config.exit_step)
        if radius <= rho + config.tie_tolerance:
            break
        indices, witnesses = oracle.reachable(w, radius)
        seen = set()
        for index, witness in zip(indices, witnesses, strict=True):
            key = tuple(y[index])
            if key in seen or np.linalg.norm(y[index] - y[nominal]) <= config.objective_tolerance:
                continue
            seen.add(key)
            value = tradeoff(y[nominal], y[index], min_improvement=config.min_improvement,
                             min_deterioration=config.min_deterioration)
            result["exits"].append(dict(index=int(index), objectives=y[index], weight=witness,
                                        **asdict(value)))
        if result["exits"]:
            result["exit_radius"] = radius
            break
        if radius == maximum:
            break
    exits = result["exits"]
    if not exits:
        result["reasons"].append("no_stability_region_exit")
    else:
        result["all_exits_active"] = all(e["active"] for e in exits)
        result["exit_tradeoff"] = min(e["ratio"] for e in exits)
        if not result["all_exits_active"]:
            result["reasons"].append("inactive_first_shell_exit")
        if result["exit_tradeoff"] < config.kappa_min:
            result["reasons"].append("exit_tradeoff_below_threshold")
    result["accepted"] = not result["reasons"]
    return result
