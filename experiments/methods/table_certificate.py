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
    """Conservative near-optimal cells, including ambiguous deterministic ties.

    A row selected by the 1e-12 tie rule must have cost <= every other cost +
    1e-12. Keeping ALL such feasible cells gives a superset of actual outputs;
    removing small strict-margin cells would not provide an upper bound.
    Guarantees remain numerical, subject to the declared LP feasibility tolerance.
    """

    def _lp(self, *args, **kwargs):
        result = super()._lp(*args, **kwargs)
        if result.status not in (0, 2):  # optimal or proved infeasible
            raise RuntimeError(f"Unresolved cell LP ({result.status}): {result.message}")
        return result

    def reachable(self, center, radius):
        center = np.asarray(center, float)
        lower, upper = np.maximum(0, center - radius), np.minimum(1, center + radius)
        whole = bool(np.all(lower == 0) and np.all(upper == 1))
        if whole and self._whole_simplex_reachability is not None:
            return tuple(a.copy() for a in self._whole_simplex_reachability)
        indices, witnesses = [], []
        for index in range(self.n):
            result = self._lp(np.zeros(self.m), A_ub=self.y[index] - self.y,
                              b_ub=np.full(self.n, 1e-12), A_eq=[np.ones(self.m)],
                              b_eq=[1], bounds=list(zip(lower, upper, strict=True)))
            if result.success:
                indices.append(index)
                witnesses.append(result.x)
        answer = np.asarray(indices, int), np.asarray(witnesses)
        if whole:
            self._whole_simplex_reachability = tuple(a.copy() for a in answer)
        return answer

    def audit(self, center, radius):
        result = super().audit(center, radius)
        # An LP witness may belong to a tie-losing row or sit slightly outside
        # the requested ball. Count only directly checked feasible witnesses as
        # lower-bound evidence. The upper envelope retains every potential row.
        center = np.asarray(center, float)
        witnessed = [result["nominal"]]
        for weight in result["witnesses"]:
            if (np.all(weight >= 0) and np.all(weight <= 1)
                    and abs(weight.sum() - 1) <= 1e-14
                    and np.max(np.abs(weight - center)) <= radius):
                witnessed.append(self.select(weight))
        result["lower_bound"] = float(np.max(np.linalg.norm(
            self.y[witnessed] - self.y[result["nominal"]], axis=1)))
        return result

    def stability_radius_fast(self, center, epsilon=0.0):
        """Radius for the conservative cell union, using LPs without strict cutoffs."""
        center = np.asarray(center, float)
        nominal = self.select(center)
        possible, _ = self.reachable(center, 1.)
        best = max_simplex_radius(center)
        for index in possible:
            if np.linalg.norm(self.y[index] - self.y[nominal]) <= epsilon:
                continue
            cell = np.column_stack([self.y[index] - self.y, np.zeros(self.n)])
            positive = np.column_stack([np.eye(self.m), -np.ones(self.m)])
            negative = np.column_stack([-np.eye(self.m), -np.ones(self.m)])
            result = self._lp(np.r_[np.zeros(self.m), 1.],
                              A_ub=np.vstack([cell, positive, negative]),
                              b_ub=np.r_[np.full(self.n, 1e-12), center, -center],
                              A_eq=[np.r_[np.ones(self.m), 0]], b_eq=[1],
                              bounds=[(0, 1)] * (self.m + 1))
            if result.success:
                best = min(best, float(result.x[-1]))
        return best


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
                  radius_status="unverified", R_upper=None, conservative_stability_radius=None,
                  exit_radius=None, exits=[], exit_tradeoff=None, all_exits_active=False,
                  accepted=False, reasons=[], evidence_scope="supplied finite weighted-sum table",
                  lp_tolerance=oracle.tolerance, tie_tolerance=1e-12)
    if not result["non_extreme"]:
        result["reasons"].append("too_close_to_anchor")
        return result
    exact = oracle.audit(w, config.radius)
    result.update(R_upper=exact["robustness"], worst_weight=exact["worst_weight"],
                  worst_index=exact["worst_index"], reachable_count=len(exact["reachable"]))
    result["R_witness"] = exact["lower_bound"]
    result["bound_method"] = "conservative near-optimal cell union; LP tolerance 1e-9"
    result["radius_status"] = radius_evidence(config.objective_tolerance,
                                             lower_bound=exact["lower_bound"],
                                             upper_bound=exact["robustness"])
    if result["radius_status"] != "verified":
        result["reasons"].append("fixed_radius_bound_exceeded" if
                                 result["radius_status"] == "refuted" else
                                 "fixed_radius_bound_unresolved")
        return result
    rho = oracle.stability_radius_fast(w, config.objective_tolerance)
    result["conservative_stability_radius"] = rho
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
