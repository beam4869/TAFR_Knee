"""LP ground truth for a finite weighted-sum table.

Reachability uses the deterministic lexicographic objective-row tie order. A weak
cell intersection alone can include a row that loses every tie; maximizing strict
margin against higher-priority rows excludes those spurious reachable points.
The stability radius is a supremum, so limits approaching a tie boundary count.
HiGHS numerical tolerances limit the word 'exact' throughout these experiments.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

LP_OPTIONS = dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9)


def canonical_priorities(y):
    y = np.asarray(y)
    order = np.lexsort(tuple(y[:, j] for j in reversed(range(y.shape[1]))))
    ranks = np.empty(len(y), int)
    ranks[order] = np.arange(len(y))
    return ranks


class ExactTabularOracle:
    def __init__(self, objectives, *, priorities=None, tolerance=1e-9):
        self.y = np.asarray(objectives, float)
        if self.y.ndim != 2 or not np.all(np.isfinite(self.y)):
            raise ValueError("A finite objective matrix is required")
        self.n, self.m = self.y.shape
        self.priorities = canonical_priorities(self.y) if priorities is None else np.asarray(priorities)
        self.tolerance = tolerance
        self.lp_calls = 0

    def _lp(self, *args, **kwargs):
        self.lp_calls += 1
        return linprog(*args, method="highs", options=LP_OPTIONS, **kwargs)

    def select(self, weight):
        cost = self.y @ np.asarray(weight)
        tied = np.flatnonzero(cost <= cost.min() + 1e-12)
        return int(tied[np.argmin(self.priorities[tied])])

    def reachable(self, center, radius):
        center = np.asarray(center, float)
        lower = np.maximum(0, center - radius)
        upper = np.minimum(1, center + radius)
        reached, witnesses = [], []
        for s in range(self.n):
            competitors = np.arange(self.n) != s
            differences = self.y[s] - self.y[competitors]
            strict = (self.priorities[competitors] < self.priorities[s]).astype(float)
            # Maximize a common strict margin against rows winning ties.
            A = np.column_stack([differences, strict])
            res = self._lp(np.r_[np.zeros(self.m), -1.0], A_ub=A,
                           b_ub=np.zeros(len(A)), A_eq=[np.r_[np.ones(self.m), 0]],
                           b_eq=[1.0], bounds=list(zip(lower, upper)) + [(0, 1)])
            if res.success and (not strict.any() or res.x[-1] > self.tolerance):
                reached.append(s)
                witnesses.append(res.x[:self.m])
        return np.asarray(reached, int), np.asarray(witnesses)

    def audit(self, center, radius):
        nominal = self.select(center)
        reachable, witnesses = self.reachable(center, radius)
        distances = np.linalg.norm(self.y[reachable] - self.y[nominal], axis=1)
        worst = int(np.argmax(distances))
        return dict(nominal=nominal, reachable=reachable, witnesses=witnesses,
                    robustness=float(distances[worst]), worst_index=int(reachable[worst]),
                    worst_weight=witnesses[worst], lp_calls=self.lp_calls)

    def stability_radius(self, center, epsilon=0.0):
        center = np.asarray(center, float)
        nominal = self.select(center)
        # Exclude cells with no full-simplex deterministic reachability.
        possible, _ = self.reachable(center, 1.0)
        best = float(np.max(np.maximum(center, 1 - center)))
        for q in possible:
            if np.linalg.norm(self.y[q] - self.y[nominal]) <= epsilon + 1e-12:
                continue
            # Min distance to the closure of q's actual optimality cell.
            cell = np.column_stack([self.y[q] - self.y, np.zeros(self.n)])
            upper = np.column_stack([np.eye(self.m), -np.ones(self.m)])
            lower = np.column_stack([-np.eye(self.m), -np.ones(self.m)])
            res = self._lp(np.r_[np.zeros(self.m), 1.0],
                           A_ub=np.vstack([cell, upper, lower]),
                           b_ub=np.r_[np.zeros(self.n), center, -center],
                           A_eq=[np.r_[np.ones(self.m), 0]], b_eq=[1],
                           bounds=[(0, 1)] * self.m + [(0, 1)])
            if res.success:
                best = min(best, float(res.x[-1]))
        return best

    def supported(self, references):
        flags = []
        for k in np.asarray(references):
            res = self._lp(np.zeros(self.m), A_ub=k - self.y,
                           b_ub=np.zeros(self.n), A_eq=[np.ones(self.m)], b_eq=[1],
                           bounds=[(0, 1)] * self.m)
            flags.append(bool(res.success))
        return np.asarray(flags)
