"""Explicit research variants; the v0.1.0 package remains unchanged."""
from dataclasses import replace
import numpy as np
from tafrknee import CallbackProblem, SolveResult
from experiments.metrics.exact_tabular_oracle import canonical_priorities


def meaningful_gain(config, threshold=.001):
    return replace(config, min_improvement=threshold, min_deterioration=threshold)


def awt_table(y, augmentation=.001):
    """Normalized augmented weighted Tchebycheff with deterministic ties.

    Requires supplied ideal=0, reference=1 when called through TAFR. This is
    an experimental preference-map callback, not a weighted-sum optimizer.
    """
    y = np.asarray(y, float)
    priorities = canonical_priorities(y)

    def solve(weights, warm_start):
        w = weights / weights.sum()
        score = np.max(y*w, axis=1) + augmentation * (y @ w)
        tied = np.flatnonzero(score <= score.min() + 1e-12)
        i = int(tied[np.argmin(priorities[tied])])
        return SolveResult(i, y[i], metadata={"scalarization": "AWT", "augmentation": augmentation})

    return CallbackProblem(n_objectives=y.shape[1], solver=solve)
