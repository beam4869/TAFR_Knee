# SPDX-License-Identifier: EPL-2.0
"""Full-decision port of the pinned PMOP CalObj implementations.

Derived from GYResearch/PMOPs-Benchmark, commit 4fb56cea31bae48f7ef08bc3180c3be150eb8f70.
Upstream notices and EPL-2.0 license: external/pmops. Preserve source anomalies;
this module must not be represented as a corrected paper-definition benchmark.
"""
import numpy as np

from experiments.benchmarks.pmop_suite import front


def objectives(number, decisions, m, linkage=False):
    x = np.atleast_2d(np.asarray(decisions, float))
    if number not in range(1, 15) or m < 3 or x.shape[1] < m:
        raise ValueError("Expected PMOP1-14, m >= 3 and at least m decision variables")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite decision vector")
    n, d = x.shape
    tail = x[:, m - 1:].copy()
    length = tail.shape[1]
    if linkage:
        i = np.arange(1, length + 1)
        factor = (1 + np.cos(.5 * np.pi * i / length)
                  if number in (2, 8, 10, 11) else 1 + i / length)
        tail = factor * tail - 10 * x[:, [0]]
        if number == 6:
            # The source writes temp2 but reads the separately zero-initialized temp.
            tail[:] = 0
    y = front(number, x[:, :m - 1])
    # PF includes g3's unit offset in PMOP3 and odd PMOP10 objectives.
    if number == 3:
        y = y / 2
    if number == 10:
        y[:, ::2] /= 2
    if number == 1:
        g = np.zeros(n)  # Explicit upstream g=0 assignment.
    elif number in (9, 13):
        g = np.max(np.abs(tail), axis=1)
    elif number == 2:
        g = np.sum(tail ** 2, axis=1)
    elif number in (3, 10):
        g = 1 + 10 * length + np.sum(tail ** 2 - 10 * np.cos(4 * np.pi * tail), axis=1)
    elif number == 4:
        z = tail - .5
        g = 100 * (length + np.sum(z ** 2 - np.cos(20 * np.pi * z), axis=1))
    elif number == 5:
        g = np.sum(100 * (tail[:, :-1] ** 2 - tail[:, 1:]) ** 2 +
                   (tail[:, :-1] - 1) ** 2, axis=1)
    elif number in (6, 12, 14):
        g = np.sum(tail ** 2 - 10 * np.cos(2 * np.pi * tail) + 10, axis=1)
    elif number == 7:
        # Upstream broadcasts each scalar across D columns and uses +prod.
        g = d * np.sum(tail ** 2 / 4000, axis=1) + \
            np.prod(np.cos(tail / np.sqrt(np.arange(1, length + 1))), axis=1) + 1
    elif number == 8:
        g = ackley(tail)
    elif number == 11:
        g = np.sum(tail ** 2, axis=1)
    if number == 10:
        y[:, ::2] *= 1 + g[:, None]
        # Upstream assigns the Griewank expression to g, leaving g7=0.
    elif number == 11:
        y[:, ::2] *= 1 + g[:, None]
        y[:, 1::2] *= 1 + np.max(np.abs(tail), axis=1)[:, None]
    elif number in (12, 14):
        y[:, ::2] *= 1 + g[:, None]
        y[:, 1::2] *= 1 + ackley(tail)[:, None]
    else:
        y *= 1 + g[:, None]
    return y


def ackley(tail):
    return (-20 * np.exp(-.2 * np.sqrt(np.mean(tail ** 2, axis=1))) -
            np.exp(np.mean(np.cos(2 * np.pi * tail), axis=1)) + 20 + np.e)


def distance_minimizer(number, count):
    """Only for parity diagnostics; the direct optimizer never fixes these values."""
    return np.full(count, .5 if number == 4 else 1. if number == 5 else 0.)
