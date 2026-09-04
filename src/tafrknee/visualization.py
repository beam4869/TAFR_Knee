"""Optional Matplotlib diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from .types import SelectionResult


def plot_objective_space(result: SelectionResult, *, normalized: bool = True) -> Any:
    """Plot audited outcomes and certified knees for two objectives."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - dependency-specific
        raise ImportError("install TAFR-Knee with the 'plot' extra") from error
    if result.ideal.size != 2:
        raise ValueError("plot_objective_space currently supports exactly two objectives")
    values = np.vstack(
        [audit.normalized_objectives if normalized else audit.objectives for audit in result.audits]
    )
    figure, axis = plt.subplots()
    axis.scatter(values[:, 0], values[:, 1], s=24, alpha=0.55, label="audited optima")
    if result.knees:
        knees = np.vstack(
            [knee.normalized_objectives if normalized else knee.objectives for knee in result.knees]
        )
        axis.scatter(knees[:, 0], knees[:, 1], marker="*", s=180, label="certified knee")
    axis.set_xlabel("normalized f1" if normalized else "f1")
    axis.set_ylabel("normalized f2" if normalized else "f2")
    axis.legend()
    figure.tight_layout()
    return figure


def plot_weight_profile(result: SelectionResult) -> Any:
    """Plot robustness and certification for a biobjective weight search."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - dependency-specific
        raise ImportError("install TAFR-Knee with the 'plot' extra") from error
    if result.ideal.size != 2:
        raise ValueError("plot_weight_profile currently supports exactly two objectives")
    ordered = sorted(result.audits, key=lambda audit: audit.weight[0])
    weights = np.array([audit.weight[0] for audit in ordered])
    robustness = np.array([audit.robustness for audit in ordered])
    certified = np.array([audit.certified for audit in ordered])
    figure, axis = plt.subplots()
    axis.plot(weights, robustness, marker="o", label=r"$R_\rho$")
    axis.scatter(weights[certified], robustness[certified], marker="*", s=140, label="certified")
    axis.set_xlabel("weight on f1")
    axis.set_ylabel("finite-radius displacement")
    axis.legend()
    figure.tight_layout()
    return figure
