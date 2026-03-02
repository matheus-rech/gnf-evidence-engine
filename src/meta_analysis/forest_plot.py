"""Publication-quality forest plot generation.

Generates forest plots suitable for systematic reviews and meta-analyses.
Supports both fixed- and random-effects pooled estimates, custom study
labels, effect-type–aware axis labels, and optional heterogeneity
annotations.

Output: matplotlib Figure (caller handles saving/displaying).

Typography follows Cochrane visual guidelines where possible.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ._result import MetaAnalysisResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EFFECT_LABELS = {
    "OR": "Odds Ratio",
    "RR": "Risk Ratio",
    "HR": "Hazard Ratio",
    "SMD": "Standardised Mean Difference",
    "MD": "Mean Difference",
    "RD": "Risk Difference",
    "COR": "Correlation Coefficient",
}
_LOG_SCALE = {"OR", "RR", "HR"}
_DIAMOND_HEIGHT = 0.35   # half-height of pooled diamond
_SQUARE_SCALE = 120      # point-size scaling for weight markers
_CI_LINEWIDTH = 1.2
_GRID_ALPHA = 0.25


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _effect_axis_label(effect_type: str) -> str:
    return _EFFECT_LABELS.get(effect_type, effect_type)


def _null_value(effect_type: str) -> float:
    """Return the null-hypothesis value for the given effect type."""
    return 1.0 if effect_type in _LOG_SCALE else 0.0


def _safe_ci(
    effect: float,
    ci_lo: float,
    ci_hi: float,
    x_min: float,
    x_max: float,
) -> tuple[float, float, bool, bool]:
    """Clip CI bounds to plot range and return arrow flags."""
    arrow_lo = ci_lo < x_min
    arrow_hi = ci_hi > x_max
    return max(ci_lo, x_min), min(ci_hi, x_max), arrow_lo, arrow_hi


def _compute_x_range(
    effects: Sequence[float],
    ci_lowers: Sequence[float],
    ci_uppers: Sequence[float],
    pooled: float,
    pool_lo: float,
    pool_hi: float,
    effect_type: str,
    padding: float = 0.15,
) -> tuple[float, float]:
    """Determine x-axis range with symmetric padding."""
    all_vals = list(effects) + list(ci_lowers) + list(ci_uppers) + [pooled, pool_lo, pool_hi]
    x_min, x_max = min(all_vals), max(all_vals)

    if effect_type in _LOG_SCALE:
        # Work on log scale
        log_min = math.log(max(x_min, 1e-6))
        log_max = math.log(max(x_max, 1e-6))
        span = log_max - log_min
        x_min = math.exp(log_min - padding * span)
        x_max = math.exp(log_max + padding * span)
    else:
        span = x_max - x_min or 1.0
        x_min -= padding * span
        x_max += padding * span

    return x_min, x_max


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def forest_plot(
    result: MetaAnalysisResult,
    title: str = "Forest Plot",
    figsize: Optional[Tuple[float, float]] = None,
    show_weights: bool = True,
    show_heterogeneity: bool = True,
    color_study: str = "#1f77b4",
    color_pooled: str = "#d62728",
) -> plt.Figure:
    """Generate a publication-quality forest plot.

    Args:
        result: MetaAnalysisResult from FixedEffectsModel or RandomEffectsModel.
        title: Plot title.
        figsize: (width, height) in inches.  Auto-calculated if None.
        show_weights: If True, scale study squares by weight.
        show_heterogeneity: If True, annotate with I², τ², Q-test.
        color_study: Hex colour for individual study CIs/squares.
        color_pooled: Hex colour for pooled diamond.

    Returns:
        matplotlib.figure.Figure  (caller must call plt.show() or savefig()).
    """
    k = result.n_studies
    labels = result.study_labels or [f"Study {i + 1}" for i in range(k)]

    effects = result.effect_sizes
    ci_lowers = result.ci_lowers
    ci_uppers = result.ci_uppers
    weights = result.weights  # percentage weights

    pooled = result.pooled_effect
    pool_lo = result.ci_lower
    pool_hi = result.ci_upper

    # Auto figure height
    if figsize is None:
        height = max(5.0, k * 0.55 + 3.0)
        figsize = (10.0, height)

    fig, ax = plt.subplots(figsize=figsize)

    # X-axis range
    x_min, x_max = _compute_x_range(
        effects, ci_lowers, ci_uppers, pooled, pool_lo, pool_hi, result.effect_type
    )

    # Log scale for ratio measures
    if result.effect_type in _LOG_SCALE:
        ax.set_xscale("log")

    # Null line
    null_val = _null_value(result.effect_type)
    ax.axvline(x=null_val, color="black", linewidth=0.8, linestyle="--", zorder=1)

    # Y positions (top = study 0)
    y_positions = list(range(k, 0, -1))   # k, k-1, ..., 1
    y_pooled = 0

    # --------------- Study rows -----------------
    for i, (y, label, eff, lo, hi) in enumerate(
        zip(y_positions, labels, effects, ci_lowers, ci_uppers)
    ):
        lo_c, hi_c, arr_lo, arr_hi = _safe_ci(eff, lo, hi, x_min, x_max)

        # CI line
        ax.plot([lo_c, hi_c], [y, y], color=color_study,
                linewidth=_CI_LINEWIDTH, zorder=2)

        # Arrow heads if CI exceeds plot range
        if arr_lo:
            ax.annotate("", xy=(x_min, y), xytext=(x_min * 1.05, y),
                        arrowprops=dict(arrowstyle="<-", color=color_study, lw=_CI_LINEWIDTH))
        if arr_hi:
            ax.annotate("", xy=(x_max, y), xytext=(x_max * 0.95, y),
                        arrowprops=dict(arrowstyle="<-", color=color_study, lw=_CI_LINEWIDTH))

        # Weight-scaled square
        if show_weights and weights:
            marker_size = max(20, weights[i] / 100 * _SQUARE_SCALE)
        else:
            marker_size = 40
        ax.scatter([eff], [y], s=marker_size, color=color_study,
                   zorder=3, marker="s")

        # Label
        ax.text(
            x_min, y, f" {label}",
            va="center", ha="left", fontsize=8,
            transform=ax.get_yaxis_transform(),
        )

    # --------------- Separator line -----------------
    ax.axhline(y=0.5, color="gray", linewidth=0.5, linestyle="-")

    # --------------- Pooled diamond -----------------
    diamond_y = [
        y_pooled,
        y_pooled + _DIAMOND_HEIGHT,
        y_pooled,
        y_pooled - _DIAMOND_HEIGHT,
        y_pooled,
    ]
    diamond_x = [pool_lo, pooled, pool_hi, pooled, pool_lo]
    poly = mpatches.Polygon(
        list(zip(diamond_x, diamond_y)),
        closed=True,
        facecolor=color_pooled,
        edgecolor=color_pooled,
        zorder=4,
    )

    # --------------- Axes formatting -----------------
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-1, k + 1.5)
    ax.set_yticks([])
    ax.set_xlabel(_effect_axis_label(result.effect_type), fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.xaxis.grid(True, alpha=_GRID_ALPHA)
    ax.set_axisbelow(True)

    # Pooled label
    model_label = {
        "fixed": "Fixed Effect",
        "random_dl": "Random (DL)",
        "random_reml": "Random (REML)",
    }.get(result.model, result.model.capitalize())
    ci_str = f"{result.pooled_effect:.3f} [{result.ci_lower:.3f}, {result.ci_upper:.3f}]"
    ax.text(
        x_min, y_pooled,
        f" {model_label}: {ci_str}",
        va="center", ha="left", fontsize=8,
        fontweight="bold",
        transform=ax.get_yaxis_transform(),
    )

    # Heterogeneity annotation
    if show_heterogeneity:
        het_str = (
            f"I² = {result.i2:.1f}%,  "
            f"τ² = {result.tau2:.4f},  "
            f"Q({result.q_df}) = {result.q_stat:.2f}, p = {result.q_p_value:.3f}"
        )
        ax.text(
            0.5, -0.07, het_str,
            transform=ax.transAxes,
            ha="center", va="top", fontsize=8, color="dimgray",
        )

    ax.add_patch(poly)
    fig.tight_layout()
    return fig
