"""Funnel plot with Egger's test and trim-and-fill method.

Provides:
    * Standard funnel plot (effect size vs. standard error)
    * Contour-enhanced funnel plot with significance regions
    * Egger's linear regression test for funnel plot asymmetry
    * Trim-and-fill (L0 estimator) for publication bias correction

References:
    Egger M, et al. Bias in meta-analysis detected by a simple,
        graphical test. BMJ 1997;315:629-634.
    Duval S, Tweedie R. Trim and fill: a simple funnel-plot-based
        method of testing and adjusting for publication bias.
        Biometrics 2000;56:455-463.
    Peters JL, et al. Contour-enhanced meta-analysis funnel plots
        help distinguish publication bias from other causes of
        asymmetry. J Clin Epidemiol 2008;61:991-996.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from ._result import MetaAnalysisResult


@dataclass
class EggerResult:
    """Results of Egger's regression test."""
    intercept: float
    intercept_se: float
    intercept_t: float
    intercept_p: float
    slope: float
    slope_se: float
    df: int
    interpretation: str


@dataclass
class TrimFillResult:
    """Results of trim-and-fill procedure."""
    n_trimmed: int
    adjusted_effect: float
    adjusted_ci_lower: float
    adjusted_ci_upper: float
    original_effect: float
    imputed_effects: List[float]
    imputed_ses: List[float]


def egger_test(effects: List[float], ses: List[float]) -> EggerResult:
    """Egger's linear regression test for funnel plot asymmetry.

    Regresses the standardised effect (effect/SE) on precision (1/SE).
    Asymmetry is indicated by a statistically significant intercept.

    Args:
        effects: Per-study effect sizes (analysis scale).
        ses: Per-study standard errors.

    Returns:
        EggerResult dataclass.

    Raises:
        ValueError: Fewer than 3 studies.
    """
    if len(effects) < 3:
        raise ValueError("Egger's test requires at least 3 studies.")

    effects_arr = np.asarray(effects, dtype=float)
    ses_arr = np.asarray(ses, dtype=float)

    precision = 1.0 / ses_arr
    std_effect = effects_arr / ses_arr

    # OLS: std_effect ~ intercept + slope * precision
    X = np.column_stack([np.ones(len(precision)), precision])
    try:
        beta, residuals, rank, sv = np.linalg.lstsq(X, std_effect, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Egger regression failed.") from exc

    intercept, slope = float(beta[0]), float(beta[1])
    n = len(effects)
    df = n - 2
    y_hat = X @ beta
    sse = float(np.sum((std_effect - y_hat) ** 2))
    mse = sse / df if df > 0 else 0.0
    cov_matrix = mse * np.linalg.inv(X.T @ X)
    se_intercept = math.sqrt(max(cov_matrix[0, 0], 0.0))
    se_slope = math.sqrt(max(cov_matrix[1, 1], 0.0))

    t_intercept = intercept / se_intercept if se_intercept > 0 else 0.0
    p_intercept = float(2 * stats.t.sf(abs(t_intercept), df=df))

    interpretation = (
        "Significant asymmetry detected (possible publication bias)."
        if p_intercept < 0.05
        else "No significant asymmetry detected."
    )

    return EggerResult(
        intercept=intercept,
        intercept_se=se_intercept,
        intercept_t=t_intercept,
        intercept_p=p_intercept,
        slope=slope,
        slope_se=se_slope,
        df=df,
        interpretation=interpretation,
    )


def trim_and_fill(
    effects: List[float],
    ses: List[float],
    tau2: float = 0.0,
    side: str = "left",
) -> TrimFillResult:
    """Duval & Tweedie L0 trim-and-fill estimator.

    Iteratively trims asymmetric studies from the specified side,
    recomputes the pooled estimate, then imputes mirror studies to
    restore symmetry.

    Args:
        effects: Per-study effect sizes (analysis scale).
        ses: Per-study standard errors.
        tau2: Between-study variance (use 0.0 for fixed-effect).
        side: Side to trim — ``"left"`` (default) or ``"right"``.

    Returns:
        TrimFillResult.
    """
    n = len(effects)
    effects_arr = np.asarray(effects, dtype=float)
    ses_arr = np.asarray(ses, dtype=float)

    def pooled_mean(eff: np.ndarray, se: np.ndarray) -> float:
        w = 1.0 / (se ** 2 + tau2)
        return float(np.sum(w * eff) / np.sum(w))

    mu = pooled_mean(effects_arr, ses_arr)

    max_iter = 50
    k0_prev = -1
    for _ in range(max_iter):
        centered = effects_arr - mu
        ranks = np.argsort(np.argsort(np.abs(centered))) + 1  # rank 1 = smallest

        if side == "left":
            signs = np.sign(centered)
            T = np.sum(ranks[signs < 0])
        else:
            signs = np.sign(centered)
            T = np.sum(ranks[signs > 0])

        k = len(effects_arr)
        # L0 estimator
        k0 = int(round(
            (4 * T - k * (k + 1)) / (2 * k - 1)
        ))
        k0 = max(0, k0)

        if k0 == k0_prev:
            break
        k0_prev = k0

        # Trim k0 most extreme studies on specified side
        if k0 > 0:
            if side == "left":
                sorted_idx = np.argsort(effects_arr)
            else:
                sorted_idx = np.argsort(effects_arr)[::-1]
            trimmed_idx = sorted_idx[:k0]
            mask = np.ones(k, dtype=bool)
            mask[trimmed_idx] = False
            mu = pooled_mean(effects_arr[mask], ses_arr[mask])
        else:
            mu = pooled_mean(effects_arr, ses_arr)

    # Impute mirror studies
    imputed_effects: List[float] = []
    imputed_ses: List[float] = []
    if k0 > 0:
        if side == "left":
            sorted_idx = np.argsort(effects_arr)
        else:
            sorted_idx = np.argsort(effects_arr)[::-1]
        for idx in sorted_idx[:k0]:
            mirror_eff = 2 * mu - effects_arr[idx]
            imputed_effects.append(float(mirror_eff))
            imputed_ses.append(float(ses_arr[idx]))

    # Adjusted pooled estimate
    all_eff = np.concatenate([effects_arr, np.asarray(imputed_effects)])
    all_ses = np.concatenate([ses_arr, np.asarray(imputed_ses)])
    if len(all_eff) > 0:
        adj_mu = pooled_mean(all_eff, all_ses)
        w_adj = 1.0 / (all_ses ** 2 + tau2)
        se_adj = math.sqrt(1.0 / np.sum(w_adj))
    else:
        adj_mu = mu
        se_adj = 0.0

    z = stats.norm.ppf(0.975)
    return TrimFillResult(
        n_trimmed=k0,
        adjusted_effect=adj_mu,
        adjusted_ci_lower=adj_mu - z * se_adj,
        adjusted_ci_upper=adj_mu + z * se_adj,
        original_effect=pooled_mean(effects_arr, ses_arr),
        imputed_effects=imputed_effects,
        imputed_ses=imputed_ses,
    )


def funnel_plot(
    result: MetaAnalysisResult,
    title: str = "Funnel Plot",
    contour_enhanced: bool = False,
    show_egger: bool = True,
    show_trim_fill: bool = False,
    figsize: Tuple[float, float] = (7.0, 6.0),
    color_study: str = "#1f77b4",
) -> plt.Figure:
    """Generate a funnel plot with optional publication-bias diagnostics.

    Args:
        result: MetaAnalysisResult.
        title: Plot title.
        contour_enhanced: If True, shade significance contours
            (p < 0.05, p < 0.01, p < 0.001).
        show_egger: If True, annotate with Egger's test result.
        show_trim_fill: If True, show trim-and-fill imputed studies.
        figsize: Figure dimensions.
        color_study: Study marker colour.

    Returns:
        matplotlib.figure.Figure.
    """
    effects = np.asarray(result.effect_sizes, dtype=float)
    ci_lo = np.asarray(result.ci_lowers, dtype=float)
    ci_hi = np.asarray(result.ci_uppers, dtype=float)
    ses = (ci_hi - ci_lo) / (2 * stats.norm.ppf(0.975))

    pooled = result.pooled_effect
    max_se = float(np.max(ses)) * 1.1

    fig, ax = plt.subplots(figsize=figsize)

    # --- Contour shading ---
    if contour_enhanced:
        se_range = np.linspace(0, max_se, 200)
        for p_thresh, alpha_shade, label in [
            (0.001, 0.35, "p < 0.001"),
            (0.01, 0.20, "p < 0.01"),
            (0.05, 0.10, "p < 0.05"),
        ]:
            z = stats.norm.ppf(1 - p_thresh / 2)
            lo_contour = pooled - z * se_range
            hi_contour = pooled + z * se_range
            ax.fill_betweenx(
                se_range, lo_contour, hi_contour,
                alpha=alpha_shade, color="lightblue", label=label
            )

    # --- Study points ---
    ax.scatter(effects, ses, color=color_study, s=40, zorder=3, label="Studies")

    # --- Pooled vertical line ---
    ax.axvline(x=pooled, color="black", linewidth=0.8, linestyle="-", zorder=2)

    # --- Trim-and-fill imputed points ---
    if show_trim_fill and len(effects) >= 3:
        tf = trim_and_fill(list(effects), list(ses), tau2=result.tau2)
        if tf.imputed_effects:
            ax.scatter(
                tf.imputed_effects, tf.imputed_ses,
                color="gray", s=40, marker="o", zorder=3,
                facecolors="none", label="Imputed (trim-fill)"
            )

    # --- Egger annotation ---
    if show_egger and len(effects) >= 3:
        try:
            eg = egger_test(list(effects), list(ses))
            egger_str = (
                f"Egger: intercept = {eg.intercept:.3f} "
                f"(SE={eg.intercept_se:.3f}, p={eg.intercept_p:.3f})"
            )
            ax.text(
                0.02, 0.02, egger_str,
                transform=ax.transAxes, fontsize=8, color="dimgray"
            )
        except (ValueError, RuntimeError):
            pass

    # --- Axes ---
    ax.invert_yaxis()  # SE=0 at top
    ax.set_xlabel(result.effect_type, fontsize=10)
    ax.set_ylabel("Standard Error", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.xaxis.grid(True, alpha=0.25)
    if contour_enhanced or show_trim_fill:
        ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    return fig
