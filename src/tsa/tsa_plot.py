"""TSA monitoring plot -- publication-quality visualization.

Generates the standard TSA monitoring plot:
  - X-axis: information fraction (0 to RIS)
  - Y-axis: cumulative z-statistic
  - O'Brien-Fleming (or other) alpha-spending boundaries
  - Futility inner boundary
  - Required information size vertical line
  - Cumulative Z-curve with study labels

Usage::

    from src.tsa.tsa_plot import TSAPlot

    tsa_result = tsa.run(...)
    plot = TSAPlot(tsa_result)
    fig = plot.render(title="TSA: Ketamine vs Placebo")
    fig.savefig("tsa_plot.png", dpi=300, bbox_inches="tight")
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from .trial_sequential import TSAResult, CumulativeStep

logger = logging.getLogger(__name__)

# Visual defaults
DEFAULTS = {
    "figsize": (10, 6),
    "dpi": 150,
    "z_curve_color": "#2E86AB",
    "boundary_color": "#E84855",
    "futility_color": "#F4A261",
    "ris_line_color": "#6B4C9A",
    "null_color": "#CCCCCC",
    "label_fontsize": 9,
    "title_fontsize": 13,
    "axis_fontsize": 11,
    "annotation_fontsize": 8,
    "linewidth_boundary": 1.8,
    "linewidth_z": 2.0,
}


class TSAPlot:
    """Create the TSA monitoring plot.

    Args:
        result: TSAResult from TrialSequentialAnalysis.run().
        show_ris_line: Whether to draw the vertical RIS (required information
            size) line.
        show_futility: Whether to draw the futility boundaries.
        show_study_labels: Whether to annotate each study on the Z-curve.
    """

    def __init__(
        self,
        result: TSAResult,
        show_ris_line: bool = True,
        show_futility: bool = True,
        show_study_labels: bool = True,
    ) -> None:
        self.result = result
        self.show_ris_line = show_ris_line
        self.show_futility = show_futility
        self.show_study_labels = show_study_labels

    def render(
        self,
        title: str = "Trial Sequential Analysis",
        custom_params: Optional[dict] = None,
    ) -> plt.Figure:
        """Render the TSA monitoring plot."""
        p = {**DEFAULTS, **(custom_params or {})}

        fig, ax = plt.subplots(figsize=p["figsize"], dpi=p["dpi"])

        # Generate smooth boundary curves
        t_grid = np.linspace(0.001, 1.0, 500)
        boundaries = self._compute_boundaries(t_grid)
        upper_b, lower_b, fut_upper, fut_lower = boundaries

        # Draw null line
        ax.axhline(0, color=p["null_color"], linewidth=1.0, linestyle=":", zorder=1)

        # Draw O'Brien-Fleming (or spending) boundaries
        ax.plot(t_grid, upper_b, color=p["boundary_color"],
                linewidth=p["linewidth_boundary"], linestyle="-",
                label=f"{self.result.spending_function} boundary", zorder=2)
        if self.result.alpha > 0 and len(self.result.steps) > 0 and \
                self.result.steps[0].boundary_lower < 0:
            ax.plot(t_grid, lower_b, color=p["boundary_color"],
                    linewidth=p["linewidth_boundary"], linestyle="-", zorder=2)

        # Futility boundaries
        if self.show_futility and fut_upper is not None:
            valid_mask = ~np.isinf(fut_upper)
            if np.any(valid_mask):
                ax.plot(
                    t_grid[valid_mask], fut_upper[valid_mask],
                    color=p["futility_color"],
                    linewidth=p["linewidth_boundary"] - 0.4,
                    linestyle="--",
                    label="Futility boundary",
                    zorder=2,
                )
                ax.plot(
                    t_grid[valid_mask], -fut_upper[valid_mask],
                    color=p["futility_color"],
                    linewidth=p["linewidth_boundary"] - 0.4,
                    linestyle="--",
                    zorder=2,
                )

        # Required information size vertical line
        if self.show_ris_line:
            ax.axvline(
                1.0,
                color=p["ris_line_color"],
                linewidth=1.4,
                linestyle="-.",
                label=f"RIS = {self.result.ris_result.ris_adjusted:.0f}",
                zorder=2,
            )

        # Cumulative Z-curve
        steps = self.result.steps
        t_vals = [s.information_fraction for s in steps]
        z_vals = [s.z_stat for s in steps]

        ax.plot(
            t_vals, z_vals,
            color=p["z_curve_color"],
            linewidth=p["linewidth_z"],
            marker="o",
            markersize=5,
            label="Cumulative Z",
            zorder=4,
        )

        # Study labels
        if self.show_study_labels:
            for step in steps:
                ax.annotate(
                    step.study_label,
                    xy=(step.information_fraction, step.z_stat),
                    xytext=(5, 4),
                    textcoords="offset points",
                    fontsize=p["annotation_fontsize"],
                    color=p["z_curve_color"],
                    clip_on=True,
                )

        # Mark conclusion point
        if self.result.conclusion_at_step is not None:
            cs = steps[self.result.conclusion_at_step]
            ax.scatter(
                [cs.information_fraction], [cs.z_stat],
                s=120, color="gold", edgecolors="black",
                zorder=5, marker="*",
            )

        # Annotation box
        conclusion_colors = {
            "FIRM_EVIDENCE": "#27AE60",
            "INSUFFICIENT": "#F39C12",
            "FUTILE": "#E74C3C",
            "HARM_SIGNAL": "#8E44AD",
        }
        conclusion_text = self.result.conclusion.replace("_", " ")
        c_color = conclusion_colors.get(self.result.conclusion, "#333333")
        props = dict(boxstyle="round,pad=0.4", facecolor=c_color, alpha=0.15)
        ax.text(
            0.02, 0.97,
            f"Conclusion: {conclusion_text}\n"
            f"t = {self.result.final_information_fraction:.3f}  "
            f"Z = {self.result.final_z:.3f}",
            transform=ax.transAxes,
            fontsize=p["annotation_fontsize"] + 1,
            va="top",
            bbox=props,
            color=c_color,
        )

        # Axes formatting
        ax.set_xlabel("Information Fraction (t = accrued N / RIS)", fontsize=p["axis_fontsize"])
        ax.set_ylabel("Cumulative Z-statistic", fontsize=p["axis_fontsize"])
        ax.set_title(title, fontsize=p["title_fontsize"], fontweight="bold")
        ax.set_xlim(0, min(1.05, max(t_vals) * 1.1))

        # Y-axis bounds: show boundaries + some padding
        max_b = max(upper_b[~np.isinf(upper_b)]) if not np.all(np.isinf(upper_b)) else 5.0
        y_lo = min(min(z_vals), -max_b) - 0.5
        y_hi = max(max(z_vals), max_b) + 0.5
        ax.set_ylim(y_lo, y_hi)

        ax.legend(fontsize=p["annotation_fontsize"] + 1, loc="lower right")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle=":")

        plt.tight_layout()
        return fig

    def _compute_boundaries(
        self,
        t_grid: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute boundary z-values over a fine information fraction grid."""
        from .spending_functions import get_spending_function

        try:
            fn = get_spending_function(
                self.result.spending_function.lower()
                .replace(" ", "_")
                .replace("'", "")
                .replace("-", "_"),
                alpha=self.result.alpha,
            )
        except ValueError:
            from .spending_functions import OBrienFleming
            fn = OBrienFleming(alpha=self.result.alpha)

        upper_b = np.full_like(t_grid, float("inf"))
        lower_b = np.full_like(t_grid, float("-inf"))
        fut_upper = np.full_like(t_grid, float("inf"))

        t_prev = 0.0
        z_alpha = stats.norm.ppf(1 - self.result.alpha / 2)
        z_beta = stats.norm.ppf(1 - self.result.beta)

        for j, t in enumerate(t_grid):
            da = fn.incremental_alpha(t_prev, t)
            if 0 < da < 1:
                z = stats.norm.ppf(1 - da / 2)
                upper_b[j] = z
                lower_b[j] = -z
            else:
                upper_b[j] = 8.0
                lower_b[j] = -8.0

            if 0 < t < 1:
                fut = max(
                    0.0,
                    z_alpha * math.sqrt(t) - z_beta * math.sqrt(1.0 - t),
                )
                fut_upper[j] = fut
            t_prev = t

        return upper_b, lower_b, fut_upper, -fut_upper
