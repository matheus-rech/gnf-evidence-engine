"""Shared result dataclass and utilities for meta-analysis modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class MetaAnalysisResult:
    """Result from a meta-analysis model fit.

    Attributes:
        pooled_effect: Pooled effect size estimate.
        ci_lower: Lower bound of pooled 95% CI.
        ci_upper: Upper bound of pooled 95% CI.
        z_value: Z-statistic for the pooled effect.
        p_value: Two-tailed p-value.
        weights: Per-study weights (list, same order as input).
        effect_sizes: Input effect sizes.
        ci_lowers: Input lower CIs.
        ci_uppers: Input upper CIs.
        study_labels: Optional study labels.
        effect_type: Effect measure type (SMD, OR, etc.).
        model: Model name ("fixed" or "random").
        tau2: Between-study variance (random effects only).
        tau: Between-study SD (random effects only).
        prediction_interval_lower: Lower prediction interval (random effects).
        prediction_interval_upper: Upper prediction interval (random effects).
        q_stat: Cochran's Q statistic.
        q_df: Degrees of freedom for Q.
        q_p_value: p-value for Q test.
        i2: I² heterogeneity statistic (0–100).
        h2: H² statistic.
        n_studies: Number of included studies.
        total_n: Total participant count.
    """

    pooled_effect: float
    ci_lower: float
    ci_upper: float
    z_value: float
    p_value: float
    weights: List[float]
    effect_sizes: List[float]
    ci_lowers: List[float]
    ci_uppers: List[float]
    effect_type: str = "SMD"
    model: str = "fixed"
    study_labels: Optional[List[str]] = None
    tau2: Optional[float] = None
    tau: Optional[float] = None
    prediction_interval_lower: Optional[float] = None
    prediction_interval_upper: Optional[float] = None
    q_stat: Optional[float] = None
    q_df: Optional[int] = None
    q_p_value: Optional[float] = None
    i2: Optional[float] = None
    h2: Optional[float] = None
    n_studies: int = 0
    total_n: int = 0
    variances: Optional[List[float]] = field(default=None)

    @property
    def is_significant(self) -> bool:
        """Whether the pooled effect is statistically significant at α=0.05."""
        return self.p_value < 0.05

    @property
    def relative_weights(self) -> List[float]:
        """Weights expressed as percentages (sum to 100)."""
        total = sum(self.weights)
        if total == 0:
            return [0.0] * len(self.weights)
        return [w / total * 100 for w in self.weights]

    def summary(self) -> str:
        """Human-readable summary string.

        Returns:
            Multi-line summary of the meta-analysis result.
        """
        lines = [
            f"Meta-Analysis Result ({self.model.title()} Effects)",
            f"  Studies:       {self.n_studies}",
            f"  Total N:       {self.total_n}",
            f"  Pooled {self.effect_type:<5}: {self.pooled_effect:.3f} "
            f"(95% CI: {self.ci_lower:.3f}, {self.ci_upper:.3f})",
            f"  Z = {self.z_value:.3f}, p = {self.p_value:.4f}",
        ]
        if self.tau2 is not None:
            lines += [
                f"  τ² = {self.tau2:.4f}, τ = {self.tau:.4f}",
            ]
            if self.prediction_interval_lower is not None:
                lines.append(
                    f"  Prediction interval: ({self.prediction_interval_lower:.3f}, "
                    f"{self.prediction_interval_upper:.3f})"
                )
        if self.i2 is not None:
            lines.append(f"  I² = {self.i2:.1f}%")
        return "\n".join(lines)
