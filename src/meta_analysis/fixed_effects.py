"""Inverse-variance weighted fixed-effect meta-analysis.

Implements the standard fixed-effect (common-effect) model where all
studies are assumed to estimate the same true population effect. The
pooled estimate is the precision-weighted average of individual effects.

Supported effect measures: SMD, MD, OR (log), RR (log), HR (log), RD, COR.

References:
    Borenstein M, et al. Introduction to Meta-Analysis. Wiley, 2009.
    Cochrane Handbook for Systematic Reviews of Interventions v6.3.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Sequence

import numpy as np
import scipy.stats as stats

from ..structured_schema.effect_record import EffectRecord
from ._result import MetaAnalysisResult

logger = logging.getLogger(__name__)

# Effect measures that require log-transformation before pooling
LOG_SCALE_MEASURES = {"OR", "RR", "HR"}


def _prepare_effect_values(
    effect_records: List[EffectRecord],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract and transform effect sizes and variances.

    For ratio measures (OR, RR, HR), log-transforms effects and CI bounds
    to ensure symmetry before pooling.

    Args:
        effect_records: List of EffectRecord objects.

    Returns:
        Tuple of (effects array, variances array) on analysis scale.

    Raises:
        ValueError: If any record has non-positive ratio measure.
    """
    effects = []
    variances = []

    for i, rec in enumerate(effect_records):
        if rec.effect_type in LOG_SCALE_MEASURES:
            if rec.effect_size <= 0:
                raise ValueError(
                    f"Study {i}: {rec.effect_type} effect size must be > 0, "
                    f"got {rec.effect_size}"
                )
            es = math.log(rec.effect_size)
            # SE on log scale from CI
            if rec.se is not None:
                # re-derive from CI to be consistent with log scale
                pass
            # Derive SE from log CI
            ci_lo = math.log(max(rec.ci_lower, 1e-10))
            ci_hi = math.log(max(rec.ci_upper, 1e-10))
        else:
            es = rec.effect_size
            ci_lo = rec.ci_lower
            ci_hi = rec.ci_upper

        # SE from CI width
        z_alpha = stats.norm.ppf(1 - (1 - rec.ci_level) / 2)
        se = (ci_hi - ci_lo) / (2 * z_alpha)
        var = se ** 2

        effects.append(es)
        variances.append(var)

    return np.array(effects, dtype=float), np.array(variances, dtype=float)


def _back_transform(
    pooled: float,
    ci_lower: float,
    ci_upper: float,
    effect_type: str,
) -> tuple[float, float, float]:
    """Back-transform pooled estimates from log scale if needed.

    Args:
        pooled: Pooled effect on analysis scale.
        ci_lower: Lower CI on analysis scale.
        ci_upper: Upper CI on analysis scale.
        effect_type: Effect measure code.

    Returns:
        (pooled, ci_lower, ci_upper) on the original reporting scale.
    """
    if effect_type in LOG_SCALE_MEASURES:
        return math.exp(pooled), math.exp(ci_lower), math.exp(ci_upper)
    return pooled, ci_lower, ci_upper


class FixedEffectsModel:
    """Inverse-variance weighted fixed-effect meta-analysis model.

    The fixed-effect (or common-effect) model assumes all studies share
    one true underlying effect; between-study variance is zero.

    The pooled estimate is:

        θ̄ = Σ(wᵢ θᵢ) / Σwᵢ,   where wᵢ = 1 / vᵢ

    Args:
        ci_level: Confidence level for CIs (default 0.95).
    """

    def __init__(self, ci_level: float = 0.95) -> None:
        if not (0.0 < ci_level < 1.0):
            raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
        self.ci_level = ci_level
        self._z_alpha = stats.norm.ppf(1 - (1 - ci_level) / 2)

    def fit(
        self,
        effect_records: List[EffectRecord],
        study_labels: Optional[List[str]] = None,
    ) -> MetaAnalysisResult:
        """Fit the fixed-effect model to a list of effect records.

        Args:
            effect_records: Studies to pool. All must share the same
                ``effect_type``; a warning is logged if mixed.
            study_labels: Optional per-study labels for forest plots.

        Returns:
            MetaAnalysisResult with pooled estimate and diagnostics.

        Raises:
            ValueError: If fewer than 2 studies are provided.
        """
        if len(effect_records) < 2:
            raise ValueError("At least 2 studies are required for meta-analysis. Got 1.")

        # Warn on mixed effect types
        effect_types = {r.effect_type for r in effect_records}
        if len(effect_types) > 1:
            logger.warning(
                "Mixed effect types detected: %s. Ensure comparability.", effect_types
            )
        primary_type = effect_records[0].effect_type

        # Transform to analysis scale
        theta, variances = _prepare_effect_values(effect_records)

        # Inverse-variance weights
        weights = 1.0 / variances

        # Pooled effect
        pooled = float(np.sum(weights * theta) / np.sum(weights))

        # Variance of pooled estimate
        var_pooled = 1.0 / np.sum(weights)
        se_pooled = math.sqrt(var_pooled)

        # CI
        ci_lower_analysis = pooled - self._z_alpha * se_pooled
        ci_upper_analysis = pooled + self._z_alpha * se_pooled

        # Z-test
        z_value = pooled / se_pooled
        p_value = float(2 * stats.norm.sf(abs(z_value)))

        # Back-transform
        pooled_bt, ci_lo_bt, ci_hi_bt = _back_transform(
            pooled, ci_lower_analysis, ci_upper_analysis, primary_type
        )

        # Cochran's Q
        q_stat = float(np.sum(weights * (theta - pooled) ** 2))
        q_df = len(effect_records) - 1
        q_p = float(stats.chi2.sf(q_stat, q_df))

        # I²
        i2 = max(0.0, (q_stat - q_df) / q_stat * 100) if q_stat > 0 else 0.0

        # H²
        h2 = q_stat / q_df if q_df > 0 else 1.0

        # Total N
        total_n = sum(r.n_total for r in effect_records)

        return MetaAnalysisResult(
            pooled_effect=pooled_bt,
            ci_lower=ci_lo_bt,
            ci_upper=ci_hi_bt,
            z_value=z_value,
            p_value=p_value,
            weights=list(weights / weights.sum() * 100),  # percentage weights
            effect_sizes=[r.effect_size for r in effect_records],
            ci_lowers=[r.ci_lower for r in effect_records],
            ci_uppers=[r.ci_upper for r in effect_records],
            effect_type=primary_type,
            model="fixed",
            study_labels=study_labels,
            tau2=0.0,
            tau=0.0,
            q_stat=q_stat,
            q_df=q_df,
            q_p_value=q_p,
            i2=i2,
            h2=h2,
            n_studies=len(effect_records),
            total_n=total_n,
            variances=list(variances),
        )

    def fit_from_arrays(
        self,
        effects: Sequence[float],
        variances: Sequence[float],
        study_labels: Optional[List[str]] = None,
        effect_type: str = "SMD",
    ) -> MetaAnalysisResult:
        """Fit directly from arrays of effects and variances.

        Useful for testing or when EffectRecord objects are not available.

        Args:
            effects: Per-study effect sizes (on analysis scale).
            variances: Per-study variance estimates.
            study_labels: Optional labels.
            effect_type: Effect measure code.

        Returns:
            MetaAnalysisResult.
        """
        if len(effects) < 2:
            raise ValueError(f"At least 2 studies are required for meta-analysis. Got {len(effects)}.")
        effects_arr = np.asarray(effects, dtype=float)
        variances_arr = np.asarray(variances, dtype=float)
        weights = 1.0 / variances_arr

        pooled = float(np.sum(weights * effects_arr) / np.sum(weights))
        var_pooled = 1.0 / np.sum(weights)
        se_pooled = math.sqrt(var_pooled)

        ci_lower = pooled - self._z_alpha * se_pooled
        ci_upper = pooled + self._z_alpha * se_pooled
        z_value = pooled / se_pooled
        p_value = float(2 * stats.norm.sf(abs(z_value)))

        q_stat = float(np.sum(weights * (effects_arr - pooled) ** 2))
        q_df = len(effects) - 1
        q_p = float(stats.chi2.sf(q_stat, q_df))
        i2 = max(0.0, (q_stat - q_df) / q_stat * 100) if q_stat > 0 else 0.0
        h2 = q_stat / q_df if q_df > 0 else 1.0

        return MetaAnalysisResult(
            pooled_effect=pooled,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            z_value=z_value,
            p_value=p_value,
            weights=list(weights / weights.sum() * 100),
            effect_sizes=list(effects),
            ci_lowers=[e - self._z_alpha * math.sqrt(v) for e, v in zip(effects, variances)],
            ci_uppers=[e + self._z_alpha * math.sqrt(v) for e, v in zip(effects, variances)],
            effect_type=effect_type,
            model="fixed",
            study_labels=study_labels,
            tau2=0.0,
            tau=0.0,
            q_stat=q_stat,
            q_df=q_df,
            q_p_value=q_p,
            i2=i2,
            h2=h2,
            n_studies=len(effects),
            total_n=0,
            variances=list(variances),
        )
