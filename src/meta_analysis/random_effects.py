"""DerSimonian-Laird and REML random-effects meta-analysis.

Implements two widely-used random-effects estimators:

* **DerSimonian-Laird (DL)**: Moment-based estimator of between-study
  variance τ².  Fast, closed-form, and the de-facto standard in most
  clinical meta-analysis software.

* **REML (Restricted Maximum Likelihood)**: Iterative estimator that
  has better statistical properties (less biased τ²) and is the default
  in metafor (R) and many modern reviews.

Supported effect measures: SMD, MD, OR (log), RR (log), HR (log), RD, COR.

References:
    DerSimonian R, Laird N. Meta-analysis in clinical trials.
        Controlled Clinical Trials 1986;7:177-188.
    Viechtbauer W. Conducting meta-analyses in R with the metafor package.
        J Stat Softw 2010;36(3):1-48.
    Cochrane Handbook for Systematic Reviews of Interventions v6.3.
"""

from __future__ import annotations

import logging
import math
from typing import List, Literal, Optional, Sequence

import numpy as np
import scipy.optimize as optimize
import scipy.stats as stats

from ..structured_schema.effect_record import EffectRecord
from ._result import MetaAnalysisResult

logger = logging.getLogger(__name__)

LOG_SCALE_MEASURES = {"OR", "RR", "HR"}
REML_MAX_ITER = 1000
REML_TOL = 1e-8


def _prepare_effect_values(
    effect_records: List[EffectRecord],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract and log-transform effect sizes and variances."""
    effects, variances = [], []
    for i, rec in enumerate(effect_records):
        if rec.effect_type in LOG_SCALE_MEASURES:
            if rec.effect_size <= 0:
                raise ValueError(
                    f"Study {i}: {rec.effect_type} effect must be > 0, got {rec.effect_size}"
                )
            es = math.log(rec.effect_size)
            ci_lo = math.log(max(rec.ci_lower, 1e-10))
            ci_hi = math.log(max(rec.ci_upper, 1e-10))
        else:
            es = rec.effect_size
            ci_lo, ci_hi = rec.ci_lower, rec.ci_upper

        z_alpha = stats.norm.ppf(1 - (1 - rec.ci_level) / 2)
        se = (ci_hi - ci_lo) / (2 * z_alpha)
        effects.append(es)
        variances.append(se ** 2)

    return np.array(effects, dtype=float), np.array(variances, dtype=float)


def _back_transform(
    pooled: float, ci_lo: float, ci_hi: float, effect_type: str
) -> tuple[float, float, float]:
    if effect_type in LOG_SCALE_MEASURES:
        return math.exp(pooled), math.exp(ci_lo), math.exp(ci_hi)
    return pooled, ci_lo, ci_hi


def _tau2_dl(theta: np.ndarray, variances: np.ndarray) -> float:
    """DerSimonian-Laird moment estimator of τ².

    τ²_DL = max(0, (Q - df) / c)

    where c = Σwᵢ - Σwᵢ²/Σwᵢ  (scaling factor).
    """
    w = 1.0 / variances
    pooled_fe = float(np.sum(w * theta) / np.sum(w))
    Q = float(np.sum(w * (theta - pooled_fe) ** 2))
    k = len(theta)
    c = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (Q - (k - 1)) / c)
    return tau2


def _tau2_reml(theta: np.ndarray, variances: np.ndarray) -> float:
    """REML estimator of τ² via profile likelihood optimisation.

    Maximises the restricted log-likelihood:

        l_R(τ²) = -½ Σ log(vᵢ + τ²)
                  - ½ log(Σ 1/(vᵢ+τ²))
                  - ½ Σ (θᵢ - θ̄)²/(vᵢ+τ²)

    Uses scipy.optimize.minimize_scalar on the negated function.
    """
    def neg_reml(tau2_val: float) -> float:
        if tau2_val < 0:
            return 1e10
        vi = variances + tau2_val
        wi = 1.0 / vi
        mu = np.sum(wi * theta) / np.sum(wi)
        ll = (
            -0.5 * np.sum(np.log(vi))
            - 0.5 * math.log(np.sum(wi))
            - 0.5 * np.sum((theta - mu) ** 2 / vi)
        )
        return -ll

    result = optimize.minimize_scalar(
        neg_reml,
        bounds=(0.0, 10.0 * float(np.var(theta))),
        method="bounded",
        options={"xatol": REML_TOL, "maxiter": REML_MAX_ITER},
    )
    return max(0.0, float(result.x))


class RandomEffectsModel:
    """DerSimonian-Laird / REML random-effects meta-analysis.

    Under the random-effects model each study estimates a different true
    effect θᵢ ~ N(μ, τ²), where μ is the grand mean and τ² is the
    between-study variance.

    The pooled estimate is the precision-weighted average under
    augmented variances vᵢ* = vᵢ + τ̂²:

        μ̂ = Σ(wᵢ* θᵢ) / Σwᵢ*,   wᵢ* = 1/(vᵢ + τ̂²)

    Args:
        method: ``"DL"`` (DerSimonian-Laird, default) or ``"REML"``.
        ci_level: Confidence level (default 0.95).
    """

    def __init__(
        self,
        method: Literal["DL", "REML"] = "DL",
        ci_level: float = 0.95,
    ) -> None:
        if method not in ("DL", "REML"):
            raise ValueError(f"method must be 'DL' or 'REML', got {method!r}")
        if not (0.0 < ci_level < 1.0):
            raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
        self.method = method
        self.ci_level = ci_level
        self._z_alpha = stats.norm.ppf(1 - (1 - ci_level) / 2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        effect_records: List[EffectRecord],
        study_labels: Optional[List[str]] = None,
    ) -> MetaAnalysisResult:
        """Fit random-effects model from EffectRecord objects.

        Args:
            effect_records: Studies to pool.  All should share the same
                ``effect_type``; a warning is logged otherwise.
            study_labels: Optional per-study labels.

        Returns:
            MetaAnalysisResult.

        Raises:
            ValueError: Fewer than 2 studies.
        """
        if len(effect_records) < 2:
            raise ValueError(
                f"At least 2 studies required, got {len(effect_records)}."
            )

        effect_types = {r.effect_type for r in effect_records}
        if len(effect_types) > 1:
            logger.warning("Mixed effect types: %s", effect_types)
        primary_type = effect_records[0].effect_type

        theta, variances = _prepare_effect_values(effect_records)
        result = self._pool(theta, variances, primary_type, study_labels)

        # Attach per-study originals
        result.effect_sizes = [r.effect_size for r in effect_records]
        result.ci_lowers = [r.ci_lower for r in effect_records]
        result.ci_uppers = [r.ci_upper for r in effect_records]
        result.total_n = sum(r.n_total for r in effect_records)
        return result

    def fit_from_arrays(
        self,
        effects: Sequence[float],
        variances: Sequence[float],
        study_labels: Optional[List[str]] = None,
        effect_type: str = "SMD",
    ) -> MetaAnalysisResult:
        """Fit from raw arrays.

        Args:
            effects: Per-study effect sizes (analysis scale).
            variances: Per-study sampling variances.
            study_labels: Optional labels.
            effect_type: Effect measure code.

        Returns:
            MetaAnalysisResult.
        """
        if len(effects) < 2:
            raise ValueError(
                f"At least 2 studies required, got {len(effects)}."
            )
        theta = np.asarray(effects, dtype=float)
        v = np.asarray(variances, dtype=float)
        result = self._pool(theta, v, effect_type, study_labels)
        result.effect_sizes = list(effects)
        result.ci_lowers = [
            e - self._z_alpha * math.sqrt(var)
            for e, var in zip(effects, variances)
        ]
        result.ci_uppers = [
            e + self._z_alpha * math.sqrt(var)
            for e, var in zip(effects, variances)
        ]
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pool(
        self,
        theta: np.ndarray,
        variances: np.ndarray,
        effect_type: str,
        study_labels: Optional[List[str]],
    ) -> MetaAnalysisResult:
        """Core pooling logic (shared by fit and fit_from_arrays)."""
        # Step 1: Estimate τ²
        if self.method == "DL":
            tau2 = _tau2_dl(theta, variances)
        else:
            tau2 = _tau2_reml(theta, variances)

        tau = math.sqrt(tau2)

        # Step 2: Augmented weights
        vi_star = variances + tau2
        weights = 1.0 / vi_star

        # Step 3: Pooled estimate
        pooled = float(np.sum(weights * theta) / np.sum(weights))
        var_pooled = 1.0 / np.sum(weights)
        se_pooled = math.sqrt(var_pooled)

        ci_lo_a = pooled - self._z_alpha * se_pooled
        ci_hi_a = pooled + self._z_alpha * se_pooled

        # Step 4: Z-test
        z_value = pooled / se_pooled
        p_value = float(2 * stats.norm.sf(abs(z_value)))

        # Step 5: Back-transform
        pooled_bt, ci_lo_bt, ci_hi_bt = _back_transform(
            pooled, ci_lo_a, ci_hi_a, effect_type
        )

        # Step 6: Heterogeneity statistics (Q on original weights)
        w_fe = 1.0 / variances
        pooled_fe = float(np.sum(w_fe * theta) / np.sum(w_fe))
        q_stat = float(np.sum(w_fe * (theta - pooled_fe) ** 2))
        q_df = len(theta) - 1
        q_p = float(stats.chi2.sf(q_stat, q_df))
        i2 = max(0.0, (q_stat - q_df) / q_stat * 100) if q_stat > 0 else 0.0
        h2 = q_stat / q_df if q_df > 0 else 1.0

        return MetaAnalysisResult(
            pooled_effect=pooled_bt,
            ci_lower=ci_lo_bt,
            ci_upper=ci_hi_bt,
            z_value=z_value,
            p_value=p_value,
            weights=list(weights / weights.sum() * 100),
            effect_sizes=[],       # filled by caller
            ci_lowers=[],          # filled by caller
            ci_uppers=[],          # filled by caller
            effect_type=effect_type,
            model=f"random_{self.method.lower()}",
            study_labels=study_labels,
            tau2=tau2,
            tau=tau,
            q_stat=q_stat,
            q_df=q_df,
            q_p_value=q_p,
            i2=i2,
            h2=h2,
            n_studies=len(theta),
            total_n=0,
            variances=list(variances),
        )
