"""Heterogeneity statistics for meta-analysis.

Provides:
    * Cochran's Q statistic and p-value
    * I² (proportion of variability due to heterogeneity)
    * H² (ratio of total to sampling variability)
    * τ² and τ with multiple estimators (DL, REML, PM, HS)
    * Prediction interval for the true effect in a new study
    * 95 % CI for I² via non-central chi-squared (Higgins & Thompson)

References:
    Higgins JPT, Thompson SG. Quantifying heterogeneity in a
        meta-analysis. Stat Med 2002;21:1539-1558.
    Viechtbauer W. Conducting meta-analyses in R with metafor.
        J Stat Softw 2010;36(3):1-48.
    Cochrane Handbook, version 6.3, Chapter 10.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np
import scipy.optimize as optimize
import scipy.stats as stats


@dataclass
class HeterogeneityResult:
    """Container for heterogeneity statistics."""

    # Cochran's Q
    q_stat: float
    q_df: int
    q_p_value: float

    # I² and H²
    i2: float          # percentage, 0–100
    i2_ci_lower: float # 95 % CI lower
    i2_ci_upper: float # 95 % CI upper
    h2: float

    # Between-study variance
    tau2: float
    tau2_se: Optional[float]
    tau: float
    estimator: str

    # Prediction interval (on reporting scale)
    prediction_lower: float
    prediction_upper: float

    # Metadata
    n_studies: int
    method_label: str = field(default="")

    def __post_init__(self) -> None:
        self.method_label = f"tau2 estimator: {self.estimator}"

    @property
    def heterogeneity_level(self) -> str:
        """Rough Higgins classification: low / moderate / substantial / considerable."""
        if self.i2 < 25:
            return "low"
        elif self.i2 < 50:
            return "moderate"
        elif self.i2 < 75:
            return "substantial"
        return "considerable"


# ---------------------------------------------------------------------------
# Internal estimators
# ---------------------------------------------------------------------------

def _q_statistic(
    theta: np.ndarray, variances: np.ndarray
) -> tuple[float, int, float]:
    """Compute Cochran's Q, df, and p-value."""
    w = 1.0 / variances
    mu_fe = float(np.sum(w * theta) / np.sum(w))
    Q = float(np.sum(w * (theta - mu_fe) ** 2))
    k = len(theta)
    df = k - 1
    p_val = float(stats.chi2.sf(Q, df))
    return Q, df, p_val


def _tau2_dl(theta: np.ndarray, variances: np.ndarray) -> tuple[float, None]:
    """DerSimonian-Laird moment estimator."""
    w = 1.0 / variances
    mu_fe = float(np.sum(w * theta) / np.sum(w))
    Q = float(np.sum(w * (theta - mu_fe) ** 2))
    k = len(theta)
    c = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (Q - (k - 1)) / c)
    return tau2, None


def _tau2_reml(theta: np.ndarray, variances: np.ndarray) -> tuple[float, None]:
    """REML estimator via profile log-likelihood."""
    def neg_ll(t2: float) -> float:
        if t2 < 0:
            return 1e10
        vi = variances + t2
        wi = 1.0 / vi
        mu = np.sum(wi * theta) / np.sum(wi)
        return 0.5 * np.sum(np.log(vi)) + 0.5 * math.log(np.sum(wi)) + 0.5 * np.sum((theta - mu) ** 2 / vi)

    res = optimize.minimize_scalar(
        neg_ll, bounds=(0.0, 10 * float(np.var(theta))), method="bounded"
    )
    return max(0.0, float(res.x)), None


def _tau2_pm(theta: np.ndarray, variances: np.ndarray) -> tuple[float, None]:
    """Paule-Mandel (PM) generalised Q-statistic estimator."""
    k = len(theta)
    def pm_eq(t2: float) -> float:
        if t2 < 0:
            return k - 1
        vi = variances + t2
        wi = 1.0 / vi
        mu = np.sum(wi * theta) / np.sum(wi)
        Q = float(np.sum(wi * (theta - mu) ** 2))
        return Q - (k - 1)

    try:
        tau2 = optimize.brentq(pm_eq, 0.0, 50 * float(np.var(theta)), xtol=1e-8)
    except ValueError:
        tau2 = 0.0
    return max(0.0, tau2), None


def _tau2_hs(theta: np.ndarray, variances: np.ndarray) -> tuple[float, None]:
    """Hunter-Schmidt estimator."""
    k = len(theta)
    mean_v = float(np.mean(variances))
    var_theta = float(np.var(theta, ddof=1))
    tau2 = max(0.0, var_theta - mean_v)
    return tau2, None


_TAU2_ESTIMATORS = {
    "DL": _tau2_dl,
    "REML": _tau2_reml,
    "PM": _tau2_pm,
    "HS": _tau2_hs,
}


def _i2_ci_higgins(
    Q: float, k: int, alpha: float = 0.05
) -> tuple[float, float]:
    """95 % CI for I² via non-central chi-squared (Higgins 2002)."""
    df = k - 1
    # Lower bound: use upper quantile of central chi2 distribution
    l_ncp = stats.ncx2.ppf(alpha / 2, df=df, nc=max(0.0, Q - df))
    u_ncp = stats.ncx2.ppf(1 - alpha / 2, df=df, nc=max(0.0, Q - df))
    i2_lo = max(0.0, (l_ncp - df) / l_ncp * 100) if l_ncp > 0 else 0.0
    i2_hi = min(100.0, (u_ncp - df) / u_ncp * 100) if u_ncp > 0 else 0.0
    return i2_lo, i2_hi


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_heterogeneity(
    effects: List[float],
    variances: List[float],
    pooled_effect: float,
    tau2_method: Literal["DL", "REML", "PM", "HS"] = "DL",
) -> HeterogeneityResult:
    """Compute full heterogeneity statistics for a set of studies.

    Args:
        effects: Per-study effect sizes (on the analysis scale, i.e.,
            log-transformed for OR/RR/HR).
        variances: Per-study sampling variances.
        pooled_effect: Pooled effect estimate (analysis scale).
        tau2_method: Between-study variance estimator.
            One of ``"DL"``, ``"REML"``, ``"PM"``, ``"HS"``.

    Returns:
        HeterogeneityResult dataclass.

    Raises:
        ValueError: Fewer than 2 studies or unknown estimator.
    """
    k = len(effects)
    if k < 2:
        raise ValueError(f"At least 2 studies required, got {k}.")
    if tau2_method not in _TAU2_ESTIMATORS:
        raise ValueError(
            f"Unknown tau2_method {tau2_method!r}. "
            f"Choose from {list(_TAU2_ESTIMATORS)}"
        )

    theta = np.asarray(effects, dtype=float)
    v = np.asarray(variances, dtype=float)

    # Cochran's Q
    Q, df, q_p = _q_statistic(theta, v)

    # τ²
    estimator_fn = _TAU2_ESTIMATORS[tau2_method]
    tau2, tau2_se = estimator_fn(theta, v)
    tau = math.sqrt(tau2)

    # I² and H²
    i2 = max(0.0, (Q - df) / Q * 100) if Q > 0 else 0.0
    h2 = Q / df if df > 0 else 1.0
    i2_lo, i2_hi = _i2_ci_higgins(Q, k)

    # Prediction interval (approximate, on analysis scale)
    # PI = μ̂ ± t_{k-2, 0.975} × sqrt(τ² + SE²_μ)
    if k >= 3:
        t_crit = stats.t.ppf(0.975, df=k - 2)
        w_re = 1.0 / (v + tau2)
        se_mu = math.sqrt(1.0 / np.sum(w_re))
        half_width = t_crit * math.sqrt(tau2 + se_mu ** 2)
        pred_lo = pooled_effect - half_width
        pred_hi = pooled_effect + half_width
    else:
        pred_lo = pred_hi = pooled_effect

    return HeterogeneityResult(
        q_stat=Q,
        q_df=df,
        q_p_value=q_p,
        i2=i2,
        i2_ci_lower=i2_lo,
        i2_ci_upper=i2_hi,
        h2=h2,
        tau2=tau2,
        tau2_se=tau2_se,
        tau=tau,
        estimator=tau2_method,
        prediction_lower=pred_lo,
        prediction_upper=pred_hi,
        n_studies=k,
    )
