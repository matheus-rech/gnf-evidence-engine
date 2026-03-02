"""Required Information Size (RIS) calculation for Trial Sequential Analysis.

The Required Information Size is the minimum cumulative sample size needed
to detect a clinically meaningful effect with specified power and type-I
error, while accounting for heterogeneity.

For binary outcomes (OR, RR, RD):
    RIS_bin = n_classical × D/(1 - τ²/σ²_between)

For continuous outcomes (SMD, MD):
    RIS_cont = n_classical × D/(1 - τ²/σ²_between)

where D is the diversity factor D = 1/(1 - I²/100) and the classical
sample size n_classical is computed from the standard power formula.

References:
    Wetterslev J, et al. Trial sequential analysis may establish when
        firm evidence is reached in cumulative meta-analysis.
        J Clin Epidemiol 2008;61:64-75.
    Thorlund K, et al. Can trial sequential monitoring boundaries reduce
        spurious inferences from meta-analyses?
        Int J Epidemiol 2009;38(1):276-286.
    Brok J, et al. Apparently conclusive meta-analyses may be
        inconclusive. J Clin Epidemiol 2009;62:64-75.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import scipy.stats as stats


@dataclass
class InformationSizeResult:
    """Required Information Size calculation output."""
    ris: float                   # Required Information Size (participants)
    n_classical: float           # Classical (unadjusted) sample size
    diversity: float             # D = 1/(1 - I²/100) — heterogeneity adjustment
    i2: float                    # Input I² used for diversity
    alpha: float                 # Type-I error used
    beta: float                  # Type-II error (1 - power)
    power: float                 # Target power
    outcome_type: str            # "binary" or "continuous"
    effect_measure: str          # e.g. "RR", "SMD"
    note: str = ""


def compute_ris(
    outcome_type: Literal["binary", "continuous"],
    effect_measure: str,
    i2: float = 0.0,
    alpha: float = 0.05,
    beta: float = 0.20,
    # Binary-specific
    p_control: Optional[float] = None,
    relative_risk_reduction: Optional[float] = None,
    risk_ratio: Optional[float] = None,
    # Continuous-specific
    smd: Optional[float] = None,
    # Common
    two_sided: bool = True,
) -> InformationSizeResult:
    """Compute the Required Information Size for a meta-analysis.

    For **binary** outcomes, the control-arm event rate (``p_control``) and
    either the ``relative_risk_reduction`` or the ``risk_ratio`` must be
    provided. The treatment-arm event rate is derived as
    ``p_treatment = p_control × RR``.

    For **continuous** outcomes, the standardised mean difference
    (``smd``) must be provided.

    Heterogeneity is corrected via the diversity factor
    D = 1 / (1 - I²/100), which inflates the classical RIS when
    heterogeneity is present.

    Args:
        outcome_type: ``"binary"`` or ``"continuous"``.
        effect_measure: Effect measure code (e.g. ``"RR"``, ``"SMD"``).
        i2: Estimated I² (0–100).  Defaults to 0 (no heterogeneity).
        alpha: Total type-I error.  Defaults to 0.05 (5%).
        beta: Type-II error.  Defaults to 0.20 (80% power).
        p_control: Control-arm event rate [binary only].
        relative_risk_reduction: RRR = 1 - RR [binary, alternative to risk_ratio].
        risk_ratio: RR directly [binary, alternative to RRR].
        smd: Standardised mean difference [continuous only].
        two_sided: If True, two-sided test (default).  One-sided halves α.

    Returns:
        InformationSizeResult dataclass.

    Raises:
        ValueError: Missing or invalid parameters.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not (0.0 < beta < 1.0):
        raise ValueError(f"beta must be in (0, 1), got {beta}")
    if not (0.0 <= i2 < 100.0):
        raise ValueError(f"i2 must be in [0, 100), got {i2}")

    power = 1.0 - beta
    alpha_use = alpha if two_sided else alpha / 2
    z_alpha = stats.norm.ppf(1 - alpha_use / 2)
    z_beta = stats.norm.ppf(power)

    if outcome_type == "binary":
        n_classical = _ris_binary(
            p_control, relative_risk_reduction, risk_ratio, z_alpha, z_beta
        )
    elif outcome_type == "continuous":
        n_classical = _ris_continuous(smd, z_alpha, z_beta)
    else:
        raise ValueError(
            f"outcome_type must be 'binary' or 'continuous', got {outcome_type!r}"
        )

    # Diversity adjustment
    if i2 >= 100.0:
        raise ValueError("I² cannot be 100 % (undefined diversity factor).")
    diversity = 1.0 / max(0.01, 1.0 - i2 / 100.0)
    ris = n_classical * diversity

    note = ""
    if diversity > 2.0:
        note = (
            f"High heterogeneity (I²={i2:.1f}%) inflates RIS by "
            f"{diversity:.1f}×. Interpret with caution."
        )

    return InformationSizeResult(
        ris=ris,
        n_classical=n_classical,
        diversity=diversity,
        i2=i2,
        alpha=alpha,
        beta=beta,
        power=power,
        outcome_type=outcome_type,
        effect_measure=effect_measure,
        note=note,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ris_binary(
    p_control: Optional[float],
    rrr: Optional[float],
    rr: Optional[float],
    z_alpha: float,
    z_beta: float,
) -> float:
    """Classical binary sample size (per arm) × 2."""
    if p_control is None:
        raise ValueError("p_control is required for binary outcomes.")
    if not (0.0 < p_control < 1.0):
        raise ValueError(f"p_control must be in (0, 1), got {p_control}")

    if rr is not None:
        risk_ratio = rr
    elif rrr is not None:
        risk_ratio = 1.0 - rrr
    else:
        raise ValueError(
            "Either risk_ratio or relative_risk_reduction must be provided for binary outcomes."
        )

    p_treatment = p_control * risk_ratio
    if not (0.0 < p_treatment < 1.0):
        raise ValueError(
            f"Derived p_treatment = {p_treatment:.4f} is outside (0, 1). "
            "Check p_control and risk_ratio values."
        )

    p_bar = (p_control + p_treatment) / 2.0
    numerator = (z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
                 + z_beta * math.sqrt(p_control * (1 - p_control)
                                       + p_treatment * (1 - p_treatment))) ** 2
    denominator = (p_control - p_treatment) ** 2
    n_per_arm = numerator / denominator
    return 2.0 * n_per_arm  # total N both arms


def _ris_continuous(
    smd: Optional[float],
    z_alpha: float,
    z_beta: float,
) -> float:
    """Classical continuous sample size (per arm) × 2."""
    if smd is None:
        raise ValueError("smd is required for continuous outcomes.")
    if smd == 0.0:
        raise ValueError("smd must be non-zero for a meaningful RIS.")

    n_per_arm = 2.0 * ((z_alpha + z_beta) / smd) ** 2
    return 2.0 * n_per_arm  # total N both arms


def minimum_ris_for_significance(
    outcome_type: Literal["binary", "continuous"],
    current_n: int,
    current_z: float,
    alpha: float = 0.05,
    i2: float = 0.0,
) -> float:
    """Compute the additional N needed to achieve significance.

    Uses the current z-statistic to back-calculate the required information
    size and return the shortfall.

    Args:
        outcome_type: ``"binary"`` or ``"continuous"``.
        current_n: Current cumulative sample size.
        current_z: Current z-statistic from the meta-analysis.
        alpha: Type-I error threshold.
        i2: Current I².

    Returns:
        Additional participants needed (0 if already significant).
    """
    z_thresh = stats.norm.ppf(1 - alpha / 2)
    if abs(current_z) >= z_thresh:
        return 0.0

    ratio = (z_thresh / current_z) ** 2
    divisor = max(0.10, 1.0 - i2 / 100.0)  # avoid division by near-zero
    return current_n * ratio / divisor
