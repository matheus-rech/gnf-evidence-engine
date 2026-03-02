"""Alpha-spending functions for Trial Sequential Analysis.

Alpha-spending functions determine how the total type-I error budget (α)
is allocated across interim analyses as information accumulates.
All functions return a monotone non-decreasing sequence of cumulative
spending values ρ(t) where t ∈ [0, 1] is the information fraction.

Implemented spending functions:
    * O'Brien-Fleming (OBF) — conservative early, liberal late
    * Pocock — approximately equal spending at each look
    * Kim-DeMets (power family)  ρ(t) = α · tᵞ
    * Hwang-Shih-DeCani (HSD)  exponential family

References:
    Lan KKG, DeMets DL. Discrete sequential boundaries for clinical
        trials. Biometrika 1983;70:659-663.
    Kim K, DeMets DL. Design and analysis of group sequential tests
        based on the type I error spending rate function.
        Biometrika 1987;74(1):149-154.
    O'Brien PC, Fleming TR. A multiple testing procedure for clinical
        trials. Biometrics 1979;35(3):549-556.
    Pocock SJ. Group sequential methods in the design and analysis of
        clinical trials. Biometrika 1977;64(2):191-199.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import scipy.stats as stats


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

SpendingFn = Callable[[float, float], float]
"""A spending function f(t, alpha) -> cumulative_spending in [0, alpha]."""


# ---------------------------------------------------------------------------
# O'Brien-Fleming
# ---------------------------------------------------------------------------

def obf_spending(t: float, alpha: float = 0.05) -> float:
    """O'Brien-Fleming alpha-spending function.

    Approximation by Lan & DeMets (1983):

        ρ_OBF(t) = 2[1 - Φ(z_{α/2} / √t)]

    where Φ is the standard normal CDF and z_{α/2} = Φ⁻¹(1 - α/2).

    Args:
        t: Information fraction in (0, 1].
        alpha: Total type-I error budget (default 0.05).

    Returns:
        Cumulative alpha spent at information fraction t.

    Raises:
        ValueError: If t ≤ 0 or t > 1 or alpha ≤ 0 or alpha ≥ 1.
    """
    _validate(t, alpha)
    z = stats.norm.ppf(1 - alpha / 2)
    return float(2 * (1 - stats.norm.cdf(z / math.sqrt(t))))


# ---------------------------------------------------------------------------
# Pocock
# ---------------------------------------------------------------------------

def pocock_spending(t: float, alpha: float = 0.05) -> float:
    """Pocock alpha-spending function.

    Approximation by Lan & DeMets (1983):

        ρ_Pocock(t) = α · ln(1 + (e - 1) · t)

    Args:
        t: Information fraction in (0, 1].
        alpha: Total type-I error budget.

    Returns:
        Cumulative alpha spent at t.
    """
    _validate(t, alpha)
    return float(alpha * math.log(1 + (math.e - 1) * t))


# ---------------------------------------------------------------------------
# Kim-DeMets power family
# ---------------------------------------------------------------------------

def kim_demets_spending(
    t: float, alpha: float = 0.05, gamma: float = 1.0
) -> float:
    """Kim-DeMets power-family alpha-spending function.

        ρ(t) = α · tᵞ

    Special cases:
        γ = 1  →  linear spending
        γ < 1  →  front-loaded (more spending early)
        γ > 1  →  back-loaded (more spending late)

    Args:
        t: Information fraction in (0, 1].
        alpha: Total type-I error budget.
        gamma: Power parameter γ > 0 (default 1.0 = linear).

    Returns:
        Cumulative alpha spent at t.

    Raises:
        ValueError: If gamma ≤ 0.
    """
    _validate(t, alpha)
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    return float(alpha * t ** gamma)


# ---------------------------------------------------------------------------
# Hwang-Shih-DeCani
# ---------------------------------------------------------------------------

def hsd_spending(
    t: float, alpha: float = 0.05, gamma: float = -4.0
) -> float:
    """Hwang-Shih-DeCani exponential alpha-spending function.

        ρ(t) = α · [1 - e^{-γt}] / [1 - e^{-γ}]  for γ ≠ 0
        ρ(t) = α · t                                 for γ = 0

    Negative γ produces conservative early spending (similar to OBF);
    positive γ produces liberal early spending (similar to Pocock).

    Args:
        t: Information fraction in (0, 1].
        alpha: Total type-I error budget.
        gamma: Shape parameter (default -4.0).

    Returns:
        Cumulative alpha spent at t.
    """
    _validate(t, alpha)
    if abs(gamma) < 1e-10:
        return float(alpha * t)
    return float(
        alpha * (1 - math.exp(-gamma * t)) / (1 - math.exp(-gamma))
    )


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _validate(t: float, alpha: float) -> None:
    if not (0 < t <= 1.0):
        raise ValueError(f"t must be in (0, 1], got {t}")
    if not (0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_SPENDING_REGISTRY: dict[str, SpendingFn] = {
    "OBF": obf_spending,
    "obrien_fleming": obf_spending,
    "Pocock": pocock_spending,
    "pocock": pocock_spending,
    "KD": lambda t, alpha=0.05: kim_demets_spending(t, alpha),
    "kim_demets": lambda t, alpha=0.05: kim_demets_spending(t, alpha),
    "HSD": lambda t, alpha=0.05: hsd_spending(t, alpha),
    "hsd": lambda t, alpha=0.05: hsd_spending(t, alpha),
}


def get_spending_function(key: str, alpha: float = 0.05) -> SpendingFn:
    """Retrieve a named spending function from the registry.

    Args:
        key: Name of the spending function.
            Supported: ``"OBF"`` / ``"obrien_fleming"``,
            ``"Pocock"`` / ``"pocock"``,
            ``"KD"`` / ``"kim_demets"``,
            ``"HSD"`` / ``"hsd"``.
        alpha: Default alpha budget (used to partially-apply the function).

    Returns:
        Callable ``f(t, alpha)`` → cumulative spending.

    Raises:
        KeyError: Unknown key.
    """
    if key not in _SPENDING_REGISTRY:
        raise KeyError(
            f"Unknown spending function {key!r}. "
            f"Available: {list(_SPENDING_REGISTRY.keys())}"
        )
    return _SPENDING_REGISTRY[key]
