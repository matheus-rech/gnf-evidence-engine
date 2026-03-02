"""Trial Sequential Analysis (TSA) engine.

TSA applies group sequential methods to cumulative meta-analysis.
As new studies are added, cumulative z-statistics are plotted against
alpha-spending boundaries adjusted for the total required information size.

Key concepts:
  - Information fraction (t): accrued N / required N (RIS)
  - Cumulative Z-statistic: derived from pooled effect at each step
  - Superiority boundary: Z > boundary → firm evidence of effect
  - Futility boundary: effect too small to matter → stop for futility
  - Conclusion: FIRM_EVIDENCE | INSUFFICIENT | FUTILE | HARM_SIGNAL

References:
    Wetterslev J, et al. Trial sequential analysis may establish when firm
        evidence is reached in cumulative meta-analysis. J Clin Epidemiol.
        2008;61:64-75.
    Thorlund K, et al. Improved power and sample size calculations for
        trial sequential analysis. Contemp Clin Trials. 2011;32:906-917.
    Higgins JPT, Thomas J, et al. Cochrane Handbook v6.3, 2022.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import numpy as np
import scipy.stats as stats

from .spending_functions import SpendingFunction, OBrienFleming, get_spending_function
from .information_size import RequiredInformationSize, RISResult

logger = logging.getLogger(__name__)

# Possible conclusions
ConclusionType = str  # "FIRM_EVIDENCE" | "INSUFFICIENT" | "FUTILE" | "HARM_SIGNAL"


@dataclass
class CumulativeStep:
    """One step in the cumulative meta-analysis sequence.

    Attributes:
        step_index: 0-based index (first study = 0).
        study_label: Label for the study added at this step.
        n_cumulative: Cumulative total participants at this step.
        pooled_effect: Pooled effect at this step.
        pooled_se: SE of pooled effect.
        z_stat: Cumulative z-statistic: pooled_effect / pooled_se.
        information_fraction: n_cumulative / RIS (capped at 1.0 for display).
        boundary_upper: Alpha-spending upper boundary z at this step.
        boundary_lower: Alpha-spending lower boundary z (negated).
        futility_upper: Futility inner boundary (beta-spending upper).
        futility_lower: Futility inner boundary (beta-spending lower).
        crossed_upper: Whether z_stat >= boundary_upper.
        crossed_lower: Whether z_stat <= boundary_lower (harm signal).
        crossed_futility: Whether |z_stat| < futility threshold.
    """

    step_index: int
    study_label: str
    n_cumulative: int
    pooled_effect: float
    pooled_se: float
    z_stat: float
    information_fraction: float
    boundary_upper: float
    boundary_lower: float
    futility_upper: float
    futility_lower: float
    crossed_upper: bool = False
    crossed_lower: bool = False
    crossed_futility: bool = False


@dataclass
class TSAResult:
    """Result from a full Trial Sequential Analysis run.

    Attributes:
        steps: Sequence of CumulativeStep objects.
        ris_result: Required Information Size calculation.
        final_z: Z-statistic at the final (or most recent) step.
        final_information_fraction: Information fraction at last step.
        conclusion: Overall TSA conclusion.
        conclusion_at_step: Step index where boundary was first crossed.
        alpha: Type-I error rate.
        beta: Type-II error rate.
        spending_function: Name of the spending function used.
        tsa_adjusted_ci_lower: TSA-adjusted lower CI for the pooled effect.
        tsa_adjusted_ci_upper: TSA-adjusted upper CI.
    """

    steps: List[CumulativeStep]
    ris_result: RISResult
    final_z: float
    final_information_fraction: float
    conclusion: ConclusionType
    conclusion_at_step: Optional[int]
    alpha: float
    beta: float
    spending_function: str
    tsa_adjusted_ci_lower: float
    tsa_adjusted_ci_upper: float

    @property
    def n_studies(self) -> int:
        """Number of studies in the analysis."""
        return len(self.steps)

    @property
    def accrued_n(self) -> int:
        """Total accrued participants at the final step."""
        return self.steps[-1].n_cumulative if self.steps else 0

    def summary(self) -> str:
        """Human-readable TSA summary.

        Returns:
            Multi-line summary string.
        """
        last = self.steps[-1] if self.steps else None
        lines = [
            "Trial Sequential Analysis",
            f"  Studies analysed:      {self.n_studies}",
            f"  Accrued N:             {self.accrued_n}",
            f"  Required N (RIS):      {self.ris_result.ris_adjusted:.0f}",
            f"  Information fraction:  {self.final_information_fraction:.3f}",
            f"  Spending function:     {self.spending_function}",
            f"  Final Z-statistic:     {self.final_z:.4f}",
        ]
        if last:
            lines += [
                f"  Upper boundary:       {last.boundary_upper:.4f}",
                f"  Lower boundary:       {last.boundary_lower:.4f}",
            ]
        lines.append(f"  CONCLUSION:           {self.conclusion}")
        return "\n".join(lines)


class TrialSequentialAnalysis:
    """Perform Trial Sequential Analysis on cumulative meta-analysis data.

    This class wraps a fixed- or random-effects cumulative meta-analysis
    with group sequential alpha-spending boundaries.

    Args:
        alpha: Two-sided type-I error rate (default 0.05).
        beta: Type-II error rate (default 0.20; power = 80%).
        spending_function: Spending function name or SpendingFunction instance.
            One of "obrien_fleming", "lan_demets", "pocock", "haybittle_peto".
        futility_spending: Whether to compute futility (beta-spending) boundaries.
        two_sided: Whether to use two-sided boundaries (default True).
    """

    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 0.20,
        spending_function: Union[str, SpendingFunction] = "obrien_fleming",
        futility_spending: bool = True,
        two_sided: bool = True,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.futility_spending = futility_spending
        self.two_sided = two_sided

        if isinstance(spending_function, str):
            self._spending = get_spending_function(spending_function, alpha=alpha)
        else:
            self._spending = spending_function

        # Futility spending (beta-spending) uses same function family
        if futility_spending:
            if isinstance(spending_function, str):
                self._futility_spending: Optional[SpendingFunction] = get_spending_function(
                    spending_function, alpha=beta
                )
            else:
                # Re-instantiate with beta as the budget
                self._futility_spending = type(spending_function)(alpha=beta)
        else:
            self._futility_spending = None

        self._ris_calc = RequiredInformationSize(alpha=alpha, beta=beta)

    def run(
        self,
        effects: Sequence[float],
        variances: Sequence[float],
        sample_sizes: Sequence[int],
        study_labels: Optional[List[str]] = None,
        ris_result: Optional[RISResult] = None,
        delta: float = 0.20,
        sigma: float = 1.0,
        i2: float = 0.0,
        outcome_type: str = "continuous",
        p_control: float = 0.30,
        rrr: float = 0.20,
    ) -> TSAResult:
        """Run TSA on a set of studies ordered chronologically.

        Args:
            effects: Per-study effect sizes (analysis scale), ordered by date.
            variances: Per-study within-study variances.
            sample_sizes: Per-study total sample sizes.
            study_labels: Optional labels.
            ris_result: Pre-computed RIS. If None, computed from delta/sigma.
            delta: MCID for continuous outcomes (SMD units).
            sigma: SD for continuous outcomes.
            i2: Expected I2 for RIS adjustment (0-1 scale).
            outcome_type: "continuous" or "binary".
            p_control: Control arm event rate (binary outcomes only).
            rrr: Relative risk reduction (binary outcomes only).

        Returns:
            TSAResult with all steps and conclusion.
        """
        if len(effects) < 2:
            raise ValueError("TSA requires at least 2 studies.")

        effects_arr = np.asarray(effects, dtype=float)
        variances_arr = np.asarray(variances, dtype=float)
        k = len(effects_arr)
        labels = study_labels or [f"Study {i+1}" for i in range(k)]

        # Compute RIS if not provided
        if ris_result is None:
            if outcome_type == "binary":
                ris_result = self._ris_calc.binary(
                    p_control=p_control,
                    relative_risk_reduction=rrr,
                    i2=i2,
                )
            else:
                ris_result = self._ris_calc.continuous(
                    delta=delta,
                    sigma=sigma,
                    i2=i2,
                )

        ris = ris_result.ris_adjusted

        # Build cumulative steps
        steps: List[CumulativeStep] = []
        alpha_spent = 0.0
        beta_spent = 0.0
        conclusion: ConclusionType = "INSUFFICIENT"
        conclusion_at_step: Optional[int] = None
        t_prev = 0.0

        cum_n = 0
        for i in range(k):
            cum_n += sample_sizes[i]
            t = min(1.0, cum_n / ris)

            # Pool effects up to step i (inverse-variance)
            theta_i = effects_arr[: i + 1]
            var_i = variances_arr[: i + 1]
            w_i = 1.0 / var_i
            pooled = float(np.sum(w_i * theta_i) / np.sum(w_i))
            var_pooled = 1.0 / np.sum(w_i)
            se_pooled = math.sqrt(var_pooled)
            z_stat = pooled / se_pooled if se_pooled > 0 else 0.0

            # Alpha boundary
            b_upper = self._alpha_boundary(t_prev, t)
            b_lower = -b_upper if self.two_sided else -float("inf")

            # Futility boundary
            fut_upper, fut_lower = self._futility_boundary(t_prev, t, t)

            # Check crossings
            crossed_upper = z_stat >= b_upper
            crossed_lower = z_stat <= b_lower
            crossed_futility = (
                b_lower < z_stat < fut_upper and t > 0.5
                if self.futility_spending
                else False
            )

            step = CumulativeStep(
                step_index=i,
                study_label=labels[i],
                n_cumulative=cum_n,
                pooled_effect=pooled,
                pooled_se=se_pooled,
                z_stat=z_stat,
                information_fraction=t,
                boundary_upper=b_upper,
                boundary_lower=b_lower,
                futility_upper=fut_upper,
                futility_lower=fut_lower,
                crossed_upper=crossed_upper,
                crossed_lower=crossed_lower,
                crossed_futility=crossed_futility,
            )
            steps.append(step)

            # First crossing -> lock conclusion
            if conclusion_at_step is None:
                if crossed_upper:
                    conclusion = "FIRM_EVIDENCE"
                    conclusion_at_step = i
                elif crossed_lower:
                    conclusion = "HARM_SIGNAL"
                    conclusion_at_step = i
                elif crossed_futility:
                    conclusion = "FUTILE"
                    conclusion_at_step = i

            t_prev = t

        # Final TSA-adjusted CI
        final_step = steps[-1]
        tsa_ci_lo, tsa_ci_hi = self._tsa_adjusted_ci(
            final_step.pooled_effect,
            final_step.pooled_se,
            final_step.boundary_upper,
        )

        return TSAResult(
            steps=steps,
            ris_result=ris_result,
            final_z=final_step.z_stat,
            final_information_fraction=final_step.information_fraction,
            conclusion=conclusion,
            conclusion_at_step=conclusion_at_step,
            alpha=self.alpha,
            beta=self.beta,
            spending_function=self._spending.name(),
            tsa_adjusted_ci_lower=tsa_ci_lo,
            tsa_adjusted_ci_upper=tsa_ci_hi,
        )

    def _alpha_boundary(self, t_prev: float, t_curr: float) -> float:
        """Compute upper alpha-spending boundary z at current look."""
        da = self._spending.incremental_alpha(t_prev, t_curr)
        if da <= 0 or da >= 1:
            return float("inf")
        return float(stats.norm.ppf(1 - da / 2))

    def _futility_boundary(
        self,
        t_prev: float,
        t_curr: float,
        t_overall: float,
    ) -> tuple[float, float]:
        """Compute inner (futility) boundaries."""
        if not self.futility_spending or self._futility_spending is None:
            return float("inf"), float("-inf")

        t = t_overall
        if t <= 0 or t >= 1:
            return float("inf"), float("-inf")

        db = self._futility_spending.incremental_alpha(t_prev, t)
        if db <= 0:
            return float("inf"), float("-inf")

        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_beta = stats.norm.ppf(1 - self.beta)
        fut = max(0.0, z_alpha * math.sqrt(t) - z_beta * math.sqrt(1 - t))
        return fut, -fut

    @staticmethod
    def _tsa_adjusted_ci(
        pooled: float,
        se: float,
        boundary_z: float,
    ) -> tuple[float, float]:
        """Compute TSA-adjusted confidence interval."""
        if math.isinf(boundary_z) or boundary_z > 10:
            boundary_z = 4.0
        ci_lower = pooled - boundary_z * se
        ci_upper = pooled + boundary_z * se
        return ci_lower, ci_upper
