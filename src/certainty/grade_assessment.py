"""GRADE certainty of evidence assessment.

Implements the Grading of Recommendations, Assessment, Development, and
Evaluation (GRADE) framework for rating certainty of evidence from
systematic reviews and meta-analyses.

Starting certainties:
  - RCT evidence -> HIGH
  - Observational (cohort, case-control) -> LOW

Downgrade domains (each can lower by 1 or 2 levels):
  - Risk of bias
  - Inconsistency (heterogeneity)
  - Indirectness
  - Imprecision (sample size / CI width)
  - Publication bias

Upgrade domains (observational only, each can raise by 1 level):
  - Large magnitude of effect
  - Dose-response gradient
  - All plausible confounders would increase effect

Final certainty: VERY LOW | LOW | MODERATE | HIGH

References:
    Guyatt GH, et al. GRADE: an emerging consensus on rating quality of
        evidence and strength of recommendations. BMJ. 2008;336:924.
    Schunemann HJ, et al. Cochrane Handbook ch. 14, 2022.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Certainty level ordering (lower index = lower certainty)
CERTAINTY_LEVELS = ["VERY LOW", "LOW", "MODERATE", "HIGH"]

# Mapping of design type to starting certainty index
DESIGN_START = {
    "RCT": 3,           # HIGH
    "cohort": 1,        # LOW
    "case-control": 1,  # LOW
    "cross-sectional": 1,
    "case-series": 0,   # VERY LOW
    "other": 1,
}

# Severity codes
NO_CONCERNS = "no_concerns"
SOME_CONCERNS = "some_concerns"  # downgrade by 1
SERIOUS = "serious"              # downgrade by 1
VERY_SERIOUS = "very_serious"    # downgrade by 2

SEVERITY_SCORES = {
    NO_CONCERNS: 0,
    SOME_CONCERNS: 0,
    SERIOUS: 1,
    VERY_SERIOUS: 2,
}


@dataclass
class GRADEDomain:
    """Assessment for one GRADE domain."""

    domain: str
    rating: str
    downgrade: int
    notes: Optional[str] = None


@dataclass
class GRADERating:
    """Complete GRADE certainty rating for one outcome."""

    outcome: str
    study_design: str
    n_studies: int
    total_n: int
    starting_certainty: str
    final_certainty: str
    final_certainty_index: int
    total_downgrade: int
    total_upgrade: int
    domains: List[GRADEDomain] = field(default_factory=list)
    pooled_effect: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    notes: Optional[str] = None

    def summary(self) -> str:
        lines = [
            f"GRADE Assessment -- {self.outcome}",
            f"  Study design:  {self.study_design} ({self.n_studies} studies, N={self.total_n})",
            f"  Starting:      {self.starting_certainty}",
            f"  Downgraded by: {self.total_downgrade} level(s)",
            f"  Upgraded by:   {self.total_upgrade} level(s)",
            f"  FINAL:         {self.final_certainty}",
        ]
        if self.pooled_effect is not None:
            lines.append(
                f"  Effect:        {self.pooled_effect:.3f} "
                f"(95% CI: {self.ci_lower:.3f}, {self.ci_upper:.3f})"
            )
        for d in self.domains:
            flag = "v" * d.downgrade if d.downgrade > 0 else "ok"
            lines.append(f"    [{flag}] {d.domain}: {d.rating}")
        return "\n".join(lines)


class GRADEAssessment:
    """Perform GRADE certainty of evidence assessment."""

    def assess(
        self,
        outcome: str,
        study_design: str,
        n_studies: int,
        total_n: int,
        risk_of_bias: str = NO_CONCERNS,
        inconsistency: str = NO_CONCERNS,
        indirectness: str = NO_CONCERNS,
        imprecision: str = NO_CONCERNS,
        publication_bias: str = NO_CONCERNS,
        large_effect: bool = False,
        dose_response: bool = False,
        confounding_attenuates: bool = False,
        pooled_effect: Optional[float] = None,
        ci_lower: Optional[float] = None,
        ci_upper: Optional[float] = None,
        i2: Optional[float] = None,
        meta_result=None,
        tsa_result=None,
        notes: Optional[str] = None,
    ) -> GRADERating:
        # Auto-ratings from meta/tsa objects
        if meta_result is not None:
            pooled_effect = pooled_effect or meta_result.pooled_effect
            ci_lower = ci_lower or meta_result.ci_lower
            ci_upper = ci_upper or meta_result.ci_upper
            if i2 is None and meta_result.i2 is not None:
                i2 = meta_result.i2
            if i2 is not None:
                inconsistency = self._rate_inconsistency(i2, inconsistency)

        if tsa_result is not None:
            imprecision = self._rate_imprecision_from_tsa(tsa_result, imprecision)

        if (
            pooled_effect is not None
            and not large_effect
            and meta_result is not None
        ):
            if meta_result.effect_type in ("OR", "RR"):
                large_effect = pooled_effect > 2.0 or pooled_effect < 0.5
            else:
                large_effect = abs(pooled_effect) > 0.8

        design_key = study_design.lower().replace(" ", "-")
        start_idx = DESIGN_START.get(design_key, DESIGN_START.get(study_design, 1))
        start_certainty = CERTAINTY_LEVELS[start_idx]

        domains: List[GRADEDomain] = []
        total_down = 0

        rob_down = self._downgrade_amount(risk_of_bias)
        domains.append(GRADEDomain("Risk of bias", risk_of_bias, rob_down))
        total_down += rob_down

        incon_down = self._downgrade_amount(inconsistency)
        domains.append(GRADEDomain("Inconsistency", inconsistency, incon_down))
        total_down += incon_down

        indir_down = self._downgrade_amount(indirectness)
        domains.append(GRADEDomain("Indirectness", indirectness, indir_down))
        total_down += indir_down

        imprecision_down = self._downgrade_amount(imprecision)
        domains.append(GRADEDomain("Imprecision", imprecision, imprecision_down))
        total_down += imprecision_down

        pub_bias_down = self._downgrade_amount(publication_bias, domain="publication_bias")
        domains.append(GRADEDomain("Publication bias", publication_bias, pub_bias_down))
        total_down += pub_bias_down

        total_up = 0
        if start_idx <= 1:
            if large_effect:
                up = 2 if pooled_effect is not None and (
                    (meta_result and meta_result.effect_type in ("OR", "RR") and
                     (pooled_effect > 5.0 or pooled_effect < 0.2))
                    or abs(pooled_effect) > 2.0
                ) else 1
                domains.append(GRADEDomain("Large effect", "upgrade", -up, notes="Upgrade"))
                total_up += up
            if dose_response:
                domains.append(GRADEDomain("Dose-response", "upgrade", -1, notes="Upgrade"))
                total_up += 1
            if confounding_attenuates:
                domains.append(GRADEDomain("Confounding attenuates", "upgrade", -1, notes="Upgrade"))
                total_up += 1

        final_idx = max(0, min(3, start_idx - total_down + total_up))
        final_certainty = CERTAINTY_LEVELS[final_idx]

        return GRADERating(
            outcome=outcome,
            study_design=study_design,
            n_studies=n_studies,
            total_n=total_n,
            starting_certainty=start_certainty,
            final_certainty=final_certainty,
            final_certainty_index=final_idx,
            total_downgrade=total_down,
            total_upgrade=total_up,
            domains=domains,
            pooled_effect=pooled_effect,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            notes=notes,
        )

    @staticmethod
    def _downgrade_amount(rating: str, domain: str = "") -> int:
        mapping = {
            NO_CONCERNS: 0,
            "undetected": 0,
            SOME_CONCERNS: 0,
            "low": 0,
            SERIOUS: 1,
            VERY_SERIOUS: 2,
        }
        if domain == "publication_bias":
            return min(1, mapping.get(rating.lower(), 0))
        return mapping.get(rating.lower(), 0)

    @staticmethod
    def _rate_inconsistency(i2: float, current: str) -> str:
        severity_order = [NO_CONCERNS, SOME_CONCERNS, SERIOUS, VERY_SERIOUS]
        current_idx = severity_order.index(current) if current in severity_order else 0
        if i2 >= 75:
            auto = VERY_SERIOUS
        elif i2 >= 50:
            auto = SERIOUS
        elif i2 >= 25:
            auto = SOME_CONCERNS
        else:
            auto = NO_CONCERNS
        auto_idx = severity_order.index(auto)
        return severity_order[max(current_idx, auto_idx)]

    @staticmethod
    def _rate_imprecision_from_tsa(tsa_result: Any, current: str) -> str:
        conclusion = getattr(tsa_result, "conclusion", "")
        if conclusion == "FIRM_EVIDENCE":
            return NO_CONCERNS
        if conclusion == "INSUFFICIENT":
            severity_order = [NO_CONCERNS, SOME_CONCERNS, SERIOUS, VERY_SERIOUS]
            current_idx = severity_order.index(current) if current in severity_order else 0
            return severity_order[max(current_idx, 2)]
        return current
