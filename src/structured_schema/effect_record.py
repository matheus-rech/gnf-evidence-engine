"""EffectRecord dataclass for storing extracted effect size data.

This module defines the atomic unit of a meta-analysis: a single
quantitative outcome extracted from one study arm comparison.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal

logger = logging.getLogger(__name__)

# Supported effect measure types
EffectType = Literal["SMD", "MD", "OR", "RR", "HR", "RD", "COR"]


@dataclass
class EffectRecord:
    """A single quantitative outcome from a study comparison.

    Attributes:
        effect_type: Effect measure type (SMD, MD, OR, RR, HR, RD, COR).
        effect_size: Point estimate of the effect.
        ci_lower: Lower bound of the confidence interval (95% by default).
        ci_upper: Upper bound of the confidence interval.
        se: Standard error of the effect estimate. If None, derived from CI.
        p_value: Two-tailed p-value for the effect estimate.
        n_treatment: Sample size in treatment arm.
        n_control: Sample size in control arm.
        outcome_name: Human-readable label for the outcome.
        timepoint: Follow-up timepoint (e.g., "6 weeks", "12 months").
        ci_level: Confidence level, default 0.95.
        variance: Variance of the effect size. Derived from SE if not provided.
        weight: Inverse-variance weight (computed during meta-analysis, not stored).
        notes: Any extraction notes or quality flags.
        record_id: Unique identifier for this record.
    """

    effect_type: EffectType
    effect_size: float
    ci_lower: float
    ci_upper: float
    n_treatment: int
    n_control: int
    outcome_name: str
    se: Optional[float] = None
    p_value: Optional[float] = None
    timepoint: Optional[str] = None
    ci_level: float = 0.95
    notes: Optional[str] = None
    record_id: Optional[str] = field(default=None)

    def __post_init__(self) -> None:
        """Validate and derive computed fields."""
        self._validate()
        if self.se is None:
            self.se = self._derive_se()
        if self.record_id is None:
            self.record_id = self._generate_id()

    def _validate(self) -> None:
        """Validate field constraints.

        Raises:
            ValueError: If any field violates expected constraints.
        """
        if self.ci_lower > self.ci_upper:
            raise ValueError(
                f"ci_lower ({self.ci_lower}) must be <= ci_upper ({self.ci_upper})"
            )
        if self.n_treatment <= 0:
            raise ValueError(f"n_treatment must be > 0, got {self.n_treatment}")
        if self.n_control <= 0:
            raise ValueError(f"n_control must be > 0, got {self.n_control}")
        if not (0.0 < self.ci_level < 1.0):
            raise ValueError(f"ci_level must be between 0 and 1, got {self.ci_level}")
        if self.p_value is not None and not (0.0 <= self.p_value <= 1.0):
            raise ValueError(f"p_value must be in [0, 1], got {self.p_value}")

    def _derive_se(self) -> float:
        """Derive standard error from confidence interval.

        Uses the formula SE = (CI_upper - CI_lower) / (2 * z_alpha/2).

        For log-transformed measures (OR, RR, HR), the CI is on the log scale.

        Returns:
            Derived standard error.
        """
        import scipy.stats as stats

        z = stats.norm.ppf(1 - (1 - self.ci_level) / 2)
        if self.effect_type in ("OR", "RR", "HR"):
            # Work on log scale
            import math
            log_upper = math.log(self.ci_upper) if self.ci_upper > 0 else float("nan")
            log_lower = math.log(self.ci_lower) if self.ci_lower > 0 else float("nan")
            return (log_upper - log_lower) / (2 * z)
        return (self.ci_upper - self.ci_lower) / (2 * z)

    @property
    def variance(self) -> float:
        """Variance of the effect size estimate.

        Returns:
            SE squared.
        """
        return self.se ** 2 if self.se is not None else float("nan")

    @property
    def n_total(self) -> int:
        """Total sample size across both arms.

        Returns:
            Sum of n_treatment and n_control.
        """
        return self.n_treatment + self.n_control

    def _generate_id(self) -> str:
        """Generate a deterministic record ID from content hash.

        Returns:
            8-character hex string.
        """
        content = json.dumps(
            {
                "effect_type": self.effect_type,
                "effect_size": self.effect_size,
                "ci_lower": self.ci_lower,
                "ci_upper": self.ci_upper,
                "n_treatment": self.n_treatment,
                "n_control": self.n_control,
                "outcome_name": self.outcome_name,
                "timepoint": self.timepoint,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:8]

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary.

        Returns:
            Dictionary representation of the record.
        """
        d = asdict(self)
        d["variance"] = self.variance
        d["n_total"] = self.n_total
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "EffectRecord":
        """Deserialise from a dictionary.

        Args:
            data: Dictionary as produced by ``to_dict()``.

        Returns:
            Reconstructed EffectRecord.
        """
        # Remove computed fields that are not constructor params
        clean = {
            k: v
            for k, v in data.items()
            if k in cls.__dataclass_fields__  # type: ignore[attr-defined]
        }
        return cls(**clean)
