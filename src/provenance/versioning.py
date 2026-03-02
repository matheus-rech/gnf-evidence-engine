"""Evidence versioning with semantic version numbers and diffs.

Each meta-analysis run produces a versioned snapshot of the evidence
state. This module tracks what changed between versions:
  - Studies added / removed
  - Changes in pooled effect and CI
  - Changes in GRADE certainty rating

Versioning follows semver-inspired conventions:
  - MAJOR.MINOR.PATCH
  - MAJOR: Change in conclusion (FIRM_EVIDENCE status changes)
  - MINOR: New studies added or removed
  - PATCH: Effect size changes within 10% of previous CI width
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class EvidenceSnapshot:
    """A complete snapshot of the evidence state at a point in time."""

    version: str
    timestamp: str
    study_ids: List[str]
    pooled_effect: float
    ci_lower: float
    ci_upper: float
    i2: float
    n_studies: int
    total_n: int
    grade_certainty: Optional[str] = None
    tsa_conclusion: Optional[str] = None
    meta_parameters: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceSnapshot":
        return cls(**d)


@dataclass
class VersionDiff:
    """Differences between two evidence snapshots."""

    from_version: str
    to_version: str
    studies_added: List[str]
    studies_removed: List[str]
    effect_change: float
    effect_change_pct: float
    i2_change: float
    conclusion_changed: bool
    grade_changed: bool
    bump_type: str

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"Version diff: {self.from_version} to {self.to_version} ({self.bump_type} bump)",
            f"  Studies added:   {len(self.studies_added)} {self.studies_added[:3]}",
            f"  Studies removed: {len(self.studies_removed)}",
            f"  Effect change:   {self.effect_change:+.4f} ({self.effect_change_pct:+.1f}%)",
            f"  I2 change:       {self.i2_change:+.1f}%",
        ]
        if self.conclusion_changed:
            lines.append("  TSA conclusion changed!")
        if self.grade_changed:
            lines.append("  GRADE certainty changed!")
        return "\n".join(lines)


class EvidenceVersioning:
    """Manage versioned snapshots of the evidence state.

    Args:
        history_path: Path to JSON file storing snapshot history.
    """

    def __init__(self, history_path: str | Path) -> None:
        self.history_path = Path(history_path)
        self._history: List[EvidenceSnapshot] = []
        self._load()

    def _load(self) -> None:
        if self.history_path.exists():
            try:
                with open(self.history_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._history = [EvidenceSnapshot.from_dict(s) for s in raw]
            except Exception as exc:
                logger.warning("Could not load snapshot history: %s", exc)
                self._history = []

    def _save(self) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in self._history], f, indent=2)

    @property
    def latest(self) -> Optional[EvidenceSnapshot]:
        return self._history[-1] if self._history else None

    def record(
        self,
        study_ids: List[str],
        pooled_effect: float,
        ci_lower: float,
        ci_upper: float,
        i2: float,
        n_studies: int,
        total_n: int,
        grade_certainty: Optional[str] = None,
        tsa_conclusion: Optional[str] = None,
        meta_parameters: Optional[dict] = None,
        notes: Optional[str] = None,
    ) -> EvidenceSnapshot:
        prev = self.latest
        new_version = self._next_version(
            prev=prev,
            new_study_ids=study_ids,
            new_effect=pooled_effect,
            new_ci_lower=ci_lower,
            new_ci_upper=ci_upper,
            new_tsa=tsa_conclusion,
            new_grade=grade_certainty,
        )

        snap = EvidenceSnapshot(
            version=new_version,
            timestamp=datetime.utcnow().isoformat(),
            study_ids=study_ids,
            pooled_effect=pooled_effect,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            i2=i2,
            n_studies=n_studies,
            total_n=total_n,
            grade_certainty=grade_certainty,
            tsa_conclusion=tsa_conclusion,
            meta_parameters=meta_parameters or {},
            notes=notes,
        )
        self._history.append(snap)
        self._save()
        return snap

    def diff(
        self,
        from_version: Optional[str] = None,
        to_version: Optional[str] = None,
    ) -> Optional[VersionDiff]:
        if len(self._history) < 2:
            return None

        snap_old = (
            self._get_by_version(from_version)
            if from_version
            else self._history[-2]
        )
        snap_new = (
            self._get_by_version(to_version)
            if to_version
            else self._history[-1]
        )

        if snap_old is None or snap_new is None:
            return None

        old_ids = set(snap_old.study_ids)
        new_ids = set(snap_new.study_ids)
        added = sorted(new_ids - old_ids)
        removed = sorted(old_ids - new_ids)

        effect_change = snap_new.pooled_effect - snap_old.pooled_effect
        effect_pct = (
            effect_change / abs(snap_old.pooled_effect) * 100
            if snap_old.pooled_effect != 0
            else 0.0
        )
        i2_change = snap_new.i2 - snap_old.i2
        conclusion_changed = snap_new.tsa_conclusion != snap_old.tsa_conclusion
        grade_changed = snap_new.grade_certainty != snap_old.grade_certainty
        bump_type = self._classify_bump(
            added, removed, conclusion_changed, effect_change,
            snap_old.ci_upper - snap_old.ci_lower
        )

        return VersionDiff(
            from_version=snap_old.version,
            to_version=snap_new.version,
            studies_added=added,
            studies_removed=removed,
            effect_change=effect_change,
            effect_change_pct=effect_pct,
            i2_change=i2_change,
            conclusion_changed=conclusion_changed,
            grade_changed=grade_changed,
            bump_type=bump_type,
        )

    def _get_by_version(self, version: str) -> Optional[EvidenceSnapshot]:
        for snap in self._history:
            if snap.version == version:
                return snap
        return None

    @staticmethod
    def _classify_bump(
        added: List[str],
        removed: List[str],
        conclusion_changed: bool,
        effect_change: float,
        ci_width: float,
    ) -> str:
        if conclusion_changed:
            return "major"
        if added or removed:
            return "minor"
        if ci_width > 0 and abs(effect_change) < 0.10 * ci_width:
            return "patch"
        return "minor"

    def _next_version(
        self,
        prev: Optional[EvidenceSnapshot],
        new_study_ids: List[str],
        new_effect: float,
        new_ci_lower: float,
        new_ci_upper: float,
        new_tsa: Optional[str],
        new_grade: Optional[str],
    ) -> str:
        if prev is None:
            return "1.0.0"
        parts = [int(x) for x in prev.version.split(".")]
        major, minor, patch = parts[0], parts[1], parts[2]
        old_ids = set(prev.study_ids)
        new_ids = set(new_study_ids)
        added = new_ids - old_ids
        removed = old_ids - new_ids
        conclusion_changed = prev.tsa_conclusion != new_tsa
        if conclusion_changed:
            return f"{major + 1}.0.0"
        if added or removed:
            return f"{major}.{minor + 1}.0"
        return f"{major}.{minor}.{patch + 1}"

    def all_versions(self) -> List[str]:
        return [s.version for s in self._history]
