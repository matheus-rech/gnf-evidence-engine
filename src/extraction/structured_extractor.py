"""Structured data extraction from study text.

Uses regex pattern matching to extract:
  - Sample sizes (n = ...)
  - Effect sizes (SMD, MD, OR, RR, HR, Cohen's d)
  - Confidence intervals
  - p-values
  - Study design indicators

Returns EffectRecord and StudyRecord objects.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from ..structured_schema.effect_record import EffectRecord

logger = logging.getLogger(__name__)

_NUM = r"[-+]?\d+(?:\.\d+)?"
_POS_NUM = r"\d+(?:\.\d+)?"

_RE_SAMPLE_SIZE = re.compile(rf"\bn\s*=\s*({_POS_NUM})\b", re.IGNORECASE)

_RE_EFFECT = re.compile(
    rf"""
    \b
    (?P<type>
        SMD | Hedge[\u2018\u2019']?s\s+g | Cohen[\u2018\u2019']?s?\s+d | MD |
        mean\s+difference | OR | odds\s+ratio |
        RR | risk\s+ratio | relative\s+risk |
        HR | hazard\s+ratio |
        r\b | correlation
    )
    \s*(?:=|was|of|:)\s*
    (?P<value>{_NUM})
    """,
    re.IGNORECASE | re.VERBOSE,
)

_RE_CI = re.compile(
    rf"""
    (?:
        (?:\d{{2}}\%\s+CI\s*[:\-\u2013\u2014]?\s*)
        |(?:\(|\[)
    )
    (?P<lower>{_NUM})
    \s*(?:,|;|\u2013|\u2014|\s+to\s+)\s*
    (?P<upper>{_NUM})
    (?:\)|\])?
    """,
    re.IGNORECASE | re.VERBOSE,
)

_RE_PVALUE = re.compile(
    r"""
    \bp[\s-]?(?:value)?\s*
    (?P<op>[=<>\u2264\u2265])
    \s*
    (?P<val>\.?\d+(?:\.\d+)?)
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_RE_ARM_N = re.compile(
    rf"""
    (?P<arm>treatment|intervention|experimental|placebo|control|comparison|active)
    \s*(?:group|arm|condition)?
    [^\n]{{0,40}}?
    n\s*=\s*(?P<n>{_POS_NUM})
    |
    n\s*=\s*(?P<n2>{_POS_NUM})
    [^\n]{{0,40}}?
    (?P<arm2>treatment|intervention|experimental|placebo|control|comparison|active)
    """,
    re.IGNORECASE,
)

_RE_OUTCOME_LABEL = re.compile(
    r"(?:primary\s+outcome|outcome\s+measure|endpoint)\s*[:\-\u2013]\s*([^\.\n]{5,80})",
    re.IGNORECASE,
)

_EFFECT_TYPE_MAP = {
    "smd": "SMD", "hedges g": "SMD", "hedge's g": "SMD", "hedge\u2019s g": "SMD",
    "cohens d": "SMD", "cohen's d": "SMD", "cohen\u2019s d": "SMD", "cohend": "SMD", "d": "SMD",
    "md": "MD", "mean difference": "MD",
    "or": "OR", "odds ratio": "OR",
    "rr": "RR", "risk ratio": "RR", "relative risk": "RR",
    "hr": "HR", "hazard ratio": "HR",
    "r": "COR", "correlation": "COR",
}


def _normalize_effect_type(raw: str) -> str:
    key = re.sub(r"\s+", " ", raw.lower().strip())
    return _EFFECT_TYPE_MAP.get(key, raw.upper()[:5])


def _extract_p_value(text: str) -> Optional[float]:
    m = _RE_PVALUE.search(text)
    if not m:
        return None
    try:
        p = float(m.group("val"))
        return min(p, 1.0)
    except ValueError:
        return None


def _extract_ci(text: str) -> Optional[Tuple[float, float]]:
    m = _RE_CI.search(text)
    if not m:
        return None
    try:
        lo = float(m.group("lower"))
        hi = float(m.group("upper"))
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi
    except (ValueError, IndexError):
        return None


def _extract_arm_sizes(text: str) -> Tuple[int, int]:
    treatment_n = 0
    control_n = 0
    for m in _RE_ARM_N.finditer(text):
        arm_key = (m.group("arm") or m.group("arm2") or "").lower()
        n_str = m.group("n") or m.group("n2") or "0"
        try:
            n = int(float(n_str))
        except ValueError:
            continue
        if any(kw in arm_key for kw in ("treatment", "intervention", "experimental", "active")):
            treatment_n = n
        elif any(kw in arm_key for kw in ("control", "placebo", "comparison")):
            control_n = n
    return treatment_n, control_n


class StructuredExtractor:
    """Extract structured quantitative data from study text."""

    def __init__(self, default_outcome: str = "Primary outcome", min_confidence_score: float = 0.5) -> None:
        self.default_outcome = default_outcome
        self.min_confidence_score = min_confidence_score

    def extract(self, text: str, study_id: Optional[str] = None) -> List[EffectRecord]:
        """Extract effect records from study text."""
        if not text or not text.strip():
            return []
        ctx = study_id or "unknown"
        records: List[EffectRecord] = []
        outcome = self._extract_outcome_label(text)
        effect_matches = list(_RE_EFFECT.finditer(text))
        if not effect_matches:
            logger.debug("[%s] No effect size patterns found", ctx)
            return []
        n_treat, n_ctrl = _extract_arm_sizes(text)
        total_n_matches = list(_RE_SAMPLE_SIZE.finditer(text))
        if n_treat == 0 and n_ctrl == 0 and total_n_matches:
            total = int(float(total_n_matches[0].group(1)))
            n_treat = total // 2
            n_ctrl = total - n_treat
        for match in effect_matches:
            raw_type = match.group("type")
            raw_value = match.group("value")
            try:
                effect_size = float(raw_value)
            except ValueError:
                continue
            effect_type = _normalize_effect_type(raw_type)
            context_start = max(0, match.start() - 100)
            context_end = min(len(text), match.end() + 300)
            context = text[context_start:context_end]
            ci = _extract_ci(context)
            p_val = _extract_p_value(context)
            if ci is None:
                continue
            ci_lower, ci_upper = ci
            if effect_type in ("OR", "RR", "HR"):
                if ci_lower <= 0 or ci_upper <= 0:
                    continue
            score = 0.4
            if n_treat > 0:
                score += 0.2
            if n_ctrl > 0:
                score += 0.2
            if p_val is not None:
                score += 0.2
            if score < self.min_confidence_score:
                continue
            try:
                record = EffectRecord(
                    effect_type=effect_type, effect_size=effect_size, ci_lower=ci_lower,
                    ci_upper=ci_upper, n_treatment=max(n_treat, 1), n_control=max(n_ctrl, 1),
                    outcome_name=outcome, p_value=p_val,
                    notes=f"Extracted by StructuredExtractor (score={score:.2f})",
                )
                records.append(record)
            except (ValueError, Exception) as exc:
                logger.warning("[%s] Could not build EffectRecord: %s", ctx, exc)
        seen: set = set()
        unique: List[EffectRecord] = []
        for rec in records:
            key = (rec.effect_type, rec.effect_size, rec.ci_lower, rec.ci_upper)
            if key not in seen:
                seen.add(key)
                unique.append(rec)
        logger.info("[%s] Extracted %d effect records", ctx, len(unique))
        return unique

    @staticmethod
    def _extract_outcome_label(text: str) -> str:
        m = _RE_OUTCOME_LABEL.search(text)
        if m:
            label = m.group(1).strip().rstrip(".,;")
            return label[:100]
        return "Primary outcome"

    def extract_from_sections(self, sections: dict, study_id: Optional[str] = None) -> List[EffectRecord]:
        """Extract effect records from a parsed sections dictionary."""
        priority = ["results", "abstract", "conclusion"]
        for key in priority:
            text = sections.get(key, "")
            if text:
                records = self.extract(text, study_id=study_id)
                if records:
                    return records
        all_text = "\n".join(sections.values())
        return self.extract(all_text, study_id=study_id)
