"""StudyRecord dataclass for storing structured study information.

Each StudyRecord corresponds to a single clinical study and aggregates
all effect records extracted from it along with provenance metadata.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal

from .effect_record import EffectRecord

logger = logging.getLogger(__name__)

StudyDesign = Literal[
    "RCT",
    "cohort",
    "cross-sectional",
    "case-control",
    "case-series",
    "systematic-review",
    "meta-analysis",
    "other",
]


@dataclass
class StudyRecord:
    """Complete representation of a single study in the evidence base.

    Attributes:
        study_id: Unique internal identifier for this study.
        title: Full title of the publication.
        authors: List of author names (Last, First format preferred).
        year: Publication year.
        journal: Journal name.
        pmid: PubMed identifier (without "PMID:" prefix).
        doi: Digital Object Identifier (without "doi:" prefix).
        study_design: Study design category.
        population: Description of the study population (e.g., "Adults with MDD").
        intervention: Description of the treatment/exposure.
        comparator: Description of the comparison condition.
        outcome: Primary outcome measure description.
        sample_size: Total enrolled sample size.
        effect_records: List of EffectRecord objects extracted from this study.
        risk_of_bias: Risk-of-bias assessment dict. Keys are domain names;
            values are ("low", "some_concerns", "high").
        extraction_timestamp: When data was extracted / last updated.
        provenance_hash: SHA-256 hash of study content for change detection.
        abstract: Full abstract text (optional).
        full_text_path: Path to local PDF / full-text file (optional).
        registration: Trial registration identifier (e.g., NCT number).
        funding: Funding sources description.
        tags: Arbitrary keyword tags for filtering.
    """

    study_id: str
    title: str
    authors: List[str]
    year: int
    journal: str
    study_design: StudyDesign
    population: str
    intervention: str
    comparator: str
    outcome: str
    sample_size: int
    effect_records: List[EffectRecord] = field(default_factory=list)
    pmid: Optional[str] = None
    doi: Optional[str] = None
    risk_of_bias: Optional[Dict[str, str]] = None
    extraction_timestamp: datetime = field(default_factory=datetime.utcnow)
    provenance_hash: Optional[str] = None
    abstract: Optional[str] = None
    full_text_path: Optional[str] = None
    registration: Optional[str] = None
    funding: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate fields and compute provenance hash if not provided."""
        self._validate()
        if self.provenance_hash is None:
            self.provenance_hash = self.compute_hash()

    def _validate(self) -> None:
        """Validate required field constraints.

        Raises:
            ValueError: If any field violates constraints.
        """
        if not self.study_id:
            raise ValueError("study_id must not be empty")
        if not self.title:
            raise ValueError("title must not be empty")
        if self.year < 1900 or self.year > datetime.utcnow().year + 1:
            raise ValueError(f"year {self.year} is out of plausible range")
        if self.sample_size < 1:
            raise ValueError(f"sample_size must be >= 1, got {self.sample_size}")

    def compute_hash(self) -> str:
        """Compute SHA-256 provenance hash from study content.

        The hash covers the stable, content-derived fields (not timestamps).

        Returns:
            64-character hex string.
        """
        content = json.dumps(
            {
                "study_id": self.study_id,
                "title": self.title,
                "authors": sorted(self.authors),
                "year": self.year,
                "journal": self.journal,
                "study_design": self.study_design,
                "population": self.population,
                "intervention": self.intervention,
                "comparator": self.comparator,
                "outcome": self.outcome,
                "sample_size": self.sample_size,
                "pmid": self.pmid,
                "doi": self.doi,
                "effect_records": [er.to_dict() for er in self.effect_records],
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @property
    def first_author(self) -> str:
        """Return first author surname.

        Returns:
            First author name, or "Unknown" if no authors listed.
        """
        if not self.authors:
            return "Unknown"
        name = self.authors[0]
        # Handle "Last, First" format
        if "," in name:
            return name.split(",")[0].strip()
        return name.split()[-1] if " " in name else name

    @property
    def citation_label(self) -> str:
        """Short citation label for forest plots.

        Returns:
            e.g., "Smith 2021" or "Smith et al. 2021"
        """
        if len(self.authors) == 1:
            return f"{self.first_author} {self.year}"
        return f"{self.first_author} et al. {self.year}"

    @property
    def overall_risk_of_bias(self) -> Optional[str]:
        """Aggregate risk-of-bias rating across domains.

        Uses the most conservative (highest risk) domain rating.

        Returns:
            "low", "some_concerns", "high", or None if not assessed.
        """
        if not self.risk_of_bias:
            return None
        priority = {"high": 2, "some_concerns": 1, "low": 0}
        ratings = [priority.get(v, 0) for v in self.risk_of_bias.values()]
        reverse = {0: "low", 1: "some_concerns", 2: "high"}
        return reverse[max(ratings)]

    def get_effects_by_outcome(self, outcome_name: str) -> List[EffectRecord]:
        """Filter effect records by outcome name.

        Args:
            outcome_name: Outcome label to filter on (case-insensitive substring match).

        Returns:
            Matching EffectRecord objects.
        """
        return [
            er
            for er in self.effect_records
            if outcome_name.lower() in er.outcome_name.lower()
        ]

    def to_dict(self, include_effects: bool = True) -> Dict[str, Any]:
        """Serialise to a plain dictionary.

        Args:
            include_effects: Whether to include nested EffectRecord dicts.

        Returns:
            Dictionary representation.
        """
        d = asdict(self)
        d["extraction_timestamp"] = self.extraction_timestamp.isoformat()
        if not include_effects:
            d.pop("effect_records", None)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StudyRecord":
        """Reconstruct a StudyRecord from a dictionary.

        Args:
            data: As produced by ``to_dict()``.

        Returns:
            Reconstructed StudyRecord.
        """
        data = dict(data)
        if "extraction_timestamp" in data and isinstance(
            data["extraction_timestamp"], str
        ):
            data["extraction_timestamp"] = datetime.fromisoformat(
                data["extraction_timestamp"]
            )
        if "effect_records" in data and data["effect_records"]:
            data["effect_records"] = [
                EffectRecord.from_dict(er) for er in data["effect_records"]
            ]
        # Remove computed properties that are not constructor parameters
        for key in ("first_author", "citation_label", "overall_risk_of_bias"):
            data.pop(key, None)
        return cls(**data)
