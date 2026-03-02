"""Provenance tracking for study records.

Computes and persists SHA-256 content hashes for every study record so
that changes between extraction runs can be detected automatically.

The provenance log is a JSON file storing one entry per (study_id, version)
pair. On each update run, newly added or changed studies are flagged.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProvenanceEntry:
    """One provenance record for a study extraction.

    Attributes:
        study_id: Unique study identifier.
        content_hash: SHA-256 of study content.
        extraction_timestamp: When this version was extracted.
        extractor: Extraction agent/tool identifier.
        version: Sequential integer version for this study.
        changed_fields: Fields that changed vs previous version (empty on first).
        source_url: Source URL or PMID link.
    """

    study_id: str
    content_hash: str
    extraction_timestamp: str  # ISO format
    extractor: str
    version: int
    changed_fields: List[str]
    source_url: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ProvenanceEntry":
        return cls(**d)


class ProvenanceTracker:
    """Track provenance for a collection of study records.

    Args:
        log_path: Path to the JSON provenance log file.
        extractor: Name/identifier of the extraction agent.
    """

    def __init__(
        self,
        log_path: str | Path,
        extractor: str = "gnf-evidence-engine",
    ) -> None:
        self.log_path = Path(log_path)
        self.extractor = extractor
        self._log: Dict[str, List[ProvenanceEntry]] = {}
        self._load()

    def _load(self) -> None:
        if self.log_path.exists():
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._log = {
                    sid: [ProvenanceEntry.from_dict(e) for e in entries]
                    for sid, entries in raw.items()
                }
                logger.info("Loaded provenance log (%d studies)", len(self._log))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("Could not load provenance log: %s. Starting fresh.", exc)
                self._log = {}
        else:
            self._log = {}

    def _save(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        raw = {
            sid: [e.to_dict() for e in entries]
            for sid, entries in self._log.items()
        }
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2, ensure_ascii=False)

    @staticmethod
    def compute_hash(content: dict) -> str:
        canonical = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def register(
        self,
        study_id: str,
        content: dict,
        source_url: Optional[str] = None,
    ) -> ProvenanceEntry:
        new_hash = self.compute_hash(content)
        history = self._log.get(study_id, [])
        version = len(history) + 1

        if history:
            latest = history[-1]
            if latest.content_hash == new_hash:
                return latest
            changed = self._detect_changed_fields(history[-1], content, new_hash)
        else:
            changed = []

        entry = ProvenanceEntry(
            study_id=study_id,
            content_hash=new_hash,
            extraction_timestamp=datetime.utcnow().isoformat(),
            extractor=self.extractor,
            version=version,
            changed_fields=changed,
            source_url=source_url,
        )
        self._log.setdefault(study_id, []).append(entry)
        self._save()
        return entry

    @staticmethod
    def _detect_changed_fields(
        prev_entry: ProvenanceEntry,
        new_content: dict,
        new_hash: str,
    ) -> List[str]:
        if prev_entry.content_hash == new_hash:
            return []
        return ["content_changed"]

    def get_history(self, study_id: str) -> List[ProvenanceEntry]:
        return self._log.get(study_id, [])

    def detect_new_or_changed(
        self,
        study_ids: List[str],
        contents: List[dict],
    ) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for sid, content in zip(study_ids, contents):
            new_hash = self.compute_hash(content)
            history = self._log.get(sid, [])
            if not history:
                result[sid] = "new"
            elif history[-1].content_hash != new_hash:
                result[sid] = "changed"
        return result

    def all_study_ids(self) -> List[str]:
        return list(self._log.keys())

    def summary(self) -> dict:
        return {
            "n_studies": len(self._log),
            "n_total_versions": sum(len(h) for h in self._log.values()),
            "n_studies_updated": sum(1 for h in self._log.values() if len(h) > 1),
            "log_path": str(self.log_path),
        }
