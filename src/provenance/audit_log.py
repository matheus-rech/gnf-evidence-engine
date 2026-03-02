"""Append-only audit log for regulatory compliance.

Records every modification to the evidence base with:
  - Who performed the action (user/system)
  - What action was taken (add_study, update_study, run_analysis, etc.)
  - When (UTC timestamp)
  - Why (reason / notes)
  - What changed (details dict)

The log is stored as newline-delimited JSON (NDJSON) for easy ingestion
into compliance systems. It is never modified in-place -- only appended.

Suitable for export in regulatory submissions (FDA, EMA evidence dossiers).
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Canonical action codes
VALID_ACTIONS = {
    "add_study",
    "update_study",
    "remove_study",
    "run_analysis",
    "run_tsa",
    "grade_assessment",
    "export",
    "import",
    "system_event",
    "user_note",
}


@dataclass
class AuditEntry:
    """One audit log entry."""

    entry_id: int
    timestamp: str
    actor: str
    action: str
    target_id: Optional[str]
    reason: str
    details: Dict[str, Any]
    checksum: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AuditEntry":
        return cls(**d)


class AuditLog:
    """Append-only audit log for evidence base modifications.

    Thread-safe. Uses a file lock to prevent concurrent writes.

    Args:
        log_path: Path to the NDJSON log file.
        actor: Default actor name (can be overridden per log call).
    """

    def __init__(
        self,
        log_path: str | Path,
        actor: str = "system",
    ) -> None:
        self.log_path = Path(log_path)
        self.default_actor = actor
        self._lock = threading.Lock()
        self._entry_count = self._count_existing_entries()

    def _count_existing_entries(self) -> int:
        if not self.log_path.exists():
            return 0
        count = 0
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
        except OSError:
            pass
        return count

    def log(
        self,
        action: str,
        reason: str,
        target_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        actor: Optional[str] = None,
    ) -> AuditEntry:
        if action not in VALID_ACTIONS:
            raise ValueError(
                f"Unknown action '{action}'. Valid actions: {sorted(VALID_ACTIONS)}"
            )

        with self._lock:
            self._entry_count += 1
            entry_id = self._entry_count
            timestamp = datetime.utcnow().isoformat()
            effective_actor = actor or self.default_actor
            details_clean = details or {}

            content = {
                "entry_id": entry_id,
                "timestamp": timestamp,
                "actor": effective_actor,
                "action": action,
                "target_id": target_id,
                "reason": reason,
                "details": details_clean,
            }
            import hashlib
            checksum = hashlib.sha256(
                json.dumps(content, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]

            entry = AuditEntry(
                entry_id=entry_id,
                timestamp=timestamp,
                actor=effective_actor,
                action=action,
                target_id=target_id,
                reason=reason,
                details=details_clean,
                checksum=checksum,
            )

            self._append(entry)
            return entry

    def _append(self, entry: AuditEntry) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def read_all(self) -> List[AuditEntry]:
        entries: List[AuditEntry] = []
        if not self.log_path.exists():
            return entries
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(AuditEntry.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError) as exc:
                        logger.warning("Skipping malformed audit entry: %s", exc)
        return entries

    def read_by_actor(self, actor: str) -> List[AuditEntry]:
        return [e for e in self.read_all() if e.actor == actor]

    def read_by_action(self, action: str) -> List[AuditEntry]:
        return [e for e in self.read_all() if e.action == action]

    def export_csv(self, output_path: str | Path) -> Path:
        import csv
        output_path = Path(output_path)
        entries = self.read_all()
        if not entries:
            return output_path
        fieldnames = ["entry_id", "timestamp", "actor", "action", "target_id", "reason", "checksum"]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for e in entries:
                writer.writerow(e.to_dict())
        return output_path

    def verify_integrity(self) -> bool:
        import hashlib
        entries = self.read_all()
        all_ok = True
        for entry in entries:
            content = {
                "entry_id": entry.entry_id,
                "timestamp": entry.timestamp,
                "actor": entry.actor,
                "action": entry.action,
                "target_id": entry.target_id,
                "reason": entry.reason,
                "details": entry.details,
            }
            expected = hashlib.sha256(
                json.dumps(content, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]
            if expected != entry.checksum:
                all_ok = False
        return all_ok

    @property
    def n_entries(self) -> int:
        return self._count_existing_entries()
