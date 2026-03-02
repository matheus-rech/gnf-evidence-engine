"""Tests for provenance modules: tracker, versioning, audit log."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.provenance.tracker import ProvenanceTracker, ProvenanceEntry
from src.provenance.versioning import EvidenceVersioning, EvidenceSnapshot
from src.provenance.audit_log import AuditLog, AuditEntry, VALID_ACTIONS


class TestProvenanceTracker:
    """Tests for the SHA-256 provenance tracker."""

    def test_register_new_study(self, tmp_provenance_path):
        tracker = ProvenanceTracker(log_path=tmp_provenance_path)
        entry = tracker.register(study_id="study_001", content={"title": "Study One", "year": 2021})
        assert entry.version == 1
        assert entry.study_id == "study_001"
        assert len(entry.content_hash) == 64

    def test_unchanged_content_returns_same_version(self, tmp_provenance_path):
        tracker = ProvenanceTracker(log_path=tmp_provenance_path)
        content = {"title": "Study One", "year": 2021}
        e1 = tracker.register(study_id="s1", content=content)
        e2 = tracker.register(study_id="s1", content=content)
        assert e1.version == e2.version == 1

    def test_changed_content_bumps_version(self, tmp_provenance_path):
        tracker = ProvenanceTracker(log_path=tmp_provenance_path)
        e1 = tracker.register("s1", {"title": "Old Title", "year": 2020})
        e2 = tracker.register("s1", {"title": "New Title", "year": 2020})
        assert e1.version == 1
        assert e2.version == 2

    def test_hash_deterministic(self, tmp_provenance_path):
        tracker = ProvenanceTracker(log_path=tmp_provenance_path)
        content = {"a": 1, "b": [1, 2, 3], "c": "hello"}
        h1 = tracker.compute_hash(content)
        h2 = tracker.compute_hash(content)
        assert h1 == h2

    def test_hash_changes_with_content(self, tmp_provenance_path):
        tracker = ProvenanceTracker(log_path=tmp_provenance_path)
        h1 = tracker.compute_hash({"a": 1})
        h2 = tracker.compute_hash({"a": 2})
        assert h1 != h2

    def test_persistence_across_instances(self, tmp_provenance_path):
        t1 = ProvenanceTracker(log_path=tmp_provenance_path)
        t1.register("s1", {"x": 1})
        t2 = ProvenanceTracker(log_path=tmp_provenance_path)
        history = t2.get_history("s1")
        assert len(history) == 1

    def test_get_history_empty_for_unknown_study(self, tmp_provenance_path):
        tracker = ProvenanceTracker(log_path=tmp_provenance_path)
        assert tracker.get_history("nonexistent") == []

    def test_detect_new_vs_changed(self, tmp_provenance_path):
        tracker = ProvenanceTracker(log_path=tmp_provenance_path)
        tracker.register("s1", {"title": "Study 1"})
        ids = ["s1", "s2"]
        contents = [{"title": "UPDATED Study 1"}, {"title": "Study 2"}]
        result = tracker.detect_new_or_changed(ids, contents)
        assert result.get("s1") == "changed"
        assert result.get("s2") == "new"

    def test_summary_stats(self, tmp_provenance_path):
        tracker = ProvenanceTracker(log_path=tmp_provenance_path)
        tracker.register("s1", {"x": 1})
        tracker.register("s1", {"x": 2})
        tracker.register("s2", {"y": 1})
        summary = tracker.summary()
        assert summary["n_studies"] == 2
        assert summary["n_total_versions"] == 3
        assert summary["n_studies_updated"] == 1

    def test_source_url_stored(self, tmp_provenance_path):
        tracker = ProvenanceTracker(log_path=tmp_provenance_path)
        entry = tracker.register(
            "s1", {"title": "x"}, source_url="https://pubmed.ncbi.nlm.nih.gov/12345/"
        )
        assert entry.source_url == "https://pubmed.ncbi.nlm.nih.gov/12345/"


class TestEvidenceVersioning:
    """Tests for semantic evidence versioning."""

    def _make_snap(self, versioning, study_ids, effect=None, tsa=None, grade=None, n=10):
        return versioning.record(
            study_ids=study_ids,
            pooled_effect=effect or -0.75,
            ci_lower=-1.0, ci_upper=-0.5, i2=30.0,
            n_studies=len(study_ids), total_n=len(study_ids) * n,
            tsa_conclusion=tsa or "INSUFFICIENT",
            grade_certainty=grade or "MODERATE",
        )

    def test_first_version_is_1_0_0(self, tmp_version_path):
        v = EvidenceVersioning(tmp_version_path)
        snap = self._make_snap(v, ["s1", "s2", "s3"])
        assert snap.version == "1.0.0"

    def test_adding_studies_bumps_minor(self, tmp_version_path):
        v = EvidenceVersioning(tmp_version_path)
        self._make_snap(v, ["s1", "s2"])
        snap2 = self._make_snap(v, ["s1", "s2", "s3"])
        assert snap2.version == "1.1.0"

    def test_conclusion_change_bumps_major(self, tmp_version_path):
        v = EvidenceVersioning(tmp_version_path)
        self._make_snap(v, ["s1", "s2"], tsa="INSUFFICIENT")
        snap2 = self._make_snap(v, ["s1", "s2"], tsa="FIRM_EVIDENCE")
        assert snap2.version.startswith("2.")

    def test_small_effect_change_bumps_patch(self, tmp_version_path):
        v = EvidenceVersioning(tmp_version_path)
        self._make_snap(v, ["s1", "s2", "s3"], effect=-0.750)
        snap2 = self._make_snap(v, ["s1", "s2", "s3"], effect=-0.751)
        assert snap2.version.endswith(".1")

    def test_latest_returns_most_recent(self, tmp_version_path):
        v = EvidenceVersioning(tmp_version_path)
        self._make_snap(v, ["s1"])
        self._make_snap(v, ["s1", "s2"])
        assert v.latest.version == "1.1.0"

    def test_diff_detects_added_studies(self, tmp_version_path):
        v = EvidenceVersioning(tmp_version_path)
        self._make_snap(v, ["s1", "s2"])
        self._make_snap(v, ["s1", "s2", "s3"])
        diff = v.diff()
        assert "s3" in diff.studies_added
        assert diff.studies_removed == []

    def test_diff_detects_removed_studies(self, tmp_version_path):
        v = EvidenceVersioning(tmp_version_path)
        self._make_snap(v, ["s1", "s2", "s3"])
        self._make_snap(v, ["s1", "s2"])
        diff = v.diff()
        assert "s3" in diff.studies_removed

    def test_no_diff_with_single_snapshot(self, tmp_version_path):
        v = EvidenceVersioning(tmp_version_path)
        self._make_snap(v, ["s1"])
        assert v.diff() is None

    def test_persistence(self, tmp_version_path):
        v1 = EvidenceVersioning(tmp_version_path)
        self._make_snap(v1, ["s1", "s2"])
        v2 = EvidenceVersioning(tmp_version_path)
        assert v2.latest is not None
        assert v2.latest.version == "1.0.0"


class TestAuditLog:
    """Tests for the append-only audit log."""

    def test_log_creates_entry(self, tmp_audit_path):
        audit = AuditLog(log_path=tmp_audit_path, actor="test_user")
        entry = audit.log(action="add_study", reason="Added from PubMed search", target_id="study_001")
        assert entry.entry_id == 1
        assert entry.action == "add_study"
        assert entry.actor == "test_user"

    def test_entries_are_sequential(self, tmp_audit_path):
        audit = AuditLog(log_path=tmp_audit_path)
        for i in range(5):
            e = audit.log(action="system_event", reason=f"Event {i}")
        entries = audit.read_all()
        ids = [e.entry_id for e in entries]
        assert ids == list(range(1, 6))

    def test_append_only_no_modification(self, tmp_audit_path):
        audit = AuditLog(log_path=tmp_audit_path)
        e1 = audit.log(action="add_study", reason="First entry", target_id="s1")
        original_ts = e1.timestamp
        audit2 = AuditLog(log_path=tmp_audit_path)
        audit2.log(action="update_study", reason="Second entry", target_id="s2")
        entries = audit2.read_all()
        assert entries[0].timestamp == original_ts

    def test_invalid_action_raises(self, tmp_audit_path):
        audit = AuditLog(log_path=tmp_audit_path)
        with pytest.raises(ValueError, match="Unknown action"):
            audit.log(action="invalid_action", reason="test")

    def test_filter_by_actor(self, tmp_audit_path):
        audit = AuditLog(log_path=tmp_audit_path)
        audit.log(action="add_study", reason="User action", actor="alice")
        audit.log(action="run_analysis", reason="System job", actor="system")
        audit.log(action="export", reason="User export", actor="alice")
        alice_entries = audit.read_by_actor("alice")
        assert len(alice_entries) == 2
        assert all(e.actor == "alice" for e in alice_entries)

    def test_filter_by_action(self, tmp_audit_path):
        audit = AuditLog(log_path=tmp_audit_path)
        for i in range(3):
            audit.log(action="add_study", reason=f"Study {i}")
        audit.log(action="run_analysis", reason="Analysis")
        add_entries = audit.read_by_action("add_study")
        assert len(add_entries) == 3

    def test_checksum_integrity(self, tmp_audit_path):
        audit = AuditLog(log_path=tmp_audit_path)
        for i in range(5):
            audit.log(action="system_event", reason=f"Event {i}", target_id=f"s{i}")
        assert audit.verify_integrity()

    def test_checksum_failure_on_tampered_log(self, tmp_audit_path):
        audit = AuditLog(log_path=tmp_audit_path)
        audit.log(action="add_study", reason="Legit entry")
        with open(tmp_audit_path, "r") as f:
            content = f.read()
        tampered = content.replace("Legit entry", "Tampered entry")
        with open(tmp_audit_path, "w") as f:
            f.write(tampered)
        audit2 = AuditLog(log_path=tmp_audit_path)
        assert not audit2.verify_integrity()

    def test_export_csv(self, tmp_audit_path, tmp_path):
        import csv
        audit = AuditLog(log_path=tmp_audit_path)
        for action in ["add_study", "run_analysis", "export"]:
            audit.log(action=action, reason=f"{action} reason")
        csv_path = tmp_path / "audit_export.csv"
        audit.export_csv(csv_path)
        assert csv_path.exists()
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3
        assert rows[0]["action"] == "add_study"

    def test_actor_override_per_call(self, tmp_audit_path):
        audit = AuditLog(log_path=tmp_audit_path, actor="default_actor")
        e = audit.log(action="user_note", reason="Note", actor="override_actor")
        assert e.actor == "override_actor"

    def test_valid_all_action_codes(self, tmp_audit_path):
        audit = AuditLog(log_path=tmp_audit_path)
        for action in VALID_ACTIONS:
            entry = audit.log(action=action, reason=f"Test {action}")
            assert entry.action == action

    def test_thread_safe_sequential_ids(self, tmp_audit_path):
        import threading
        audit = AuditLog(log_path=tmp_audit_path)
        errors = []

        def log_entry():
            try:
                audit.log(action="system_event", reason="Thread test")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=log_entry) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        entries = audit.read_all()
        ids = sorted(e.entry_id for e in entries)
        assert len(ids) == 20
