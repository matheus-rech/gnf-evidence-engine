"""Pytest configuration and shared fixtures for GNF Evidence Engine tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.structured_schema.effect_record import EffectRecord
from src.structured_schema.study_record import StudyRecord


@pytest.fixture(autouse=True)
def seed_random():
    """Set numpy random seed before each test for reproducibility."""
    np.random.seed(42)
    yield


@pytest.fixture
def smd_effects():
    """Six SMD effect records with known properties for meta-analysis tests."""
    data = [
        (-0.82, -1.23, -0.41, 30, 30),
        (-0.65, -1.02, -0.28, 40, 40),
        (-0.71, -1.00, -0.42, 33, 33),
        (-0.93, -1.35, -0.51, 42, 42),
        (-0.59, -0.93, -0.25, 35, 35),
        (-0.77, -1.12, -0.42, 28, 28),
    ]
    return [
        EffectRecord(
            effect_type="SMD", effect_size=es, ci_lower=lo, ci_upper=hi,
            n_treatment=nt, n_control=nc, outcome_name="Depression severity",
        )
        for es, lo, hi, nt, nc in data
    ]


@pytest.fixture
def or_effects():
    """Four OR effect records (log-scale pooling)."""
    data = [
        (1.45, 1.10, 1.92, 50, 50),
        (1.32, 0.98, 1.78, 45, 45),
        (1.68, 1.25, 2.25, 60, 60),
        (1.21, 0.87, 1.68, 40, 40),
    ]
    return [
        EffectRecord(
            effect_type="OR", effect_size=es, ci_lower=lo, ci_upper=hi,
            n_treatment=nt, n_control=nc, outcome_name="Response",
        )
        for es, lo, hi, nt, nc in data
    ]


@pytest.fixture
def study_records(smd_effects):
    """Six StudyRecord objects with SMD effects."""
    records = []
    for i, eff in enumerate(smd_effects):
        records.append(
            StudyRecord(
                study_id=f"study_{i+1:03d}", title=f"Test Study {i+1}",
                authors=[f"Author {i+1}"], year=2015 + i,
                journal="Test Journal", study_design="RCT",
                population="Adults with MDD", intervention="Drug A",
                comparator="Placebo", outcome="MADRS change",
                sample_size=eff.n_total, effect_records=[eff],
            )
        )
    return records


@pytest.fixture
def effects_array():
    """Numpy array of effect sizes from the smd_effects fixture."""
    return np.array([-0.82, -0.65, -0.71, -0.93, -0.59, -0.77])


@pytest.fixture
def variances_array():
    """Numpy array of variances derived from CI widths."""
    import scipy.stats as stats
    z = stats.norm.ppf(0.975)
    ci_widths = np.array([
        (-0.41) - (-1.23), (-0.28) - (-1.02), (-0.42) - (-1.00),
        (-0.51) - (-1.35), (-0.25) - (-0.93), (-0.42) - (-1.12),
    ])
    ses = ci_widths / (2 * z)
    return ses ** 2


@pytest.fixture
def tmp_provenance_path(tmp_path):
    """Temporary path for provenance JSON log."""
    return tmp_path / "provenance.json"


@pytest.fixture
def tmp_audit_path(tmp_path):
    """Temporary path for audit NDJSON log."""
    return tmp_path / "audit.ndjson"


@pytest.fixture
def tmp_version_path(tmp_path):
    """Temporary path for versioning JSON."""
    return tmp_path / "versions.json"
