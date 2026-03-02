# %% [markdown]
# # 01 - PubMed Extraction Demo
#
# This notebook demonstrates:
# 1. Searching PubMed using the `PubMedFetcher`
# 2. Parsing study records into `StudyRecord` objects
# 3. Extracting structured effect data with `StructuredExtractor`
# 4. Persisting provenance hashes

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve().parent / "src"))

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from src.extraction.pubmed_fetcher import PubMedFetcher
from src.extraction.structured_extractor import StructuredExtractor
from src.structured_schema.study_record import StudyRecord
from src.provenance.tracker import ProvenanceTracker

print("Imports OK")

# %% [markdown]
# ## 1. Search PubMed

# %%
RESEARCHER_EMAIL = "researcher@example.com"

fetcher = PubMedFetcher(
    email=RESEARCHER_EMAIL,
    tool="gnf-evidence-engine-demo",
)

search_result = fetcher.esearch(
    query="ketamine treatment-resistant depression",
    mesh_terms=["Ketamine", "Depressive Disorder, Treatment-Resistant"],
    date_range=("2015/01/01", "2024/12/31"),
    study_types=["Randomized Controlled Trial"],
    max_results=20,
)

print(f"Total matching records: {search_result.total_count}")
print(f"PMIDs retrieved: {len(search_result.pmids)}")
print(f"Query translation: {search_result.query_translation}")

# %% [markdown]
# ## 2. Fetch and Parse StudyRecords

# %%
if search_result.pmids:
    demo_pmids = search_result.pmids[:5]
    studies = list(fetcher.efetch(demo_pmids))
    print(f"Parsed {len(studies)} studies")
else:
    from src.structured_schema.effect_record import EffectRecord
    print("No live results - using synthetic demo data")
    studies = [
        StudyRecord(
            study_id="pmid_demo_001",
            title="Ketamine vs Placebo in TRD: A Double-Blind RCT",
            authors=["Smith, John A.", "Jones, Mary B."],
            year=2021, journal="JAMA Psychiatry", pmid="12345001",
            doi="10.1001/jamapsychiatry.2021.0001",
            study_design="RCT",
            population="Adults 18-65 with TRD",
            intervention="Ketamine 0.5 mg/kg IV",
            comparator="Normal saline IV",
            outcome="MADRS score change at 1 week",
            sample_size=60,
            abstract="Results: SMD = -0.82 (95% CI: -1.23, -0.41), p < 0.001, n = 30 treatment, n = 30 control.",
        ),
        StudyRecord(
            study_id="pmid_demo_002",
            title="Rapid Antidepressant Effect of Ketamine: An RCT",
            authors=["Chen, Wei", "Nakamura, Hiroshi"],
            year=2020, journal="Lancet Psychiatry", pmid="12345002",
            study_design="RCT",
            population="Adults with treatment-resistant major depression",
            intervention="Ketamine 0.5 mg/kg IV over 40 min",
            comparator="Midazolam active control",
            outcome="MADRS remission at 24h",
            sample_size=80,
            abstract="SMD = -0.65 (95% CI: -1.02, -0.28), p = 0.001",
        ),
    ]

# %% [markdown]
# ## 3. Extract Structured Effect Data

# %%
extractor = StructuredExtractor(
    default_outcome="Depression severity",
    min_confidence_score=0.4,
)

all_effect_records = []
for study in studies:
    text = study.abstract or study.title
    records = extractor.extract(text, study_id=study.study_id)
    study.effect_records.extend(records)
    all_effect_records.extend(records)
    print(f"\n{study.citation_label}: {len(records)} effect record(s)")

# %% [markdown]
# ## 4. Provenance Tracking

# %%
tracker = ProvenanceTracker(
    log_path="/tmp/gnf_provenance_demo.json",
    extractor="notebook_01_demo",
)

for study in studies:
    entry = tracker.register(
        study_id=study.study_id,
        content=study.to_dict(include_effects=True),
        source_url=f"https://pubmed.ncbi.nlm.nih.gov/{study.pmid}/" if study.pmid else None,
    )
    print(f"[{study.study_id}] v{entry.version} - hash: {entry.content_hash[:12]}...")

print("\nProvenance summary:", tracker.summary())

# %% [markdown]
# ## 5. Summary Table

# %%
import pandas as pd

rows = []
for study in studies:
    for rec in study.effect_records:
        rows.append({
            "Study": study.citation_label, "Year": study.year,
            "N (total)": study.sample_size, "Effect type": rec.effect_type,
            "Effect size": round(rec.effect_size, 3),
            "CI lower": round(rec.ci_lower, 3), "CI upper": round(rec.ci_upper, 3),
            "p-value": rec.p_value, "Outcome": rec.outcome_name,
        })

if rows:
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
else:
    print("No structured effect records available.")
