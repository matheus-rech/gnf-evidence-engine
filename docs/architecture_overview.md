# Architecture Overview

## System Context

The GNF Evidence Engine is a continuous meta-analysis platform for translational
neuroscience research. It maintains a **living systematic review** that
automatically updates when new studies appear in PubMed.

---

## Component Map

```
+-----------------------------------------------------------------------------+
|                           GNF Evidence Engine                               |
|                                                                             |
|  +-------------------------------------------------------------------------+
|  |                          Data Ingestion Layer                           |
|  |                                                                         |
|  |  PubMedFetcher          PDFParser           StructuredExtractor         |
|  |  E-utilities API        fitz/PyMuPDF         Regex patterns             |
|  |  Rate limiting          Table detection       Effect size extraction     |
|  |  Retry logic            Section detection     CI / p-value parsing       |
|  |  XML -> StudyRecord     Tables -> PDFTable    Returns EffectRecord[]     |
|  +-------------------------------------------------------------------------+
|                                    |
|                                    v
|  +-------------------------------------------------------------------------+
|  |                          Structured Schema                              |
|  |                                                                         |
|  |   StudyRecord (dataclass)              EffectRecord (dataclass)         |
|  |   study_id, title, authors             effect_type (SMD/OR/RR/HR)       |
|  |   year, journal, design                effect_size, ci_lower/upper      |
|  |   population, intervention             se, p_value, n_treatment         |
|  |   sample_size, effect_records          n_control, outcome_name          |
|  |   risk_of_bias, provenance_hash                                         |
|  +-------------------------------------------------------------------------+
|                                    |
|                    +---------------+---------------+
|                    v                               v
|  +---------------------------+   +-------------------------------------+
|  |   Meta-Analysis Layer     |   |    Trial Sequential Analysis        |
|  |                           |   |                                      |
|  |  FixedEffectsModel        |   |  TrialSequentialAnalysis             |
|  |  RandomEffectsModel (DL)  |   |  +- SpendingFunctions               |
|  |  HeterogeneityAnalysis    |   |  |  OBrienFleming, LanDeMets        |
|  |  ForestPlot               |   |  |  Pocock, HaybittlePeto           |
|  |  FunnelPlot + Egger's     |   |  +- RequiredInformationSize         |
|  |  Trim-and-fill            |   |  +- TSAPlot                         |
|  |                           |   |                                      |
|  |  Returns MetaAnalysisResult   |  Returns TSAResult                   |
|  +---------------------------+   +-------------------------------------+
|                    |                               |
|                    +---------------+---------------+
|                                    v
|  +-------------------------------------------------------------------------+
|  |                         Certainty Assessment                            |
|  |                                                                         |
|  |   GRADEAssessment                                                       |
|  |   Auto-rates from I2, TSA conclusion, effect size                       |
|  |   Domains: risk of bias, inconsistency, indirectness,                   |
|  |           imprecision, publication bias                                 |
|  |   Output: VERY LOW | LOW | MODERATE | HIGH                              |
|  +-------------------------------------------------------------------------+
|                                    |
|                    +---------------+---------------+
|                    v                               v
|  +---------------------------+   +-------------------------------------+
|  |    Provenance Layer       |   |       Update Scheduler              |
|  |                           |   |                                      |
|  |  ProvenanceTracker        |   |  UpdateScheduler                     |
|  |  SHA-256 content hashing  |   |  APScheduler background jobs         |
|  |  Change detection         |   |  Periodic PubMed searches            |
|  |                           |   |  New study detection                 |
|  |  EvidenceVersioning       |   |  Auto re-analysis                    |
|  |  Semantic versioning      |   |  Notification callbacks              |
|  |  Snapshot diffs           |   |                                      |
|  |                           |   +-------------------------------------+
|  |  AuditLog (NDJSON)        |
|  |  Append-only              |
|  |  Integrity checksums      |
|  |  CSV export               |
|  +---------------------------+
|                                    |
|                                    v
|  +-------------------------------------------------------------------------+
|  |                         Dashboard (Dash/Plotly)                         |
|  |                                                                         |
|  |  Live forest plot, TSA monitoring plot, Funnel plot, GRADE table        |
|  |  Study table, KPI cards, Auto-refresh                                   |
|  +-------------------------------------------------------------------------+
+-----------------------------------------------------------------------------+
```

---

## Data Flow

```
PubMed / PDF
     |
     v
[Extraction]
 PubMedFetcher / PDFParser
 StructuredExtractor
     |
     v
[Schema]
 StudyRecord[]
 EffectRecord[]
     |
     +----------------------------------+
     v                                  v
[Meta-Analysis]                    [Provenance]
 Fixed / Random Effects              SHA-256 hash
 Heterogeneity                       Version history
 Forest / Funnel plots               Audit log
     |
     v
[TSA]
 Spending boundaries
 Information fractions
 Conclusion: FIRM / INSUFFICIENT / FUTILE
     |
     v
[GRADE]
 Certainty: HIGH / MODERATE / LOW / VERY LOW
     |
     v
[Dashboard]
 Interactive monitoring
```

---

## Key Design Decisions

### Python + R Bridge
Statistical methods in meta-analysis have mature R implementations
(metafor, meta). We use `rpy2` to call R's `rma()` function for REML tau2
estimation, with pure-Python DL fallback when R is unavailable.

### Dataclass-First Design
All intermediate and result objects are Python dataclasses. This gives:
- Type safety and IDE support
- Easy serialisation (`.to_dict()` / `.from_dict()`)
- Immutable-by-convention semantics for results

### Provenance by Hash
Every study record is content-addressed via SHA-256. This means:
- Re-extracting the same paper yields the same hash (no spurious updates)
- Any modification (corrected data, added effect record) is immediately detected
- Provenance is auditable without needing a database

### Alpha-Spending Architecture
Spending functions are decoupled from the TSA engine via the `SpendingFunction`
abstract base class. Adding a new boundary type requires only implementing
`spent_alpha(t)`. The TSA engine queries spending functions at each step.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Core language | Python 3.11+ |
| Statistics | numpy, scipy |
| R bridge | rpy2 + metafor |
| Visualization | matplotlib, plotly |
| Dashboard | Dash + Bootstrap |
| PDF parsing | PyMuPDF (fitz) |
| PubMed API | requests + lxml |
| Scheduling | APScheduler |
| Testing | pytest + pytest-cov |
| Containerization | Docker |
