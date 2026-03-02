# GNF Evidence Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-pytest-green)](https://pytest.org)

A continuous meta-analysis and evidence synthesis engine for translational neuroscience research. Built by the Global Neuro Foundry (GNF) to support living systematic reviews with full provenance tracking.

## Features

- **Automated PubMed fetching** — search and retrieve studies via E-utilities API
- **PDF text extraction** — parse PDFs with PyMuPDF for offline or supplemental studies
- **Structured data extraction** — regex + heuristic extraction of effect sizes, sample sizes, and study metadata
- **Fixed-effects meta-analysis** — inverse-variance weighted pooling
- **Random-effects meta-analysis** — DerSimonian-Laird and REML estimators
- **Heterogeneity statistics** — Cochran's Q, I², τ², H² with confidence intervals
- **Trial Sequential Analysis (TSA)** — group sequential boundaries for cumulative meta-analysis
- **Forest plots and funnel plots** — publication-quality matplotlib figures
- **TSA monitoring plots** — cumulative Z-score with alpha/beta spending boundaries
- **GRADE certainty assessment** — automated evidence grading
- **Provenance tracking** — SHA-256 content hashing for reproducibility
- **Evidence versioning** — semantic versioning of meta-analysis snapshots
- **Audit log** — append-only log of all modifications
- **Update scheduler** — APScheduler-based periodic PubMed re-searches
- **Interactive dashboard** — Dash/Plotly web UI for real-time evidence monitoring

## Quick Start

```bash
# Clone the repository
git clone https://github.com/matheus-rech/gnf-evidence-engine.git
cd gnf-evidence-engine

# Install dependencies
pip install -e .

# Run tests
pytest tests/ -v

# Launch dashboard
python dashboard/app.py
```

## Project Structure

```
gnf-evidence-engine/
├── src/
│   ├── structured_schema/     # StudyRecord and EffectRecord dataclasses
│   ├── extraction/            # PubMed fetcher, PDF parser, structured extractor
│   ├── meta_analysis/         # Fixed/random effects, heterogeneity, plots
│   ├── tsa/                   # Trial Sequential Analysis engine
│   ├── provenance/            # Content hashing, versioning, audit log
│   ├── update_scheduler/      # APScheduler-based update automation
│   └── certainty/             # GRADE evidence certainty assessment
├── dashboard/                 # Dash web application
├── schemas/                   # JSON Schema definitions
├── notebooks/                 # Jupyter-style demo notebooks
├── tests/                     # pytest test suite
├── docs/                      # Architecture and methods documentation
└── docker/                    # Container configuration
```

## Architecture

See [docs/architecture_overview.md](docs/architecture_overview.md) for a full description of the system design.

## Meta-Analysis Methods

See [docs/meta_analysis_methods.md](docs/meta_analysis_methods.md) for details on the statistical methods implemented.

## Trial Sequential Analysis

See [docs/tsa_explainer.md](docs/tsa_explainer.md) for an explanation of TSA methodology.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linter
ruff check src/

# Type check
mypy src/

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html
```

## Docker

```bash
docker build -t gnf-evidence-engine -f docker/Dockerfile .
docker run -p 8050:8050 gnf-evidence-engine
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

Global Neuro Foundry — science@globalneuro.org
