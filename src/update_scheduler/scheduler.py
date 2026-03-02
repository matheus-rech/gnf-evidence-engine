"""Periodic update scheduler for PubMed searches and meta-analysis updates.

Uses APScheduler to run background jobs at configurable intervals:
  1. Search PubMed for new studies matching defined criteria
  2. Compare against provenance log to detect new/changed records
  3. Re-run meta-analysis if new studies found
  4. Send notification if TSA boundary is crossed

Configuration is driven by a YAML/dict job specification.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class SearchSpec:
    """Specification for a recurring PubMed search."""

    name: str
    query: str
    email: str
    mesh_terms: Optional[List[str]] = None
    study_types: Optional[List[str]] = None
    max_results: int = 200


@dataclass
class JobResult:
    """Result of one scheduled update run."""

    job_name: str
    run_at: str
    new_studies: List[str] = field(default_factory=list)
    changed_studies: List[str] = field(default_factory=list)
    analysis_updated: bool = False
    boundary_crossed: bool = False
    error: Optional[str] = None


class UpdateScheduler:
    """Schedule and run periodic evidence base updates."""

    def __init__(
        self,
        search_specs: List[SearchSpec],
        provenance_log_path: str | Path,
        results_dir: str | Path,
        notification_callback: Optional[Callable[[JobResult], None]] = None,
    ) -> None:
        self.search_specs = search_specs
        self.provenance_log_path = Path(provenance_log_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.notification_callback = notification_callback
        self._scheduler = None

    def run_once(self, spec_name: Optional[str] = None) -> List[JobResult]:
        from ..extraction.pubmed_fetcher import PubMedFetcher
        from ..provenance.tracker import ProvenanceTracker

        results = []
        specs = (
            [s for s in self.search_specs if s.name == spec_name]
            if spec_name
            else self.search_specs
        )

        for spec in specs:
            logger.info("Running update job: %s", spec.name)
            result = JobResult(job_name=spec.name, run_at=datetime.utcnow().isoformat())

            try:
                fetcher = PubMedFetcher(email=spec.email)
                tracker = ProvenanceTracker(
                    log_path=self.provenance_log_path,
                    extractor=f"scheduler:{spec.name}",
                )

                studies = fetcher.search_and_fetch(
                    query=spec.query,
                    mesh_terms=spec.mesh_terms,
                    study_types=spec.study_types,
                    max_results=spec.max_results,
                )

                study_ids = [s.study_id for s in studies]
                contents = [s.to_dict(include_effects=False) for s in studies]
                changes = tracker.detect_new_or_changed(study_ids, contents)

                result.new_studies = [sid for sid, status in changes.items() if status == "new"]
                result.changed_studies = [
                    sid for sid, status in changes.items() if status == "changed"
                ]

                for study, content in zip(studies, contents):
                    tracker.register(
                        study_id=study.study_id,
                        content=content,
                        source_url=f"https://pubmed.ncbi.nlm.nih.gov/{study.pmid}/"
                        if study.pmid
                        else None,
                    )

                if result.new_studies or result.changed_studies:
                    result.analysis_updated = True

            except Exception as exc:
                logger.error("Job %s failed: %s", spec.name, exc, exc_info=True)
                result.error = str(exc)

            self._save_result(result)

            if result.boundary_crossed and self.notification_callback:
                try:
                    self.notification_callback(result)
                except Exception as exc:
                    logger.warning("Notification callback failed: %s", exc)

            results.append(result)

        return results

    def start(
        self,
        interval_hours: float = 24.0,
        trigger: str = "interval",
    ) -> None:
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
        except ImportError as exc:
            raise ImportError("apscheduler is required: pip install apscheduler") from exc

        self._scheduler = BackgroundScheduler()
        self._scheduler.add_job(
            func=self.run_once,
            trigger=trigger,
            hours=interval_hours,
            id="gnf_update_all",
            name="GNF Evidence Engine Update",
            replace_existing=True,
        )
        self._scheduler.start()

    def stop(self) -> None:
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=True)

    def _save_result(self, result: JobResult) -> None:
        fname = f"job_{result.job_name}_{result.run_at.replace(':', '-')[:19]}.json"
        with open(self.results_dir / fname, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "job_name": result.job_name,
                    "run_at": result.run_at,
                    "new_studies": result.new_studies,
                    "changed_studies": result.changed_studies,
                    "analysis_updated": result.analysis_updated,
                    "boundary_crossed": result.boundary_crossed,
                    "error": result.error,
                },
                f,
                indent=2,
            )


def main() -> None:
    """CLI entry point for the scheduler."""
    import argparse

    parser = argparse.ArgumentParser(description="GNF Update Scheduler")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument("--run-once", action="store_true", help="Run immediately and exit")
    parser.add_argument("--interval-hours", type=float, default=24.0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    specs = [SearchSpec(**s) for s in cfg.get("search_specs", [])]
    scheduler = UpdateScheduler(
        search_specs=specs,
        provenance_log_path=cfg.get("provenance_log", "provenance.json"),
        results_dir=cfg.get("results_dir", "scheduler_results"),
    )

    if args.run_once:
        results = scheduler.run_once()
        for r in results:
            print(f"[{r.job_name}] new={len(r.new_studies)} changed={len(r.changed_studies)}")
    else:
        scheduler.start(interval_hours=args.interval_hours)
        import time
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            scheduler.stop()
