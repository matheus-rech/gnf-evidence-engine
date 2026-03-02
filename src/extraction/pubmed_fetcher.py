"""PubMed E-utilities API integration for study fetching.

Supports searching by free text, MeSH terms, date range, and study type filters.
Implements rate limiting and retry logic as required by NCBI guidelines.
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator, List, Optional
from urllib.parse import urlencode

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..structured_schema.study_record import StudyRecord
from ..structured_schema.effect_record import EffectRecord

logger = logging.getLogger(__name__)

# NCBI E-utilities base URL
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# NCBI rate limits: 3 requests/second without API key, 10/second with key
DEFAULT_RATE_LIMIT = 3  # requests per second


@dataclass
class SearchResult:
    """Raw search results from NCBI ESearch.

    Attributes:
        pmids: List of PubMed IDs matching the search.
        total_count: Total number of matching records.
        query_translation: How NCBI interpreted the query.
        search_params: Original search parameters used.
    """

    pmids: List[str]
    total_count: int
    query_translation: str
    search_params: dict = field(default_factory=dict)


class PubMedFetcher:
    """Fetch and parse records from the PubMed E-utilities API.

    Implements NCBI's recommended practices: rate limiting, retries on
    transient failures, and batch fetching via EFetch.

    Args:
        email: Researcher email (required by NCBI).
        api_key: Optional NCBI API key (allows 10 req/s vs 3 req/s).
        tool: Tool name sent to NCBI for usage tracking.
        rate_limit: Maximum requests per second.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        email: str,
        api_key: Optional[str] = None,
        tool: str = "gnf-evidence-engine",
        rate_limit: Optional[int] = None,
        timeout: int = 30,
    ) -> None:
        self.email = email
        self.api_key = api_key
        self.tool = tool
        self.timeout = timeout
        self._min_interval = 1.0 / (rate_limit or (10 if api_key else DEFAULT_RATE_LIMIT))
        self._last_request_time: float = 0.0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": f"{tool}/1.0 (mailto:{email})"})

    def _throttle(self) -> None:
        """Sleep if needed to respect rate limits."""
        elapsed = time.monotonic() - self._last_request_time
        wait = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.monotonic()

    def _base_params(self) -> dict:
        """Return common E-utilities parameters."""
        params: dict = {"tool": self.tool, "email": self.email}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    @retry(
        retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
    )
    def _get(self, endpoint: str, params: dict) -> requests.Response:
        """Execute a throttled GET request with automatic retries."""
        self._throttle()
        url = f"{EUTILS_BASE}/{endpoint}"
        response = self._session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response

    def esearch(
        self,
        query: str,
        mesh_terms: Optional[List[str]] = None,
        date_range: Optional[tuple] = None,
        study_types: Optional[List[str]] = None,
        max_results: int = 500,
        db: str = "pubmed",
    ) -> SearchResult:
        """Search PubMed and return matching PMIDs."""
        terms: List[str] = [query] if query else []
        if mesh_terms:
            terms += [f'"{t}"[MeSH Terms]' for t in mesh_terms]
        if study_types:
            type_q = " OR ".join(f'"{st}"[Publication Type]' for st in study_types)
            terms.append(f"({type_q})")
        combined_query = " AND ".join(f"({t})" for t in terms) if terms else "*"
        params = {
            **self._base_params(),
            "db": db,
            "term": combined_query,
            "retmax": str(max_results),
            "usehistory": "y",
            "retmode": "xml",
        }
        if date_range:
            params["datetype"] = "pdat"
            params["mindate"] = date_range[0]
            params["maxdate"] = date_range[1]
        logger.info("ESearch query: %s", combined_query)
        resp = self._get("esearch.fcgi", params)
        root = ET.fromstring(resp.content)
        pmids = [id_elem.text for id_elem in root.findall(".//Id") if id_elem.text]
        total = int(root.findtext(".//Count") or 0)
        translation = root.findtext(".//QueryTranslation") or ""
        logger.info("Found %d total records (fetching up to %d)", total, max_results)
        return SearchResult(pmids=pmids, total_count=total, query_translation=translation, search_params=params)

    def efetch(self, pmids: List[str], batch_size: int = 100) -> Iterator[StudyRecord]:
        """Fetch and parse full study records for a list of PMIDs."""
        for start in range(0, len(pmids), batch_size):
            batch = pmids[start : start + batch_size]
            params = {**self._base_params(), "db": "pubmed", "id": ",".join(batch), "retmode": "xml", "rettype": "abstract"}
            logger.info("EFetch batch %d/%d (%d records)", start // batch_size + 1, (len(pmids) + batch_size - 1) // batch_size, len(batch))
            resp = self._get("efetch.fcgi", params)
            yield from self._parse_pubmed_xml(resp.content)

    def search_and_fetch(self, query: str, mesh_terms=None, date_range=None, study_types=None, max_results: int = 200) -> List[StudyRecord]:
        """Combined search + fetch pipeline."""
        search_result = self.esearch(query=query, mesh_terms=mesh_terms, date_range=date_range, study_types=study_types, max_results=max_results)
        if not search_result.pmids:
            logger.warning("No results found for query: %s", query)
            return []
        records = list(self.efetch(search_result.pmids))
        logger.info("Successfully parsed %d records", len(records))
        return records

    def _parse_pubmed_xml(self, xml_bytes: bytes) -> Iterator[StudyRecord]:
        """Parse PubMed XML response into StudyRecord objects."""
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as exc:
            logger.error("Failed to parse XML response: %s", exc)
            return
        for article in root.findall(".//PubmedArticle"):
            try:
                record = self._parse_article(article)
                if record is not None:
                    yield record
            except Exception as exc:
                pmid = article.findtext(".//PMID") or "unknown"
                logger.warning("Failed to parse article PMID=%s: %s", pmid, exc)

    def _parse_article(self, article: ET.Element) -> Optional[StudyRecord]:
        """Parse a single PubmedArticle XML element."""
        pmid = article.findtext(".//PMID")
        if not pmid:
            return None
        title = (article.findtext(".//ArticleTitle") or "").strip()
        authors: List[str] = []
        for author in article.findall(".//Author"):
            last = author.findtext("LastName") or ""
            fore = author.findtext("ForeName") or ""
            initials = author.findtext("Initials") or ""
            if last:
                authors.append(f"{last}, {fore or initials}".strip(", "))
            else:
                collective = author.findtext("CollectiveName")
                if collective:
                    authors.append(collective)
        pub_date = article.find(".//PubDate")
        year_text = (pub_date.findtext("Year") if pub_date is not None else None) or ""
        try:
            year = int(year_text[:4]) if year_text else datetime.utcnow().year
        except ValueError:
            year = datetime.utcnow().year
        journal = article.findtext(".//Journal/Title") or article.findtext(".//ISOAbbreviation") or "Unknown Journal"
        abstract_texts = article.findall(".//AbstractText")
        abstract = " ".join((elem.get("Label", "") + " " + (elem.text or "")).strip() for elem in abstract_texts).strip()
        doi: Optional[str] = None
        for id_elem in article.findall(".//ArticleId"):
            if id_elem.get("IdType") == "doi":
                doi = id_elem.text
                break
        pub_types = [pt.text or "" for pt in article.findall(".//PublicationType")]
        study_design = self._infer_study_design(pub_types)
        try:
            record = StudyRecord(study_id=f"pmid_{pmid}", title=title or f"Untitled (PMID {pmid})", authors=authors or ["Unknown"], year=year, journal=journal, pmid=pmid, doi=doi, study_design=study_design, population="", intervention="", comparator="", outcome="", sample_size=1, abstract=abstract or None)
        except Exception as exc:
            logger.warning("Could not build StudyRecord for PMID %s: %s", pmid, exc)
            return None
        return record

    @staticmethod
    def _infer_study_design(pub_types: List[str]) -> str:
        """Map publication type strings to internal study design labels."""
        lower_types = [t.lower() for t in pub_types]
        if any("randomized" in t or "rct" in t for t in lower_types):
            return "RCT"
        if any("cohort" in t or "longitudinal" in t for t in lower_types):
            return "cohort"
        if any("case-control" in t for t in lower_types):
            return "case-control"
        if any("cross-sectional" in t for t in lower_types):
            return "cross-sectional"
        if any("meta-analysis" in t for t in lower_types):
            return "meta-analysis"
        if any("systematic review" in t for t in lower_types):
            return "systematic-review"
        return "other"
