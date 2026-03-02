"""PDF text and table extraction for GNF Evidence Engine.

Uses PyMuPDF (fitz) for robust PDF parsing. Detects sections
(Methods, Results, Discussion) and extracts tables via heuristics.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SECTION_PATTERNS: Dict[str, re.Pattern] = {
    "abstract": re.compile(r"\babstract\b", re.IGNORECASE),
    "introduction": re.compile(r"\bintroduction\b", re.IGNORECASE),
    "methods": re.compile(r"\b(methods?|materials?\s+and\s+methods?|study\s+design|participants?)\b", re.IGNORECASE),
    "results": re.compile(r"\bresults?\b", re.IGNORECASE),
    "discussion": re.compile(r"\bdiscussion\b", re.IGNORECASE),
    "conclusion": re.compile(r"\bconclusions?\b", re.IGNORECASE),
    "references": re.compile(r"\breferences?\b|\bbibliography\b", re.IGNORECASE),
}


@dataclass
class PDFTable:
    """A table extracted from a PDF page."""
    page_number: int
    rows: List[List[str]]
    caption: Optional[str] = None

    @property
    def n_rows(self) -> int:
        return max(0, len(self.rows) - 1)

    @property
    def n_cols(self) -> int:
        if not self.rows:
            return 0
        return max(len(r) for r in self.rows)

    def to_dicts(self) -> List[Dict[str, str]]:
        if len(self.rows) < 2:
            return []
        headers = self.rows[0]
        return [{headers[i]: row[i] if i < len(row) else "" for i in range(len(headers))} for row in self.rows[1:]]


@dataclass
class ParsedPDF:
    """Structured content extracted from a PDF."""
    file_path: str
    full_text: str
    sections: Dict[str, str] = field(default_factory=dict)
    tables: List[PDFTable] = field(default_factory=list)
    page_count: int = 0
    metadata: Dict[str, str] = field(default_factory=dict)


class PDFParser:
    """Extract structured content from PDF publications."""

    def __init__(self, extract_tables: bool = True, min_table_cols: int = 2, min_table_rows: int = 2) -> None:
        try:
            import fitz  # noqa: F401
        except ImportError as exc:
            raise ImportError("PyMuPDF (fitz) is required: pip install pymupdf") from exc
        self.extract_tables = extract_tables
        self.min_table_cols = min_table_cols
        self.min_table_rows = min_table_rows

    def parse(self, file_path) -> ParsedPDF:
        """Parse a PDF file into structured content."""
        import fitz
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        logger.info("Parsing PDF: %s", path.name)
        try:
            doc = fitz.open(str(path))
        except Exception as exc:
            raise RuntimeError(f"Failed to open PDF {path}: {exc}") from exc
        try:
            metadata = {k: str(v) for k, v in doc.metadata.items() if v}
            pages_text: List[str] = []
            tables: List[PDFTable] = []
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                pages_text.append(text)
                if self.extract_tables:
                    page_tables = self._extract_tables_from_page(page, page_num)
                    tables.extend(page_tables)
            full_text = "\n".join(pages_text)
            sections = self._detect_sections(full_text)
            return ParsedPDF(file_path=str(path), full_text=full_text, sections=sections, tables=tables, page_count=doc.page_count, metadata=metadata)
        finally:
            doc.close()

    def _detect_sections(self, text: str) -> Dict[str, str]:
        """Detect standard section headers and split text accordingly."""
        lines = text.split("\n")
        sections: Dict[str, str] = {}
        current_section: Optional[str] = "preamble"
        section_lines: List[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                section_lines.append(line)
                continue
            matched_section = self._match_section_header(stripped)
            if matched_section and matched_section != current_section:
                if current_section and section_lines:
                    sections[current_section] = "\n".join(section_lines).strip()
                current_section = matched_section
                section_lines = []
            else:
                section_lines.append(line)
        if current_section and section_lines:
            sections[current_section] = "\n".join(section_lines).strip()
        return sections

    @staticmethod
    def _match_section_header(line: str) -> Optional[str]:
        if len(line) > 80:
            return None
        if line.count(".") > 1:
            return None
        for name, pattern in SECTION_PATTERNS.items():
            if pattern.match(line):
                return name
        return None

    def _extract_tables_from_page(self, page, page_number: int) -> List[PDFTable]:
        tables: List[PDFTable] = []
        try:
            found_tables = page.find_tables()
            for tbl in found_tables.tables:
                rows = tbl.extract()
                if not rows:
                    continue
                cleaned_rows = [[str(cell).strip() if cell else "" for cell in row] for row in rows]
                if len(cleaned_rows) >= self.min_table_rows and max(len(r) for r in cleaned_rows) >= self.min_table_cols:
                    tables.append(PDFTable(page_number=page_number, rows=cleaned_rows))
        except AttributeError:
            tables.extend(self._heuristic_table_extraction(page, page_number))
        return tables

    def _heuristic_table_extraction(self, page, page_number: int) -> List[PDFTable]:
        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[6] == 0]
        if not text_blocks:
            return []
        rows: Dict[int, List] = {}
        for block in text_blocks:
            y_key = round(block[1] / 10) * 10
            rows.setdefault(y_key, []).append(block)
        table_rows: List[List[str]] = []
        for y_key in sorted(rows.keys()):
            row_blocks = sorted(rows[y_key], key=lambda b: b[0])
            if len(row_blocks) >= self.min_table_cols:
                table_rows.append([b[4].strip().replace("\n", " ") for b in row_blocks])
        if len(table_rows) >= self.min_table_rows:
            return [PDFTable(page_number=page_number, rows=table_rows)]
        return []

    def extract_section_text(self, parsed: ParsedPDF, section: str) -> str:
        return parsed.sections.get(section.lower(), "")
