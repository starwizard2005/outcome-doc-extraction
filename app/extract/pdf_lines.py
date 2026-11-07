"""Position-aware text lines for PDFs (x offset, bold flag), plus repeated-boilerplate removal.

`pdftotext -layout` is good for reading, but table PDFs need two more facts: how far right a line
starts (a wrapped commentary column continues on lines indented to that column) and whether it is bold
(section headers). Both come from pdfplumber word boxes.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..parse.types import ParsedDoc

log = logging.getLogger("pdf_lines")

CHAR_W = 4.4          # points per character cell when re-creating a monospace layout
LINE_TOL = 2.6


@dataclass
class Line:
    page: int
    text: str
    x0: float
    bold: bool = False
    words: list[tuple[float, str]] = field(default_factory=list)

    @property
    def stripped(self) -> str:
        return self.text.strip()


def pdf_lines(path: Path) -> list[Line]:
    import pdfplumber

    out: list[Line] = []
    with pdfplumber.open(path) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(keep_blank_chars=False, use_text_flow=False, extra_attrs=["fontname"])
            words.sort(key=lambda w: (w["top"], w["x0"]))
            rows: list[list[dict]] = []
            for w in words:
                if rows and abs(w["top"] - rows[-1][0]["top"]) <= LINE_TOL:
                    rows[-1].append(w)
                else:
                    rows.append([w])
            for row in rows:
                row.sort(key=lambda w: w["x0"])
                s = ""
                for w in row:
                    col = int(w["x0"] / CHAR_W)
                    s += (" " * (col - len(s)) if col > len(s) else " ") + w["text"]
                nbold = sum(len(w["text"]) for w in row if "bold" in (w.get("fontname") or "").lower())
                ntot = sum(len(w["text"]) for w in row) or 1
                out.append(Line(page=pno, text=s, x0=row[0]["x0"], bold=nbold / ntot >= 0.6,
                                words=[(w["x0"], w["text"]) for w in row]))
    return out


def lines_from_doc(doc: ParsedDoc) -> list[Line]:
    """Position-aware lines for PDFs with a text layer; plain text lines otherwise."""
    if doc.kind == "pdf" and not doc.scanned_pages:
        try:
            ls = pdf_lines(doc.path)
            if ls:
                return ls
        except Exception as e:  # noqa: BLE001
            log.warning("pdfplumber line extraction failed, using text lines: %s", e)
    out: list[Line] = []
    for p in doc.pages:
        for t in p.text.splitlines():
            if t.strip():
                out.append(Line(page=p.number, text=t, x0=(len(t) - len(t.lstrip())) * CHAR_W))
    return out


_DIG = re.compile(r"\d+")


def boilerplate_keys(lines: list[Line], n_pages: int) -> set[str]:
    """Normalised lines that repeat on at least half the pages (page headers/footers, column headings)."""
    if n_pages < 2:
        return set()
    pages_by_key: dict[str, set[int]] = {}
    for ln in lines:
        k = _DIG.sub("#", ln.stripped)
        if k:
            pages_by_key.setdefault(k, set()).add(ln.page)
    need = max(2, int(0.5 * n_pages + 0.5))
    return {k for k, ps in pages_by_key.items() if len(ps) >= need}


def strip_boilerplate(lines: list[Line], keys: set[str]) -> list[Line]:
    return [ln for ln in lines if _DIG.sub("#", ln.stripped) not in keys]
