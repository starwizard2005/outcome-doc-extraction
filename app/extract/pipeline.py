"""End-to-end extraction: file + field list -> contract-shaped result.

    parse -> choose path by schema shape (a `primary` field means "one entry per table row")
          -> table path : spreadsheets -> LLM-written spec executed deterministically
                          PDF/DOCX tables -> deterministic row segmentation + LLM row conversion
          -> record path: one cheap pass, deterministic checks, targeted strong-model re-read, page-image check for scans
          -> assemble entries {value, citation, confidence, page_numbers | source_locations} + metadata
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

from ..llm import BudgetExceeded, DeadlineExceeded, LLMError
from ..parse import load_document
from ..parse.types import ParsedDoc
from .ctx import JobContext
from .fieldspec import FieldCard, build_cards
from .records import FieldOut, extract_record
from .sheet_exec import Cell
from .spec_gen import SpecFailed, generate_spec
from .text_tables import extract_table_chunked, extract_table_text, sheet_or_page_lines
from .pdf_lines import lines_from_doc
from .vision import fill_missing, verify_scanned

log = logging.getLogger("pipeline")


def _clean_value(v: Any) -> Any:
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, 6)
    if isinstance(v, list):
        return [_clean_value(x) for x in v]
    if hasattr(v, "isoformat"):
        return v.isoformat()[:10]
    if isinstance(v, str):
        return v.strip() if v.strip() != "" else None
    return v


def _entry_from_fieldouts(rec: dict[str, FieldOut], cards: list[FieldCard], pdf: bool) -> dict[str, dict]:
    entry: dict[str, dict] = {}
    for c in cards:
        f = rec.get(c.key) or FieldOut()
        v = _clean_value(f.value)
        empty = v is None or v == []
        cite = ""
        if not empty:
            q = (f.quote or "").strip().replace("\n", " ")
            cite = (f"[Page {f.page}] {q}" if f.page else q)[:500] if pdf else q[:500]
        out: dict[str, Any] = {"value": v, "citation": "" if empty else cite,
                               "confidence": round(max(0.0, min(1.0, f.confidence if not empty else max(f.confidence, 0.5))), 3)}
        if f.page and not empty and pdf:
            out["page_numbers"] = [int(f.page)]
        entry[c.key] = out
    return entry


def _entry_from_cells(rec: dict[str, Cell], cards: list[FieldCard]) -> dict[str, dict]:
    entry: dict[str, dict] = {}
    for c in cards:
        cell = rec.get(c.key) or Cell(None)
        v = _clean_value(cell.value)
        out: dict[str, Any] = {"value": v, "citation": "", "confidence": 0.9}
        if v is not None and cell.refs:
            shown = v if not isinstance(v, str) else v[:80]
            out["citation"] = f"[{cell.refs[0]}] {shown}"
            out["source_locations"] = cell.refs[:5]
            out["confidence"] = 0.97
        elif v is not None:
            out["citation"] = ""
            out["confidence"] = 0.9
        entry[c.key] = out
    return entry


def _table_from_sheets(doc: ParsedDoc, schema: dict, cards: list[FieldCard], primary: str, ctx: JobContext) -> list[dict]:
    try:
        ctx.stage("PLANNING")
        spec, res = generate_spec(doc, schema, cards, primary, ctx)
        ctx.stage("ASSEMBLING")
        log.info("spec ok", extra={"job_id": ctx.job_id, "rows": len(res.rows)})
        return [_entry_from_cells(r, cards) for r in res.rows]
    except (SpecFailed, LLMError) as e:
        if isinstance(e, (BudgetExceeded, DeadlineExceeded)):
            raise
        ctx.degrade(f"spec generation failed ({e}); fell back to chunked LLM extraction")
        lines = lines_from_doc(doc) if not doc.is_spreadsheet else []
        rows = extract_table_chunked(doc, schema, cards, primary, lines, ctx)
        return [_entry_from_fieldouts(r, cards, pdf=False) for r in rows]


def run_extraction(path: Path, schema: dict, ctx: JobContext) -> tuple[dict, str]:
    """Returns (result, status) where status is SUCCESS or PARTIAL. Raises only on unrecoverable failure."""
    t0 = time.time()
    ctx.stage("PARSING")
    doc = load_document(path)
    for w in doc.warnings:
        ctx.warn("parse: " + w)
    cards = build_cards(schema)
    if not cards:
        raise ValueError("schema has no fields")
    primary = next((c.key for c in cards if c.primary), None)
    pdf = doc.kind in ("pdf", "docx")
    mode = "table" if primary else "record"

    if mode == "table":
        if doc.is_spreadsheet:
            entries = _table_from_sheets(doc, schema, cards, primary, ctx)       # type: ignore[arg-type]
        else:
            rows = extract_table_text(doc, schema, cards, primary, ctx)          # type: ignore[arg-type]
            bad = sum(1 for r in rows if any("row_failed" in f.flags for f in r.values()))
            if bad:
                ctx.degrade(f"{bad} row(s) could not be extracted")
            entries = [_entry_from_fieldouts(r, cards, pdf=pdf) for r in rows]
        if not entries:
            raise ValueError("no rows could be extracted from the document")
    else:
        res = extract_record(doc, cards, ctx)
        try:
            verify_scanned(doc, cards, res, ctx)
            fill_missing(doc, cards, res, ctx)
        except LLMError as e:
            if isinstance(e, (BudgetExceeded, DeadlineExceeded)):
                raise
            ctx.warn(f"vision verification skipped: {e}")
        entries = [_entry_from_fieldouts(res, cards, pdf=pdf)]

    meta = ctx.tracker.metadata()
    meta.update({"latency_ms": int((time.time() - t0) * 1000), "mode": mode, "document_kind": doc.kind,
                 "pages": len(doc.pages) or None, "scanned_pages": len(doc.scanned_pages),
                 "warnings": ctx.warnings[:40], "degraded": ctx.degraded})
    meta = {k: v for k, v in meta.items() if v is not None}
    status = "PARTIAL" if ctx.degraded else "SUCCESS"
    return {"entries": entries, "metadata": meta}, status
