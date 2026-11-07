"""Row extraction for tables that live in PDF/DOCX text (e.g. a budget analysis report).

Strategy (deterministic segmentation, model reads clean segments):
  1. A cheap model looks at the first page and names (a) a regex that identifies the first line of a data row,
     (b) which fields are document-level vs per row.
  2. Code removes page headers/footers, finds every row start, and attaches the continuation lines of a wrapped
     right-hand text column (found from line geometry, not from header text) to the row above.
  3. The model converts batches of already-segmented rows to JSON; the primary key is taken from the regex match
     (trusted) rather than from the model.
  4. If no usable row regex exists, fall back to chunked extraction over the whole text.
Document-level fields come from a separate small call over the first pages.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..llm import LLMError, chat_json
from ..parse.types import ParsedDoc, render_rows
from .ctx import JobContext
from .fieldspec import FieldCard, card_lines
from .normalize import is_blank
from .pdf_lines import Line, boilerplate_keys, lines_from_doc, strip_boilerplate
from .records import FieldOut, _coerce, extract_record

log = logging.getLogger("text_tables")

BATCH_ROWS = 20
BATCH_CHARS = 14000


@dataclass
class Segment:
    key: str
    anchor: Line
    trailing: str = ""
    cont: list[str] = field(default_factory=list)
    section_hint: str = ""
    context: list[str] = field(default_factory=list)


PLAN_SYSTEM = """\
You analyse a document that contains a table inside text (columns of numbers, a code or name at the start of each row).
You will be given the schema's fields and the start of the document. Return ONLY JSON:
{"row_anchor_regex": "<Python regex that matches the START of the line that begins every data row and ONLY those lines; capture group 1 = the row's primary identifier>",
 "row_fields": ["<keys whose value differs per row>"],
 "doc_fields": ["<keys that are the same for every row: names, years, areas taken from the page header>"]}
Section headings, subtotals, totals and column headings must NOT match the regex. Use ^\\s* at the start. If the document has no
repeating row structure return {"row_anchor_regex": null, ...}.
"""

PLAN_USER = """\
SCHEMA: {name} - {desc}
PRIMARY (row identifier) FIELD: {primary}

FIELDS
{fields}

FIRST LINES OF THE DOCUMENT (page 1) AND A SAMPLE FROM PAGE 2
{text}
"""

ROWS_SYSTEM = """\
You convert pre-segmented table rows into JSON, with perfect fidelity to the printed values.
Rules:
- One output object per input ROW, in the same order. Include every requested key.
- Numbers: plain JSON numbers (no $ or commas); keep the sign as printed; a value printed as N/A or blank is null.
- Percentages: the number exactly as printed (-4.31 for "-4.31%").
- Text fields: copy exactly as printed (collapse runs of spaces). A wrapped commentary column is the text to the right of the numbers on the
  ROW LINE followed by the CONTINUATION LINES, joined with single spaces, in order. If there is none use "" (empty string).
- Columns: read the COLUMN HEADERS and map numbers to columns left to right. Do not confuse similarly named columns.
- Never invent a value. Use the 'section hint' only as a hint for fields that name a section/heading above the row.
Return ONLY JSON: {"rows": [ {<key>: <value>, ...}, ... ]}
"""

ROWS_USER = """\
SCHEMA: {name} - {desc}

FIELDS TO EXTRACT FOR EACH ROW
{fields}

COLUMN HEADERS (as printed)
{headers}

ROWS
{rows}
"""


# ------------------------------------------------------------------ plan

def _head_text(lines: list[Line], n1: int = 55, n2: int = 25) -> str:
    p1 = [ln.text.rstrip() for ln in lines if ln.page == 1][:n1]
    p2 = [ln.text.rstrip() for ln in lines if ln.page == 2][:n2]
    return "\n".join(p1) + ("\n--- page 2 ---\n" + "\n".join(p2) if p2 else "")


def make_plan(doc: ParsedDoc, schema: dict, cards: list[FieldCard], primary: str, lines: list[Line], ctx: JobContext) -> dict:
    user = PLAN_USER.format(name=schema.get("name", ""), desc=schema.get("description", ""), primary=primary,
                            fields=card_lines(cards), text=_head_text(lines))
    obj, _ = chat_json(ctx.settings.model_spec, [{"role": "system", "content": PLAN_SYSTEM}, {"role": "user", "content": user}],
                       tracker=ctx.tracker, max_tokens=1500, deadline=ctx.deadline, tag="table-plan")
    return obj if isinstance(obj, dict) else {}


# ------------------------------------------------------------------ segmentation

def _free_text_x(lines: list[Line], anchor: re.Pattern) -> float | None:
    """x of a wrapped right-hand text column: the commonest left edge among non-row lines in the right 60%."""
    maxx = max((x for ln in lines for x, _ in ln.words), default=0)
    if maxx <= 0:
        return None
    xs = Counter(round(ln.x0 * 2) / 2 for ln in lines if not anchor.match(ln.text) and ln.x0 > 0.4 * maxx)
    if not xs:
        return None
    x, n = xs.most_common(1)[0]
    return x if n >= 3 else None


def segment(lines: list[Line], pattern: re.Pattern, n_pages: int) -> tuple[list[Segment], str, float | None]:
    """-> (segments, column header block, free-text x)"""
    first_anchor = next((i for i, ln in enumerate(lines) if pattern.match(ln.text)), None)
    header_block = ""
    if first_anchor is not None:
        pg = lines[first_anchor].page
        before = [ln for ln in lines[max(0, first_anchor - 14):first_anchor] if ln.page == pg]
        header_block = "\n".join(ln.text.rstrip() for ln in before[-9:])
    clean = strip_boilerplate(lines, boilerplate_keys(lines, n_pages))
    fx = _free_text_x(clean, pattern)
    segs: list[Segment] = []
    cur: Segment | None = None
    ctx_lines: list[str] = []
    section = ""
    for ln in clean:
        m = pattern.match(ln.text)
        if m:
            key = (m.group(1) if m.re.groups else m.group(0)).strip()
            trailing = " ".join(w for x, w in ln.words if fx is not None and x >= fx - 3)
            cur = Segment(key=key, anchor=ln, trailing=trailing, section_hint=section, context=ctx_lines[-4:])
            segs.append(cur)
            ctx_lines = []
        elif cur is not None and fx is not None and ln.x0 >= fx - 3:
            cur.cont.append(ln.stripped)
        else:
            ctx_lines.append(("[bold] " if ln.bold else "") + ln.stripped)
            if ln.bold and not ln.stripped.lower().startswith("total"):
                section = re.sub(r"\s+", " ", ln.stripped)
    return segs, header_block, fx


# ------------------------------------------------------------------ row extraction

def _render_segment(i: int, s: Segment) -> str:
    out = [f"### ROW {i + 1}  key={s.key}  section hint (nearest bold heading above): {s.section_hint or 'none'}"]
    if s.context:
        out.append("lines between previous row and this one: " + " / ".join(s.context))
    out.append("ROW LINE: " + re.sub(r"[ ]{3,}", "   ", s.anchor.text.strip()))
    if s.cont:
        out.append("CONTINUATION LINES:\n" + "\n".join("  " + c for c in s.cont))
    return "\n".join(out)


def _extract_batch(batch: list[Segment], offset: int, schema: dict, row_cards: list[FieldCard], headers: str,
                   ctx: JobContext, depth: int = 0) -> list[dict | None]:
    user = ROWS_USER.format(name=schema.get("name", ""), desc=schema.get("description", ""), fields=card_lines(row_cards),
                            headers=headers or "(not available)", rows="\n\n".join(_render_segment(offset + i, s) for i, s in enumerate(batch)))
    try:
        obj, _ = chat_json(ctx.settings.model_cheap, [{"role": "system", "content": ROWS_SYSTEM}, {"role": "user", "content": user}],
                           tracker=ctx.tracker, max_tokens=min(16000, 700 + 330 * len(batch)), deadline=ctx.deadline,
                           tag=f"table-rows-{offset}")
        rows = obj.get("rows") if isinstance(obj, dict) else obj
        if isinstance(rows, list) and len(rows) == len(batch):
            return [r if isinstance(r, dict) else None for r in rows]
        ctx.warn(f"batch at row {offset}: model returned {len(rows) if isinstance(rows, list) else 'no'} rows for {len(batch)}")
    except LLMError as e:
        ctx.warn(f"batch at row {offset} failed: {e}")
    if len(batch) > 1 and depth < 3:           # split and retry: smaller batches are easier to align
        mid = len(batch) // 2
        return (_extract_batch(batch[:mid], offset, schema, row_cards, headers, ctx, depth + 1) +
                _extract_batch(batch[mid:], offset + mid, schema, row_cards, headers, ctx, depth + 1))
    return [None] * len(batch)


def extract_table_text(doc: ParsedDoc, schema: dict, cards: list[FieldCard], primary: str, ctx: JobContext
                       ) -> list[dict[str, FieldOut]]:
    keys = [c.key for c in cards]
    ctx.stage("PLANNING")
    lines = lines_from_doc(doc)
    plan: dict = {}
    try:
        plan = make_plan(doc, schema, cards, primary, lines, ctx)
    except LLMError as e:
        ctx.warn(f"table plan failed: {e}")
    rx = plan.get("row_anchor_regex")
    segs: list[Segment] = []
    headers, fx = "", None
    if isinstance(rx, str) and rx:
        try:
            pattern = re.compile(rx)
            segs, headers, fx = segment(lines, pattern, len(doc.pages))
        except re.error as e:
            ctx.warn(f"row regex invalid: {e}")
    uniq = len({s.key for s in segs})
    if not segs or uniq < 0.8 * len(segs):
        ctx.warn(f"row segmentation unusable ({len(segs)} segments, {uniq} unique keys): falling back to chunked extraction")
        return extract_table_chunked(doc, schema, cards, primary, lines, ctx)

    row_keys = [k for k in plan.get("row_fields", keys) if k in keys] or keys
    doc_keys = [k for k in keys if k not in row_keys]
    if primary not in row_keys:
        row_keys.insert(0, primary)
        doc_keys = [k for k in doc_keys if k != primary]
    row_cards = [c for c in cards if c.key in row_keys]
    doc_cards = [c for c in cards if c.key in doc_keys]
    by_key = {c.key: c for c in cards}

    ctx.stage("EXTRACTING")
    batches: list[tuple[int, list[Segment]]] = []
    cur: list[Segment] = []
    size = 0
    start = 0
    for i, s in enumerate(segs):
        sz = len(_render_segment(i, s))
        if cur and (len(cur) >= BATCH_ROWS or size + sz > BATCH_CHARS):
            batches.append((start, cur))
            cur, size, start = [], 0, i
        cur.append(s)
        size += sz
    if cur:
        batches.append((start, cur))
    with ThreadPoolExecutor(max_workers=min(4, len(batches))) as ex:
        results = list(ex.map(lambda b: _extract_batch(b[1], b[0], schema, row_cards, headers, ctx), batches))
    raw_rows: list[dict | None] = [r for res in results for r in res]

    docvals: dict[str, FieldOut] = {}
    if doc_cards:
        head = ParsedDoc(kind=doc.kind, path=doc.path, pages=doc.pages[:2], warnings=[])
        docvals = extract_record(head, doc_cards, ctx)

    rows: list[dict[str, FieldOut]] = []
    for seg, raw in zip(segs, raw_rows):
        rec: dict[str, FieldOut] = {}
        for c in row_cards:
            if c.key == primary:
                rec[c.key] = FieldOut(value=seg.key, quote=seg.anchor.stripped[:120], page=seg.anchor.page, confidence=0.98)
                continue
            if raw is None:
                rec[c.key] = FieldOut(value=None, confidence=0.0, flags={"row_failed"})
                continue
            v, ok = _coerce(raw.get(c.key), c)
            if isinstance(v, str):
                v = re.sub(r"\s+", " ", v).strip()
            f = FieldOut(value=v, quote=seg.anchor.stripped[:120] if v is not None else "", page=seg.anchor.page,
                         confidence=0.9 if ok else 0.4)
            if not ok:
                f.flags.add("type_fail")
            rec[c.key] = f
        for c in doc_cards:
            rec[c.key] = docvals[c.key]
        rows.append(rec)
    return rows


# ------------------------------------------------------------------ fallback

CHUNK_LINES = 110
CHUNK_OVERLAP = 8


def sheet_or_page_lines(doc: ParsedDoc, lines: list[Line]) -> list[str]:
    if doc.is_spreadsheet:
        out = []
        for s in doc.sheets:
            out.append(f"## SHEET {s.name}")
            out += render_rows(s, s.rows(), limit=80).splitlines()
        return out
    return [ln.text.rstrip() for ln in lines]


CHUNK_SYSTEM = """\
You extract every data row of a table from a chunk of a document into JSON, with perfect fidelity.
Rules: one object per row that STARTS in the chunk's main part (lines marked MAIN); rows that start in the OVERLAP lines at the
top belong to the previous chunk and must be skipped. Header rows, section headings, and totals that the schema says are not rows
are not rows. Numbers are plain JSON numbers; text is copied exactly; null when absent. Return ONLY JSON: {"rows":[{...}]}
"""


def extract_table_chunked(doc: ParsedDoc, schema: dict, cards: list[FieldCard], primary: str, lines: list[Line],
                          ctx: JobContext) -> list[dict[str, FieldOut]]:
    ctx.stage("EXTRACTING")
    text_lines = sheet_or_page_lines(doc, lines)
    head = "\n".join(text_lines[:40])
    chunks = [(i, text_lines[max(0, i - CHUNK_OVERLAP):i + CHUNK_LINES], min(CHUNK_OVERLAP, i))
              for i in range(0, len(text_lines), CHUNK_LINES)]

    def one(args) -> list[dict]:
        i, chunk, n_over = args
        body = "\n".join(("[OVERLAP] " if j < n_over else "[MAIN] ") + ln for j, ln in enumerate(chunk))
        user = (f"SCHEMA: {schema.get('name', '')} - {schema.get('description', '')}\nPRIMARY FIELD: {primary}\n\nFIELDS\n"
                f"{card_lines(cards)}\n\nDOCUMENT START (for header/context values)\n{head}\n\nCHUNK (lines {i + 1}+)\n{body}")
        try:
            obj, _ = chat_json(ctx.settings.model_cheap, [{"role": "system", "content": CHUNK_SYSTEM}, {"role": "user", "content": user}],
                               tracker=ctx.tracker, max_tokens=12000, deadline=ctx.deadline, tag=f"chunk-{i}")
        except LLMError as e:
            ctx.warn(f"chunk at line {i} failed: {e}")
            return []
        rows = obj.get("rows") if isinstance(obj, dict) else obj
        return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []

    with ThreadPoolExecutor(max_workers=min(4, max(1, len(chunks)))) as ex:
        parts = list(ex.map(one, chunks))
    seen: set[str] = set()
    rows: list[dict[str, FieldOut]] = []
    for part in parts:
        for raw in part:
            pk = raw.get(primary)
            pk = None if is_blank(pk) else str(pk).strip()
            if pk is None or pk in seen:
                continue
            seen.add(pk)
            rec = {}
            for c in cards:
                v, ok = _coerce(raw.get(c.key), c)
                rec[c.key] = FieldOut(value=v, quote="", confidence=0.75 if ok else 0.3, flags=set() if ok else {"type_fail"})
            rows.append(rec)
    return rows
