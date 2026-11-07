"""Deterministic executor for spreadsheet extraction specs.

The LLM never emits rows for a spreadsheet. It reads a *skeleton* of the sheet (header rows,
row-shape patterns, examples) and writes a small declarative spec; this module applies that spec
to every row. A 3,000-row rent roll therefore costs one short LLM call, takes milliseconds to
execute, and cannot drift or hallucinate a row.

Spec (JSON):
{
  "sheet": "Report1",                       # sheet name (default: first non-hidden sheet)
  "layout": "flat" | "blocks",              # flat: one row = one record. blocks: an anchor row followed
                                            #       by detail rows (e.g. a unit line + its charge lines)
  "first_row": 8, "last_row": null,
  "record": [TEST, ...],                    # ALL must hold for a row to be a record / block anchor
  "skip":   [TEST, ...],                    # ANY holds -> row ignored
  "stop":   [TEST, ...],                    # ANY holds -> stop reading the sheet here
  "duplicates": "first" | "last" | "keep",  # policy when the primary key repeats
  "block": {"label_col": "H"},              # blocks only: column holding each detail line's label
  "fields": {"<key>": SOURCE, ...}
}
TEST   := {"col":"A","op":"nonempty|empty|numeric|text|regex|not_regex|eq|in","value":...}
        | {"cols":["C","D"],"op":"any_numeric|none_numeric"} | {"any":[TEST]} | {"all":[TEST]} | {"not":TEST}
SOURCE := {"col":"B"} | {"cell":"A3"[, "sheet":"S"]} | {"const":X}
          [+ "regex":"..", "group":1]                       value picked from the cell text
        | {"above":{"tests":[TEST], "col":"B"}}             nearest earlier row satisfying tests (section headers)
        | {"count_numeric_nonzero":["C","D",...]} | {"sum_cols":["C","D"]}
        | {"detail_sum":{"label_in":[..]|"label_not_in":[..], "label_col":"H", "value_col":"I"}}
        | {"detail_value":{"label_in":[..], "label_col":"G", "value_col":"I"}}
        | {"map":{"from":SOURCE, "table":{"a":"b"}, "default":null}}
        | {"rules":[{"when":[TEST], "value":X | "source":SOURCE}, ...], "default":X}   first matching rule wins
     any SOURCE may add:  "as": string|number|integer|date|month_start|month_end|year_month|boolean,
                          "default": value used when the result is blank, "scope": "doc"|"row"
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..parse.types import ParsedDoc, Sheet, col_index, col_letter, parse_ref
from .normalize import coerce, is_blank, to_number


class SpecError(ValueError):
    """The spec is malformed or cannot be applied. Message is fed back to the model for repair."""


@dataclass
class Cell:
    value: Any
    refs: list[str] = field(default_factory=list)


# ----------------------------------------------------------------- tests

def _val(sheet: Sheet, row: int, col: str) -> Any:
    return sheet.get(row, col_index(col))


def _is_num(v: Any) -> bool:
    return to_number(v) is not None and not isinstance(v, str) or (isinstance(v, str) and to_number(v) is not None)


def run_test(sheet: Sheet, row: int, t: dict) -> bool:
    if "any" in t:
        return any(run_test(sheet, row, x) for x in t["any"])
    if "all" in t:
        return all(run_test(sheet, row, x) for x in t["all"])
    if "not" in t:
        return not run_test(sheet, row, t["not"])
    op = t.get("op")
    if op in ("any_numeric", "none_numeric"):
        cols = t.get("cols") or ([t["col"]] if "col" in t else [])
        has = any(_is_num(_val(sheet, row, c)) for c in cols)
        return has if op == "any_numeric" else not has
    if "col" not in t or op is None:
        raise SpecError(f"bad test {t!r}: needs 'col' and 'op'")
    v = _val(sheet, row, t["col"])
    s = "" if v is None else str(v).strip()
    if op == "nonempty":
        return s != ""
    if op == "empty":
        return s == ""
    if op == "numeric":
        return s != "" and _is_num(v)
    if op == "text":
        return s != "" and not _is_num(v)
    if op == "regex":
        return s != "" and re.search(t["value"], s) is not None
    if op == "not_regex":
        return s == "" or re.search(t["value"], s) is None
    if op == "eq":
        return s.lower() == str(t["value"]).strip().lower()
    if op == "in":
        return s.lower() in {str(x).strip().lower() for x in t["value"]}
    raise SpecError(f"unknown test op {op!r}")


def _all(sheet: Sheet, row: int, tests: list[dict] | None) -> bool:
    return all(run_test(sheet, row, t) for t in (tests or []))


def _any(sheet: Sheet, row: int, tests: list[dict] | None) -> bool:
    return any(run_test(sheet, row, t) for t in (tests or []))


# ----------------------------------------------------------------- sources

@dataclass
class _Ctx:
    doc: ParsedDoc
    sheet: Sheet
    spec: dict
    row: int | None = None                # record / anchor row
    detail_rows: list[int] = field(default_factory=list)   # blocks: anchor row + detail rows


def _pick_sheet(doc: ParsedDoc, name: str | None) -> Sheet:
    if name:
        for s in doc.sheets:
            if s.name == name:
                return s
        low = {s.name.lower().strip(): s for s in doc.sheets}
        if name.lower().strip() in low:
            return low[name.lower().strip()]
        raise SpecError(f"sheet {name!r} not found; available: {[s.name for s in doc.sheets]}")
    for s in doc.sheets:
        if not s.hidden:
            return s
    return doc.sheets[0]


def _regex(text: Any, src: dict) -> Any:
    if "regex" not in src:
        return text
    if text is None:
        return None
    try:
        m = re.search(src["regex"], str(text))
    except re.error as e:
        raise SpecError(f"invalid regex {src['regex']!r}: {e}") from e
    if not m:
        return None
    return m.group(src.get("group", 1)) if m.re.groups else m.group(0)


def _finish(raw: Cell, src: dict) -> Cell:
    v = raw.value
    if isinstance(v, str) and src.get("strip", True):
        v = re.sub(r"\s+", " ", v).strip() if src.get("collapse_ws") else v.strip()
    out = coerce(v, src.get("as")) if not is_blank(v) else None
    if out is None and "default" in src:
        out = src["default"]
    return Cell(out, raw.refs)


def eval_source(src: dict, ctx: _Ctx) -> Cell:
    sh, spec = ctx.sheet, ctx.spec
    if "const" in src:
        return _finish(Cell(src["const"]), src)
    if "cell" in src:
        target = _pick_sheet(ctx.doc, src["sheet"]) if src.get("sheet") else sh
        r, c = parse_ref(src["cell"])
        v = target.get(r, c)
        return _finish(Cell(_regex(v, src), [target.ref(r, c)] if v is not None else []), src)
    if "col" in src:
        if ctx.row is None:
            raise SpecError("'col' source used for a document-level field; use 'cell' instead")
        c = col_index(src["col"])
        v = sh.get(ctx.row, c)
        return _finish(Cell(_regex(v, src), [sh.ref(ctx.row, c)] if v is not None else []), src)
    if "above" in src:
        a = src["above"]
        if ctx.row is None:
            raise SpecError("'above' needs a row context")
        for r in range(ctx.row - 1, 0, -1):
            if _all(sh, r, a.get("tests")):
                c = col_index(a["col"])
                v = sh.get(r, c)
                return _finish(Cell(_regex(v, src), [sh.ref(r, c)] if v is not None else []), src)
        return _finish(Cell(None), src)
    if "count_numeric_nonzero" in src:
        n, refs = 0, []
        for col in src["count_numeric_nonzero"]:
            x = to_number(sh.get(ctx.row, col_index(col)))
            if x is not None and abs(x) > 1e-12:
                n += 1
                refs.append(sh.ref(ctx.row, col_index(col)))
        return _finish(Cell(n, refs[:1]), {**src, "as": src.get("as", "integer")})
    if "sum_cols" in src:
        tot, refs, seen = 0.0, [], False
        for col in src["sum_cols"]:
            x = to_number(sh.get(ctx.row, col_index(col)))
            if x is not None:
                tot += x
                seen = True
                refs.append(sh.ref(ctx.row, col_index(col)))
        return _finish(Cell(tot if seen else None, refs[:3]), {**src, "as": src.get("as", "number")})
    if "detail_sum" in src or "detail_value" in src:
        kind = "detail_sum" if "detail_sum" in src else "detail_value"
        d = src[kind]
        label_col = d.get("label_col") or (spec.get("block") or {}).get("label_col")
        value_col = d.get("value_col")
        if not label_col or not value_col:
            raise SpecError(f"{kind} needs label_col (or block.label_col) and value_col")
        lc, vc = col_index(label_col), col_index(value_col)
        label_in = {str(x).strip().lower() for x in d["label_in"]} if "label_in" in d else None
        label_not_in = {str(x).strip().lower() for x in d["label_not_in"]} if "label_not_in" in d else None
        total, refs, found = 0.0, [], False
        for r in ctx.detail_rows:
            lab = sh.get(r, lc)
            if lab is None or str(lab).strip() == "":
                continue
            l = str(lab).strip().lower()
            if label_in is not None and l not in label_in:
                continue
            if label_not_in is not None and l in label_not_in:
                continue
            x = to_number(sh.get(r, vc))
            if x is None:
                continue
            found = True
            refs.append(sh.ref(r, vc))
            if kind == "detail_value":
                return _finish(Cell(x, refs), {**src, "as": src.get("as", "number")})
            total += x
        if kind == "detail_value":
            return _finish(Cell(None), src)
        return _finish(Cell(total if found else None, refs[:3]), {**src, "as": src.get("as", "number"),
                                                                  "default": src.get("default", 0)})
    if "map" in src:
        m = src["map"]
        inner = eval_source(m["from"], ctx)
        key = None if inner.value is None else str(inner.value).strip()
        table = m.get("table", {})
        if key is not None:
            lowered = {str(k).strip().lower(): v for k, v in table.items()}
            if key.lower() in lowered:
                return _finish(Cell(lowered[key.lower()], inner.refs), src)
        return _finish(Cell(m.get("default"), inner.refs), src)
    if "rules" in src:
        for rule in src["rules"]:
            if ctx.row is not None and _all(sh, ctx.row, rule.get("when")):
                if "source" in rule:
                    return _finish(eval_source(rule["source"], ctx), src)
                return _finish(Cell(rule.get("value")), src)
        return _finish(Cell(src.get("default")), src)
    raise SpecError(f"unrecognised field source {src!r}")


# ----------------------------------------------------------------- driver

@dataclass
class SheetResult:
    rows: list[dict[str, Cell]]
    warnings: list[str] = field(default_factory=list)
    sheet: str = ""


def run_spec(doc: ParsedDoc, spec: dict, field_keys: list[str], primary: str | None) -> SheetResult:
    if not isinstance(spec, dict):
        raise SpecError("spec must be a JSON object")
    sheet = _pick_sheet(doc, spec.get("sheet"))
    layout = spec.get("layout", "flat")
    if layout not in ("flat", "blocks"):
        raise SpecError(f"layout must be 'flat' or 'blocks', got {layout!r}")
    fields: dict = spec.get("fields") or {}
    if not fields:
        raise SpecError("spec has no 'fields'")
    record_tests = spec.get("record") or []
    if not record_tests:
        raise SpecError("spec has no 'record' tests (which rows are data rows?)")
    first = int(spec.get("first_row") or 1)
    last = int(spec.get("last_row") or sheet.max_row)
    warnings: list[str] = []

    # 1. find record rows (flat) or block anchors (blocks), honouring skip/stop
    anchors: list[int] = []
    stop_row: int | None = None
    for r in range(first, last + 1):
        if spec.get("stop") and _any(sheet, r, spec["stop"]):
            stop_row = r
            break
        if spec.get("skip") and _any(sheet, r, spec["skip"]):
            continue
        if _all(sheet, r, record_tests):
            anchors.append(r)
    end = (stop_row - 1) if stop_row else last

    # 2. group detail rows per anchor
    groups: list[tuple[int, list[int]]] = []
    for i, a in enumerate(anchors):
        if layout == "blocks":
            nxt = anchors[i + 1] if i + 1 < len(anchors) else end + 1
            detail = [r for r in range(a, nxt) if sheet.row_values(r) and not (spec.get("skip") and _any(sheet, r, spec["skip"]))]
        else:
            detail = [a]
        groups.append((a, detail))

    # 3. evaluate fields
    doc_cache: dict[str, Cell] = {}
    rows: list[dict[str, Cell]] = []
    for a, detail in groups:
        ctx = _Ctx(doc=doc, sheet=sheet, spec=spec, row=a, detail_rows=detail)
        rec: dict[str, Cell] = {}
        for key in field_keys:
            src = fields.get(key)
            if src is None:
                rec[key] = Cell(None)
                continue
            is_doc = ("cell" in src or "const" in src) and "col" not in src
            if is_doc and key in doc_cache:
                rec[key] = doc_cache[key]
                continue
            cell = eval_source(src, ctx)
            if is_doc:
                doc_cache[key] = cell
            rec[key] = cell
        rows.append(rec)

    # 4. duplicate policy on the primary key
    policy = spec.get("duplicates", "first")
    if primary and policy in ("first", "last"):
        seen: dict[str, int] = {}
        keep: list[dict[str, Cell]] = []
        dropped = 0
        for rec in rows:
            k = rec[primary].value
            k = "" if k is None else str(k).strip()
            if k in seen and k != "":
                dropped += 1
                if policy == "last":
                    keep[seen[k]] = rec
                continue
            seen[k] = len(keep)
            keep.append(rec)
        if dropped:
            warnings.append(f"dropped {dropped} duplicate row(s) by primary key ({policy} kept)")
        rows = keep
    return SheetResult(rows=rows, warnings=warnings, sheet=sheet.name)


def check_result(res: SheetResult, field_keys: list[str], primary: str | None, spec: dict) -> list[str]:
    """Problems a model can act on. Empty list = looks sane."""
    issues: list[str] = []
    if not res.rows:
        issues.append("the spec matched ZERO data rows: 'record' tests are too strict or first_row/sheet is wrong")
        return issues
    if primary:
        vals = [r[primary].value for r in res.rows]
        blank = sum(1 for v in vals if v in (None, ""))
        if blank:
            issues.append(f"{blank}/{len(vals)} rows have an empty primary key '{primary}'")
        if len(set(map(str, vals))) < len(vals) and spec.get("duplicates") == "keep":
            issues.append(f"primary key '{primary}' is not unique across rows")
    fields = spec.get("fields", {})
    for key in field_keys:
        if key not in fields:
            continue
        col_vals = [r[key].value for r in res.rows]
        if all(v is None for v in col_vals) and "default" not in fields[key] and "const" not in fields[key]:
            issues.append(f"field '{key}' is null in every row (wrong column/cell?)")
    return issues
