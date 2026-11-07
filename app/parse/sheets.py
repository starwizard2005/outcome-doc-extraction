"""XLSX / CSV -> Sheet grids.

XLSX: we read cached values (data_only) and fall back to evaluating simple formulas
(SUM / + / -) when a file was saved without cached results, so computed totals are never lost.
Merged cells keep only their top-left value, like Excel itself.
"""
from __future__ import annotations

import csv
import io
import logging
import re
from pathlib import Path
from typing import Any

from .types import Sheet, col_index, parse_ref

log = logging.getLogger("parse.sheets")

_NUM = re.compile(r"^-?(0|[1-9]\d*)(\.\d+)?$")


def _maybe_number(s: str) -> Any:
    t = s.strip()
    if _NUM.match(t):
        return float(t) if "." in t else int(t)
    return s


# ----------------------------------------------------------------- csv

def parse_csv(path: Path, warnings: list[str]) -> list[Sheet]:
    raw = path.read_bytes()
    text = None
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except Exception:  # noqa: BLE001
            continue
    assert text is not None
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    cells: dict[tuple[int, int], Any] = {}
    max_c = 0
    n = 0
    for r, row in enumerate(csv.reader(io.StringIO(text), dialect), start=1):
        n = r
        for c, v in enumerate(row, start=1):
            if v is not None and v.strip() != "":
                cells[(r, c)] = _maybe_number(v)
                max_c = max(max_c, c)
    return [Sheet(name="csv", cells=cells, max_row=n, max_col=max_c)]


# ----------------------------------------------------------------- xlsx

_RANGE = re.compile(r"^\$?([A-Z]{1,3})\$?(\d+):\$?([A-Z]{1,3})\$?(\d+)$")


def _eval_formula(expr: str, get, depth: int = 0) -> float | None:
    """Tiny evaluator: =SUM(range|ref,...) and +/- of refs and numbers. None if unsupported."""
    e = expr.lstrip("=").strip()
    if depth > 3:
        return None

    def ref_val(ref: str) -> float | None:
        r, c = parse_ref(ref)
        v = get(r, c, depth + 1)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    m = re.match(r"^SUM\((.*)\)$", e, re.I)
    if m:
        total = 0.0
        for part in m.group(1).split(","):
            part = part.strip()
            rm = _RANGE.match(part)
            if rm:
                c1, r1, c2, r2 = col_index(rm.group(1)), int(rm.group(2)), col_index(rm.group(3)), int(rm.group(4))
                for rr in range(min(r1, r2), max(r1, r2) + 1):
                    for cc in range(min(c1, c2), max(c1, c2) + 1):
                        v = get(rr, cc, depth + 1)
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            total += v
            else:
                try:
                    total += float(part)
                except ValueError:
                    v = ref_val(part)
                    if v is None:
                        return None
                    total += v
        return total
    toks = re.findall(r"[+-]?\s*[A-Za-z]{1,3}\d+|[+-]?\s*\d+(?:\.\d+)?", e)
    if toks and "".join(t.replace(" ", "") for t in toks) == e.replace(" ", ""):
        total = 0.0
        for t in toks:
            t = t.replace(" ", "")
            sign = -1.0 if t.startswith("-") else 1.0
            body = t.lstrip("+-")
            if re.match(r"^[A-Za-z]", body):
                v = ref_val(body)
                if v is None:
                    return None
                total += sign * v
            else:
                total += sign * float(body)
        return total
    return None


def parse_xlsx(path: Path, warnings: list[str]) -> list[Sheet]:
    import openpyxl

    wb_v = openpyxl.load_workbook(path, data_only=True)
    wb_f = openpyxl.load_workbook(path, data_only=False)
    sheets: list[Sheet] = []
    n_eval = n_unresolved = 0
    for ws_v in wb_v.worksheets:
        ws_f = wb_f[ws_v.title]
        cells: dict[tuple[int, int], Any] = {}
        formulas: dict[tuple[int, int], str] = {}
        for row in ws_f.iter_rows():
            for cell in row:
                if isinstance(cell.value, str) and cell.value.startswith("="):
                    formulas[(cell.row, cell.column)] = cell.value
        for row in ws_v.iter_rows():
            for cell in row:
                v = cell.value
                if isinstance(v, str):
                    if v.strip() == "":
                        continue
                    v = v.strip() if len(v) < 2000 else v
                if v is not None:
                    cells[(cell.row, cell.column)] = v

        def get(r: int, c: int, depth: int = 0, _cells=cells, _f=formulas) -> Any:
            if (r, c) in _cells:
                return _cells[(r, c)]
            if (r, c) in _f:
                return _eval_formula(_f[(r, c)], get, depth)
            return None

        for (r, c), expr in formulas.items():
            if (r, c) not in cells:
                val = _eval_formula(expr, get)
                if val is None:
                    n_unresolved += 1
                else:
                    cells[(r, c)] = val
                    n_eval += 1
        sheets.append(Sheet(name=ws_v.title, cells=cells, max_row=ws_v.max_row, max_col=ws_v.max_column,
                            hidden=ws_v.sheet_state != "visible"))
    if n_eval:
        warnings.append(f"evaluated {n_eval} formula cell(s) with no cached value")
    if n_unresolved:
        warnings.append(f"{n_unresolved} formula cell(s) had no cached value and could not be evaluated")
    return sheets
