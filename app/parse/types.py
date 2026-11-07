"""Normalised in-memory representation of an input document."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def col_letter(n: int) -> str:
    """1 -> A, 27 -> AA"""
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def col_index(letters: str) -> int:
    n = 0
    for ch in letters.upper():
        n = n * 26 + (ord(ch) - 64)
    return n


_REF = re.compile(r"^([A-Za-z]{1,3})(\d+)$")


def parse_ref(ref: str) -> tuple[int, int]:
    """'F4' -> (row=4, col=6)"""
    m = _REF.match(ref.strip().replace("$", ""))
    if not m:
        raise ValueError(f"bad cell reference {ref!r}")
    return int(m.group(2)), col_index(m.group(1))


@dataclass
class Page:
    number: int                 # 1-based
    text: str
    scanned: bool = False       # no usable text layer in the source file
    ocr: bool = False           # text came from OCR


@dataclass
class Sheet:
    name: str
    cells: dict[tuple[int, int], Any]      # (row, col) -> value, only non-empty cells
    max_row: int
    max_col: int
    hidden: bool = False

    def get(self, row: int, col: int) -> Any:
        return self.cells.get((row, col))

    def row_values(self, row: int) -> dict[int, Any]:
        return {c: v for (r, c), v in self.cells.items() if r == row}

    def rows(self) -> list[int]:
        return sorted({r for (r, _c) in self.cells})

    def ref(self, row: int, col: int) -> str:
        return f"{self.name}!{col_letter(col)}{row}"


@dataclass
class ParsedDoc:
    kind: str                   # pdf | docx | xlsx | csv
    path: Path
    pages: list[Page] = field(default_factory=list)
    sheets: list[Sheet] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_spreadsheet(self) -> bool:
        return self.kind in ("xlsx", "csv")

    @property
    def scanned_pages(self) -> list[int]:
        return [p.number for p in self.pages if p.scanned]

    def page_text(self, n: int) -> str:
        for p in self.pages:
            if p.number == n:
                return p.text
        return ""

    def full_text(self, sep: str = "\n\n") -> str:
        if self.pages:
            return sep.join(f"<<<PAGE {p.number}>>>\n{p.text}" for p in self.pages)
        return sep.join(render_sheet(s) for s in self.sheets)


def _cell_str(v: Any, limit: int) -> str:
    if v is None:
        return ""
    if hasattr(v, "isoformat"):
        try:
            v = v.date().isoformat() if hasattr(v, "date") and getattr(v, "hour", 0) == 0 and getattr(v, "minute", 0) == 0 else v.isoformat()
        except Exception:  # noqa: BLE001
            v = str(v)
    s = str(v).replace("\n", " ⏎ ").strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


def render_rows(sheet: Sheet, rows: list[int], *, limit: int = 60) -> str:
    """'7: [A]Bldg-Unit [B]Unit Type …' — compact, coordinate-preserving view for prompts."""
    out = []
    for r in rows:
        vals = sheet.row_values(r)
        if not vals:
            continue
        out.append(f"{r}: " + " ".join(f"[{col_letter(c)}]{_cell_str(v, limit)}" for c, v in sorted(vals.items())))
    return "\n".join(out)


def render_sheet(sheet: Sheet, *, limit: int = 200) -> str:
    return f"<<<SHEET {sheet.name}>>>\n" + render_rows(sheet, sheet.rows(), limit=limit)
