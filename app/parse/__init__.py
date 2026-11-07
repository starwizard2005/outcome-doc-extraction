"""Document loading: sniff the real file type (never trust the extension) and parse."""
from __future__ import annotations

import zipfile
from pathlib import Path

from .types import ParsedDoc


class UnsupportedFile(ValueError):
    pass


def sniff_kind(path: Path, filename: str = "") -> str:
    """pdf | docx | xlsx | csv, from content. Raises UnsupportedFile otherwise."""
    with open(path, "rb") as f:
        head = f.read(8)
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"PK"):
        try:
            names = set(zipfile.ZipFile(path).namelist())
        except zipfile.BadZipFile as e:
            raise UnsupportedFile("corrupt zip container") from e
        if any(n.startswith("word/") for n in names):
            return "docx"
        if any(n.startswith("xl/") for n in names):
            return "xlsx"
        raise UnsupportedFile("zip archive is neither DOCX nor XLSX")
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        raise UnsupportedFile("legacy .doc/.xls files are not supported; please save as DOCX/XLSX")
    # text-like → CSV, provided it decodes and looks tabular
    raw = path.read_bytes()[:65536]
    if b"\x00" in raw and not raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise UnsupportedFile("binary file of unknown type")
    printable = sum(1 for b in raw if b in (9, 10, 13) or 32 <= b < 127 or b >= 128)
    if raw and printable / len(raw) < 0.95:
        raise UnsupportedFile("binary file of unknown type")
    first = next((ln for ln in raw.decode("utf-8", "replace").splitlines() if ln.strip()), "")
    if not (filename.lower().endswith(".csv") or any(d in first for d in (",", ";", "\t", "|"))):
        raise UnsupportedFile("unsupported file type: expected PDF, DOCX, XLSX or CSV")
    return "csv"


def load_document(path: Path) -> ParsedDoc:
    path = Path(path)
    kind = sniff_kind(path)
    doc = ParsedDoc(kind=kind, path=path)
    if kind == "pdf":
        from .pdf import parse_pdf
        doc.pages = parse_pdf(path, doc.warnings)
    elif kind == "docx":
        from .docx_ import parse_docx
        doc.pages = parse_docx(path, doc.warnings)
    elif kind == "xlsx":
        from .sheets import parse_xlsx
        doc.sheets = parse_xlsx(path, doc.warnings)
    else:
        from .sheets import parse_csv
        doc.sheets = parse_csv(path, doc.warnings)
    return doc
