"""DOCX -> pages of text (paragraphs and tables, in document order)."""
from __future__ import annotations

from pathlib import Path

from .types import Page

_EST_CHARS_PER_PAGE = 3000


def parse_docx(path: Path, warnings: list[str]) -> list[Page]:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(str(path))
    blocks: list[tuple[str, int]] = []       # (text, page_breaks_before)
    real_breaks = 0
    body = d.element.body
    for child in body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            p = Paragraph(child, d)
            xml = child.xml
            brk = xml.count("lastRenderedPageBreak") + xml.count('w:type="page"')
            real_breaks += brk
            txt = p.text.strip()
            if txt or brk:
                blocks.append((txt, brk))
        elif tag == "tbl":
            t = Table(child, d)
            for row in t.rows:
                seen, cells = set(), []
                for c in row.cells:
                    if id(c._tc) in seen:        # merged cell repeats
                        continue
                    seen.add(id(c._tc))
                    cells.append(c.text.strip().replace("\n", " "))
                line = " | ".join(x for x in cells if x)
                if line:
                    blocks.append((line, 0))

    pages: list[list[str]] = [[]]
    if real_breaks:
        for txt, brk in blocks:
            if brk and pages[-1]:
                pages.append([])
            if txt:
                pages[-1].append(txt)
    else:
        warnings.append("docx has no page-break information; page numbers are estimated")
        size = 0
        for txt, _ in blocks:
            if size > _EST_CHARS_PER_PAGE and pages[-1]:
                pages.append([])
                size = 0
            pages[-1].append(txt)
            size += len(txt) + 1
    return [Page(number=i + 1, text="\n".join(lines)) for i, lines in enumerate(pages) if lines]
