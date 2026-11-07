"""PDF -> pages of text. Digital pages use the text layer; pages without one are OCR'd.

Open-source tooling only: poppler (`pdftotext`, `pdftoppm`) and tesseract. No cloud OCR.
`pdftotext -layout` keeps column alignment, which matters for tables inside prose PDFs.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .types import Page

log = logging.getLogger("parse.pdf")

MIN_TEXT_CHARS = 40          # below this a page is treated as image-only
OCR_DPI = 200


def _run(cmd: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, timeout=timeout)


def page_count(path: Path) -> int:
    try:
        out = _run(["pdfinfo", str(path)]).stdout.decode("utf-8", "replace")
        for line in out.splitlines():
            if line.startswith("Pages:"):
                return int(line.split()[1])
    except Exception:  # noqa: BLE001
        pass
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        return len(pdf.pages)


def _text_layer(path: Path, n_pages: int) -> list[str]:
    if shutil.which("pdftotext"):
        r = _run(["pdftotext", "-layout", str(path), "-"])
        if r.returncode == 0:
            pages = r.stdout.decode("utf-8", "replace").split("\f")
            pages = pages[:n_pages] + [""] * max(0, n_pages - len(pages))
            return pages
    import pdfplumber
    out = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            out.append(p.extract_text(layout=True) or "")
    return out


def render_page_png(path: Path, page: int, dpi: int = OCR_DPI, gray: bool = True) -> bytes:
    """Rasterise one page (1-based) to PNG bytes."""
    with tempfile.TemporaryDirectory() as td:
        prefix = os.path.join(td, "p")
        cmd = ["pdftoppm", "-r", str(dpi), "-f", str(page), "-l", str(page), "-png"]
        if gray:
            cmd.append("-gray")
        r = _run(cmd + [str(path), prefix])
        files = sorted(Path(td).glob("p*.png"))
        if r.returncode != 0 or not files:
            raise RuntimeError(f"pdftoppm failed for page {page}: {r.stderr.decode('utf-8', 'replace')[:200]}")
        return files[0].read_bytes()


def render_page_jpeg(path: Path, page: int, dpi: int = 130, quality: int = 70, max_side: int = 1700) -> bytes:
    """Smaller image for vision-model calls (cost scales with pixels)."""
    from io import BytesIO

    from PIL import Image
    img = Image.open(BytesIO(render_page_png(path, page, dpi=dpi, gray=False))).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


# Tesseract is OpenMP-parallel by default; with several pages and several jobs in flight the threads fight
# each other and a 2-second page takes minutes. One thread per process + a process-wide cap on concurrent
# tesseract runs keeps latency flat under load.
_OCR_SLOTS = threading.BoundedSemaphore(max(1, min(4, os.cpu_count() or 1)))
_OCR_ENV = {**os.environ, "OMP_THREAD_LIMIT": "1", "OMP_NUM_THREADS": "1", "OMP_WAIT_POLICY": "PASSIVE"}


def _ocr_page(path: Path, page: int) -> str:
    last: Exception | None = None
    for dpi, timeout in ((OCR_DPI, 90), (150, 120)):
        try:
            png = render_page_png(path, page, dpi=dpi, gray=True)
            with tempfile.TemporaryDirectory() as td:
                img = Path(td) / "page.png"
                img.write_bytes(png)
                with _OCR_SLOTS:
                    r = subprocess.run(["tesseract", str(img), "stdout", "-l", "eng", "--psm", "3"],
                                       capture_output=True, timeout=timeout, env=_OCR_ENV)
                if r.returncode != 0:
                    raise RuntimeError(f"tesseract failed on page {page}: {r.stderr.decode('utf-8', 'replace')[:200]}")
                return r.stdout.decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            last = e
    raise last  # type: ignore[misc]


def parse_pdf(path: Path, warnings: list[str]) -> list[Page]:
    n = page_count(path)
    texts = _text_layer(path, n)
    pages = [Page(number=i + 1, text=t.rstrip(), scanned=len("".join(t.split())) < MIN_TEXT_CHARS)
             for i, t in enumerate(texts)]
    todo = [p for p in pages if p.scanned]
    if todo:
        if not shutil.which("tesseract"):
            warnings.append(f"{len(todo)} scanned page(s) but tesseract is not installed; text unavailable")
            return pages
        workers = max(1, min(4, os.cpu_count() or 1))
        log.info("ocr start", extra={"pages": len(todo), "workers": workers})

        def work(p: Page) -> None:
            try:
                p.text = _ocr_page(path, p.number).rstrip()
                p.ocr = True
            except Exception as e:  # noqa: BLE001
                warnings.append(f"OCR failed on page {p.number}: {e}")

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(work, todo))
    return pages
