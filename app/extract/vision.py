"""Page-image verification for scanned documents.

OCR cannot see handwriting or initials and mangles digits. For every extracted field whose evidence
is on a scanned page we show a vision model the page image next to the candidate value and let it
confirm or correct. Changes are accepted only when the vision model is confident, and only the pages
that actually carry cited fields are sent (cost scales with fields found, not document length).
"""
from __future__ import annotations

import base64
import logging
from concurrent.futures import ThreadPoolExecutor

from ..llm import LLMError, chat_json
from ..parse.pdf import render_page_jpeg
from ..parse.types import ParsedDoc
from . import prompts
from .ctx import JobContext
from .fieldspec import FieldCard
from .records import FieldOut, _coerce

log = logging.getLogger("vision")

MAX_PAGES = 14
ACCEPT_CONF = 0.85


def verify_scanned(doc: ParsedDoc, cards: list[FieldCard], res: dict[str, FieldOut], ctx: JobContext) -> None:
    if doc.kind != "pdf" or not doc.scanned_pages:
        return
    scanned = set(doc.scanned_pages)
    by_page: dict[int, list[FieldCard]] = {}
    for c in cards:
        f = res.get(c.key)
        if f and f.value is not None and f.page in scanned and f.value != []:
            by_page.setdefault(f.page, []).append(c)           # type: ignore[arg-type]
    if not by_page:
        return
    pages = sorted(by_page, key=lambda p: -len(by_page[p]))[:MAX_PAGES]
    ctx.stage("VISION-CHECK")
    model = ctx.settings.model_vision

    def one(page: int) -> None:
        cs = by_page[page]
        try:
            img = render_page_jpeg(doc.path, page)
        except Exception as e:  # noqa: BLE001
            ctx.warn(f"could not render page {page} for vision check: {e}")
            return
        lines = [f'- "{c.key}": {c.description}\n    candidate: {res[c.key].value!r}' for c in cs]
        text = ("FIELDS TO VERIFY\n" + "\n".join(lines) +
                f"\n\nOCR TEXT OF THIS PAGE (may contain errors)\n{doc.page_text(page)[:7000]}")
        content = [{"type": "text", "text": text},
                   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(img).decode()}}]
        try:
            obj, _ = chat_json(model, [{"role": "system", "content": prompts.VISION_SYSTEM},
                                       {"role": "user", "content": content}],
                               tracker=ctx.tracker, max_tokens=2500, deadline=ctx.deadline, tag=f"vision-p{page}")
        except LLMError as e:
            ctx.warn(f"vision check failed for page {page}: {e}")
            return
        raw = obj.get("fields", obj) if isinstance(obj, dict) else {}
        for c in cs:
            r = raw.get(c.key) if isinstance(raw, dict) else None
            if not isinstance(r, dict):
                continue
            f = res[c.key]
            conf = r.get("confidence")
            conf = float(conf) if isinstance(conf, (int, float)) else 0.0
            if r.get("agrees") is True:
                f.confidence = min(1.0, max(f.confidence, 0.85) + 0.05)
                f.flags.add("vision_agrees")
                continue
            if r.get("agrees") is False and conf >= ACCEPT_CONF and "value" in r:
                v, ok = _coerce(r["value"], c)
                if ok and v is not None and v != f.value:
                    log.info("vision changed value", extra={"job_id": ctx.job_id, "field": c.key,
                                                            "old": str(f.value)[:60], "new": str(v)[:60]})
                    f.value, f.source, f.confidence = v, "vision", min(conf, 0.95)
                    f.flags.add("vision_changed")
                    f.calc = (f.calc + " | " if f.calc else "") + f"vision: {str(r.get('note', ''))[:80]}"

    with ThreadPoolExecutor(max_workers=min(4, len(pages))) as ex:
        list(ex.map(one, pages))
