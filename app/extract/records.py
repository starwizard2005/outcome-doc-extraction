"""Single-record extraction (leases, offering memoranda, anything without a primary row key).

Pipeline: one cheap all-fields pass -> deterministic checks (types, quote really in the document,
cross-field arithmetic) -> a targeted strong-model re-read ONLY for the fields that failed a check
or that the cheap model was unsure about -> optional page-image verification for scanned documents.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..llm import LLMError, chat_json
from ..parse.types import Page, ParsedDoc
from . import prompts
from .ctx import JobContext
from .derive import apply_rules
from .fieldspec import FieldCard, card_lines
from .normalize import coerce, is_blank, to_bool, to_date, to_number
from .relevance import select_pages
from .verify import quote_support, value_matches_quote

log = logging.getLogger("records")

FIRST_PASS_CHARS = 330_000      # ~90k tokens; beyond this we retrieve the relevant pages instead
ESCALATE_CHARS = 150_000
LOW_CONF = 0.80
QUOTE_OK = 0.72

_NUM_TOKEN = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")
_ORDINAL_DATE = re.compile(r"(\d{1,2})(?:st|nd|rd|th)?\s+day\s+of\s+([A-Za-z]+),?\s+(\d{4})", re.I)


@dataclass
class FieldOut:
    value: Any = None
    quote: str = ""
    page: int | None = None
    confidence: float = 0.0
    calc: str = ""
    flags: set[str] = field(default_factory=set)
    source: str = "first"        # first | strong | vision | derived


def _pages_text(pages: list[Page]) -> str:
    return "\n\n".join(f"<<<PAGE {p.number}>>>\n{p.text}" for p in pages)


def _num_from_text(s: str) -> float | None:
    toks = _NUM_TOKEN.findall(s)
    if len(toks) == 1:
        return to_number(toks[0])
    return None


def _coerce_scalar(v: Any, t: str) -> tuple[Any, bool]:
    """-> (value, ok). Non-destructive: on failure for text-like content the original string survives."""
    if is_blank(v):
        return None, True
    if isinstance(v, dict):                       # e.g. {"amount": 2845, "unit": "month"} -> take the first usable scalar
        for key in ("value", "amount", "monthly", "rent", "number", "total"):
            if key in v and not isinstance(v[key], (dict, list)):
                return _coerce_scalar(v[key], t)
        for x in v.values():
            if isinstance(x, (int, float, str)) and not is_blank(x):
                c, ok = _coerce_scalar(x, t)
                if ok:
                    return c, ok
        return None, False
    if t in ("number", "integer"):
        x = to_number(v)
        if x is None and isinstance(v, str):
            x = _num_from_text(v)
        if x is None:
            return (v if isinstance(v, str) and len(v) > 12 else None), False
        return (int(round(x)) if t == "integer" else x), True
    if t == "date":
        d = to_date(v)
        if d is None and isinstance(v, str):
            m = _ORDINAL_DATE.search(v)
            if m:
                d = to_date(f"{m.group(2)} {m.group(1)}, {m.group(3)}")
        return d, d is not None
    if t == "boolean":
        b = to_bool(v)
        return b, b is not None
    return coerce(v, "string"), True


def _coerce(v: Any, card: FieldCard) -> tuple[Any, bool]:
    if card.type == "list":
        if v is None:
            return None, True
        items = v if isinstance(v, list) else [v]
        out, ok_all = [], True
        for it in items:
            c, ok = _coerce_scalar(it, card.item_type if card.item_type != "string" else "string")
            out.append(c)
            ok_all &= ok
        return out, ok_all
    return _coerce_scalar(v, card.type)


def _parse_fields(obj: Any, cards: list[FieldCard]) -> dict[str, FieldOut]:
    raw = obj.get("fields", obj) if isinstance(obj, dict) else {}
    res: dict[str, FieldOut] = {}
    for c in cards:
        r = raw.get(c.key) if isinstance(raw, dict) else None
        if not isinstance(r, dict):
            r = {"value": r} if r is not None else {}
        v, ok = _coerce(r.get("value"), c)
        conf = r.get("confidence")
        conf = float(conf) if isinstance(conf, (int, float)) and 0 <= conf <= 1 else 0.5
        page = r.get("page")
        out = FieldOut(value=v, quote=str(r.get("quote") or "")[:400], page=page if isinstance(page, int) else None,
                       confidence=conf, calc=str(r.get("calc") or ""))
        if not ok:
            out.flags.add("type_fail")
            logging.getLogger("records").warning("type coercion failed", extra={"field": c.key, "type": c.type,
                                                                                 "raw": repr(r.get("value"))[:160]})
        res[c.key] = out
    return res


def _check_quotes(res: dict[str, FieldOut], doc: ParsedDoc, cards: list[FieldCard] | None = None) -> None:
    ftypes = {c.key: c.type for c in cards} if cards else {}
    for k, f in res.items():
        empty = f.value is None or (isinstance(f.value, list) and (not f.value or all(x is None for x in f.value)))
        if empty and f.quote and re.search(r"\d|[A-Za-z]{4}", f.quote):
            f.flags.add("null_but_quoted")          # the model cited evidence yet gave no usable value
            continue
        if isinstance(f.value, list) and any(x is None for x in f.value):
            f.flags.add("null_in_list")
        if f.value is None or isinstance(f.value, list) and not f.value:
            continue
        if not f.quote and not f.calc:
            f.flags.add("no_quote")
            f.confidence = min(f.confidence, 0.7)
            continue
        if not f.quote:
            continue
        score, page = quote_support(f.quote, doc.pages, f.page)
        if score < QUOTE_OK:
            f.flags.add("quote_unverified")
            f.confidence = min(f.confidence, 0.6)
        elif page is not None:
            f.page = page
        if "quote_unverified" not in f.flags and not f.calc and ftypes:
            if value_matches_quote(f.value, ftypes.get(k, "string"), f.quote) is False:
                f.flags.add("value_not_in_quote")
                f.confidence = min(f.confidence, 0.55)


def _needs_second_look(f: FieldOut) -> bool:
    return bool(f.flags & {"type_fail", "quote_unverified", "value_not_in_quote", "null_but_quoted", "null_in_list"}) or f.confidence < LOW_CONF


def _call(model: str, system: str, user: str, ctx: JobContext, tag: str, max_tokens: int = 8000) -> Any:
    obj, _ = chat_json(model, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                       tracker=ctx.tracker, max_tokens=max_tokens, deadline=ctx.deadline, tag=tag)
    return obj


def extract_record(doc: ParsedDoc, cards: list[FieldCard], ctx: JobContext) -> dict[str, FieldOut]:
    s = ctx.settings
    query = " ".join(f"{c.key} {c.name} {c.description}" for c in cards)

    # ---- pass 1: everything, cheap model
    ctx.stage("EXTRACTING")
    pages = select_pages(doc.pages, query, FIRST_PASS_CHARS)
    if len(pages) < len(doc.pages):
        ctx.warn(f"document too long for one pass: using {len(pages)}/{len(doc.pages)} most relevant pages")
    user = prompts.RECORD_USER.format(fields=card_lines(cards), document=_pages_text(pages))
    try:
        res = _parse_fields(_call(s.model_record, prompts.RECORD_SYSTEM, user, ctx, "record-pass1"), cards)
    except LLMError as e:
        ctx.warn(f"first pass failed with {s.model_record}: {e}; retrying with {s.model_strong}")
        res = _parse_fields(_call(s.model_strong, prompts.RECORD_SYSTEM, user, ctx, "record-pass1-fallback"), cards)

    # ---- deterministic checks
    ctx.stage("VERIFYING")
    _check_quotes(res, doc, cards)
    _derive(res, cards, ctx)

    # ---- pass 2: strong model, only the doubtful fields, only the relevant pages
    doubtful = [c for c in cards if _needs_second_look(res[c.key])]
    if doubtful and ctx.time_left() > 120:
        _escalate(doc, cards, doubtful, res, ctx)
        _check_quotes({c.key: res[c.key] for c in doubtful}, doc, cards)
        _derive(res, cards, ctx)
    return res


def _derive(res: dict[str, FieldOut], cards: list[FieldCard], ctx: JobContext) -> None:
    notes: list[str] = []
    changes = apply_rules({k: f.value for k, f in res.items()}, notes)
    for k, v in changes.items():
        f = res[k]
        f.value, f.source = v, "derived"
        f.confidence = max(f.confidence, 0.9)
        f.flags.discard("type_fail")
        f.calc = f.calc or "recomputed from other extracted fields"
    for n in notes:
        ctx.warn("derive: " + n)


def _escalate(doc: ParsedDoc, cards: list[FieldCard], doubtful: list[FieldCard], res: dict[str, FieldOut],
              ctx: JobContext) -> None:
    s = ctx.settings
    ctx.stage("RE-READING")
    q = " ".join(f"{c.key} {c.name} {c.description} {res[c.key].quote}" for c in doubtful)
    pages = select_pages(doc.pages, q, ESCALATE_CHARS, always_first=2)
    lines = []
    for c in doubtful:
        f = res[c.key]
        tt = f"list of {c.item_type}" if c.type == "list" else c.type
        cand = "null" if f.value is None else repr(f.value)
        lines.append(f'- "{c.key}" ({tt}): {c.description}\n    candidate: {cand}  (page {f.page}, quote: {f.quote[:120]!r})')
    why = "low confidence, unverifiable quote or wrong type"
    user = prompts.ESCALATE_USER.format(why=why, fields="\n".join(lines), document=_pages_text(pages))
    try:
        new = _parse_fields(_call(s.model_strong, prompts.RECORD_SYSTEM, user, ctx, "record-escalate"), doubtful)
    except LLMError as e:
        ctx.warn(f"escalation failed: {e}")
        return
    doc_for_check = doc
    for c in doubtful:
        n, old = new[c.key], res[c.key]
        _check_quotes({c.key: n}, doc_for_check, [c])
        # strong model wins when its answer is verifiable, or when the old one was broken anyway
        if n.value is None and old.value is not None and not (old.flags & {"quote_unverified", "value_not_in_quote", "type_fail"}):
            old.confidence = min(old.confidence, 0.6)       # disagreement: keep candidate but mark unsure
            continue
        if (n.flags & {"quote_unverified", "value_not_in_quote"}) and old.value is not None and not (
                old.flags & {"quote_unverified", "value_not_in_quote", "type_fail"}):
            continue
        n.source = "strong"
        res[c.key] = n
