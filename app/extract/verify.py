"""Self-checks that do not need a model: is the cited quote really in the document?"""
from __future__ import annotations

import re
from collections import Counter

from ..parse.types import Page

_WS = re.compile(r"\s+")
_NON = re.compile(r"[^a-z0-9$%.,/\- ]")


def norm_text(s: str) -> str:
    return _WS.sub(" ", _NON.sub(" ", s.lower())).strip()


def _toks(s: str) -> list[str]:
    return [t for t in norm_text(s).split(" ") if t]


def quote_support(quote: str, pages: list[Page], hint_page: int | None = None) -> tuple[float, int | None]:
    """Best fuzzy support of `quote` in the document: (score 0..1, page number).
    Tolerates OCR noise: compares token overlap in the best window rather than exact substrings."""
    q = _toks(quote)
    if not q:
        return 0.0, None
    qn = " ".join(q)
    order = list(pages)
    if hint_page is not None:
        order.sort(key=lambda p: 0 if p.number == hint_page else 1)
    best = (0.0, None)
    qc = Counter(q)
    for p in order:
        pn = norm_text(p.text)
        if qn in pn:
            return 1.0, p.number
        pt = pn.split(" ")
        if len(pt) < 1:
            continue
        w = max(len(q) + 3, int(len(q) * 1.4))
        # coarse-to-fine: slide in steps of ~half a quote
        step = max(1, len(q) // 2)
        for i in range(0, max(1, len(pt) - w + step), step):
            win = Counter(pt[i:i + w])
            overlap = sum(min(c, win.get(t, 0)) for t, c in qc.items())
            sc = overlap / len(q)
            if sc > best[0]:
                best = (sc, p.number)
        if best[0] >= 0.97:
            break
    return best


def value_in_text(value, pages: list[Page]) -> bool:
    """Weak check that a scalar's digits/words appear somewhere (used for numbers and dates)."""
    if value is None or isinstance(value, bool):
        return True
    s = str(value)
    hay = "\n".join(p.text for p in pages)
    if isinstance(value, (int, float)):
        digits = f"{value:,.2f}".rstrip("0").rstrip(".")
        plain = str(int(value)) if float(value).is_integer() else str(value)
        return any(x in hay for x in (digits, plain, f"{value:,.0f}", f"{value:,.2f}"))
    return norm_text(s) in norm_text(hay)


# ---------------------------------------------------------------- value vs quote

_NUM_TOK = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december"]


def _numbers_in(text: str) -> list[float]:
    out = []
    for t in _NUM_TOK.findall(text):
        try:
            out.append(float(t.replace("$", "").replace(",", "")))
        except ValueError:
            pass
    return out


def value_matches_quote(value, ftype: str, quote: str) -> bool | None:
    """Is the extracted scalar actually visible in the text it cites? None = cannot tell / not applicable.

    Catches the classic LLM failure of citing the right sentence but reporting a different number/date.
    Lists, booleans and long free-text answers are not checked (they are legitimately paraphrased)."""
    if value is None or isinstance(value, (bool, list)) or not quote:
        return None
    if ftype in ("number", "integer") and isinstance(value, (int, float)):
        nums = _numbers_in(quote)
        if not nums:
            return False
        v = float(value)
        return any(abs(n - v) <= max(0.011, abs(v) * 0.005) or abs(n * 12 - v) <= 0.01 or abs(n / 100 - v) <= 1e-9
                   for n in nums)
    if ftype == "date" and isinstance(value, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", value):
        y, m, d = value.split("-")
        q = quote.lower()
        if y not in q and y[2:] not in re.findall(r"\d{2}", q):
            return False
        month_ok = _MONTHS[int(m) - 1] in q or _MONTHS[int(m) - 1][:3] in q or re.search(rf"\b0?{int(m)}[/-]", q) is not None
        day_ok = re.search(rf"\b0?{int(d)}(?:st|nd|rd|th)?\b", q) is not None
        return bool(month_ok and day_ok)
    if ftype == "string" and isinstance(value, str):
        vt = [t for t in _toks(value) if len(t) >= 3]
        if not vt or len(vt) > 5:
            return None
        qt = set(_toks(quote))
        return sum(1 for t in vt if t in qt) / len(vt) >= 0.6
    return None
