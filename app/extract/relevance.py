"""Cheap lexical page retrieval (BM25-style) so long documents never overflow a context window
and escalation calls only pay for the pages that matter."""
from __future__ import annotations

import math
import re
from collections import Counter

from ..parse.types import Page

_STOP = set("""a an and are as at be by for from has have in is it of on or that the their this to was were will with
shall tenant landlord lease premises per each any all not no if as such may must than then which who whom whose
value field number text date string exactly stated written document list null otherwise only given use used using
""".split())


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9$%]+", text.lower()) if t not in _STOP and len(t) > 1]


def rank_pages(pages: list[Page], query: str, k: int | None = None) -> list[Page]:
    """Pages ordered by relevance to `query` (best first)."""
    docs = [Counter(tokens(p.text)) for p in pages]
    n = len(docs)
    avg = sum(sum(d.values()) for d in docs) / max(n, 1) or 1.0
    df: Counter = Counter()
    for d in docs:
        df.update(d.keys())
    q = Counter(tokens(query))
    scored = []
    for p, d in zip(pages, docs):
        dl = sum(d.values()) or 1
        s = 0.0
        for t, qf in q.items():
            f = d.get(t, 0)
            if not f:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * (f * 2.2) / (f + 1.2 * (0.25 + 0.75 * dl / avg))
        scored.append((s, p))
    scored.sort(key=lambda x: -x[0])
    out = [p for _, p in scored]
    return out[:k] if k else out


def select_pages(pages: list[Page], query: str, max_chars: int, always_first: int = 2) -> list[Page]:
    """Keep document order, but drop the least relevant pages until the text fits `max_chars`."""
    if sum(len(p.text) for p in pages) <= max_chars:
        return list(pages)
    keep: dict[int, Page] = {p.number: p for p in pages[:always_first]}
    used = sum(len(p.text) for p in keep.values())
    for p in rank_pages(pages, query):
        if p.number in keep:
            continue
        if used + len(p.text) > max_chars:
            continue
        keep[p.number] = p
        used += len(p.text)
    return [keep[n] for n in sorted(keep)]
