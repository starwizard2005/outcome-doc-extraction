"""A scripted stand-in for the LLM gateway, used to test plumbing offline (no network, no cost).

It answers each kind of pipeline call from the sample answer keys / golden specs. For record
extraction it deliberately returns a few WRONG values in pass 1 so the escalation path is exercised.
This verifies orchestration, parsing, normalisation, output shape and the validator contract; it says
nothing about real model quality (that is measured by eval/run_eval.py against the real gateway).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from app import llm as L

SAMPLE = Path(__file__).resolve().parents[1] / "assignment" / "sample"
GOLD = Path(__file__).with_name("golden_specs")
SCHEMA_TO_GOLD = {"boxscore": "boxscore", "operating_statement_lines": "t12", "rent_roll_units": "rentroll"}

CURRENT = {"doc_id": None, "calls": []}


def truth(doc_id: str) -> list[dict]:
    return json.loads((SAMPLE / "sample-data-output-validated" / f"{doc_id}.json").read_text())["entries"]


def _val(e, k):
    f = e.get(k)
    return f["value"] if isinstance(f, dict) else f


def _result(text: str, model: str, tracker) -> L.LLMResult:
    if tracker:
        tracker.add(model, 1000, 200, 0.0003, cached=False)
    return L.LLMResult(text, model, 1000, 200, 0.0003, False, 5)


def fake_chat(model, messages, *, tracker=None, max_tokens=4096, temperature=0.0, deadline=None, cache=True, tag="", max_attempts=8):
    user = messages[-1]["content"] if isinstance(messages[-1]["content"], str) else json.dumps(messages[-1]["content"])
    for m in reversed(messages):
        if m["role"] == "user" and isinstance(m["content"], str) and ("SCHEMA:" in m["content"] or "FIELDS" in m["content"]):
            user = m["content"]
            break
    CURRENT["calls"].append((tag, model))
    doc_id = CURRENT["doc_id"]
    T = truth(doc_id)

    if tag == "spec-audit":
        return _result('{"problems": []}', model, tracker)
    if tag.startswith("spec-"):
        name = re.search(r"SCHEMA: (\S+)", user).group(1)
        return _result((GOLD / f"{SCHEMA_TO_GOLD[name]}.json").read_text(), model, tracker)
    if tag == "table-plan":
        keys = list(T[0].keys())
        doc_keys = ["property_name", "property_rsf", "budget_year"]
        return _result(json.dumps({"row_anchor_regex": r"^\s*(\d{4}-\d{4})\b",
                                   "row_fields": [k for k in keys if k not in doc_keys], "doc_fields": doc_keys}), model, tracker)
    if tag.startswith("table-rows"):
        row_keys = re.findall(r'^- "(\w+)" \(', user, re.M)
        wanted = re.findall(r"key=(\S+)", user)
        by = {_val(e, "gl_code"): e for e in T}
        rows = [{k: _val(by[w], k) for k in row_keys} for w in wanted]
        return _result(json.dumps({"rows": rows}), model, tracker)
    keys = re.findall(r'^- "(\w+)" \(', user, re.M)
    e = T[0]
    if tag.startswith("record"):
        fields = {}
        noisy = set(keys[3:6]) if tag == "record-pass1" and len(keys) > 12 else set()
        for k in keys:
            v = _val(e, k)
            if k in noisy and v is not None:
                v = "WRONG"
            cite = (e[k].get("citation") if isinstance(e.get(k), dict) else "") or ""
            fields[k] = {"value": v, "quote": re.sub(r"^\[[^\]]*\]\s*", "", cite)[:120] if v is not None else "",
                         "page": (e[k].get("page_numbers") or [None])[0] if isinstance(e.get(k), dict) else None,
                         "confidence": 0.95}
        return _result(json.dumps({"fields": fields}), model, tracker)
    if tag.startswith("vision"):
        return _result(json.dumps({"fields": {k: {"agrees": True, "confidence": 0.9} for k in keys}}), model, tracker)
    return _result("{}", model, tracker)


def install(monkeypatch=None):
    L.chat = fake_chat            # chat_json looks `chat` up in the module namespace
    from app import config
    import os
    os.environ.setdefault("LLM_BASE_URL", "http://fake")
    os.environ.setdefault("LLM_API_KEY", "fake")
    config.reset_settings_cache()
