"""Pipeline end-to-end with the scripted fake LLM. Skipped if the sample data is not present."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SAMPLE = ROOT / "assignment" / "sample"
pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample data not available")

from tests import fake_llm  # noqa: E402

CASES = [("boxscore_001", "boxscore_001.csv", "boxscore", 0.999),
         ("t12_001", "t12_001.xlsx", "operating_statement_lines", 0.999),
         ("rentroll_001", "rentroll_001.xlsx", "rent_roll_units", 0.999),
         ("budget_001", "budget_001.pdf", "budget_variance", 0.97),
         ("lease_002", "lease_002.docx", "lease_abstraction", 0.9),
         ("om_001", "om_001.pdf", "offering_memorandum", 0.9)]


@pytest.mark.parametrize("doc_id,fname,schema_name,min_score", CASES)
def test_pipeline_matches_truth(doc_id, fname, schema_name, min_score):
    fake_llm.install()
    from app.extract.ctx import JobContext
    from app.extract.pipeline import run_extraction
    sys.path.insert(0, str(SAMPLE.parent / "tools"))
    import validate as V

    fake_llm.CURRENT["doc_id"] = doc_id
    schema = json.loads((SAMPLE / "schemas" / f"{schema_name}.json").read_text())
    result, status = run_extraction(SAMPLE / "sample-data-documents" / fname, schema, JobContext())
    assert status in ("SUCCESS", "PARTIAL")
    assert not V.validate_contract(result, schema), V.validate_contract(result, schema)[:5]
    truth = json.loads((SAMPLE / "sample-data-output-validated" / f"{doc_id}.json").read_text())
    rep = V.score(result, truth, schema)
    assert rep["score"] >= min_score, (rep["score"], rep["misses"][:5], rep["hallucinations"][:5])
    assert result["metadata"]["model_calls"] >= 1


def test_coercion_and_null_flags():
    from app.extract.records import FieldOut, _check_quotes, _coerce
    from app.extract.fieldspec import FieldCard
    c = FieldCard(key="m", name="m", description="monthly rent as a list of numbers", type="list", item_type="number")
    assert _coerce([{"amount": 2845, "unit": "month"}], c) == ([2845.0], True)
    assert _coerce(["$2,845.00"], c) == ([2845.0], True)
    f = FieldOut(value=[None], quote="October 1, 2025 through April 30, 2027 $2,845.00", page=1, confidence=0.95)
    from app.parse.types import Page, ParsedDoc
    doc = ParsedDoc(kind="pdf", path=None)
    doc.pages = [Page(number=1, text="October 1, 2025 through April 30, 2027 $2,845.00")]
    _check_quotes({"m": f}, doc, [c])
    assert "null_but_quoted" in f.flags


def test_vision_fill_accepts_confident_value(monkeypatch):
    from app.extract import vision
    from app.extract.ctx import JobContext
    from app.extract.fieldspec import FieldCard
    from app.extract.records import FieldOut
    from app.parse.types import Page, ParsedDoc
    seen = {}

    def fake_chat_json(model, messages, **kw):
        seen["tag"] = kw.get("tag")
        return {"fields": {"deposit": {"value": "$2,150", "confidence": 0.9, "note": "handwritten"}}}, None

    monkeypatch.setattr(vision, "chat_json", fake_chat_json)
    monkeypatch.setattr(vision, "render_page_jpeg", lambda *a, **k: b"\xff\xd8fake")
    doc = ParsedDoc(kind="pdf", path="x.pdf")
    doc.pages = [Page(number=1, text="cover", scanned=True), Page(number=2, text="SECURITY DEPOSIT: garbled", scanned=True)]
    card = FieldCard(key="deposit", name="security deposit", description="security deposit amount", type="number")
    res = {"deposit": FieldOut(value=None, quote="", page=None, confidence=0.5)}
    vision.fill_missing(doc, [card], res, JobContext())
    assert res["deposit"].value == 2150.0 and res["deposit"].source == "vision" and "vision_filled" in res["deposit"].flags
    # a confident null from the text models is left alone
    res2 = {"deposit": FieldOut(value=None, quote="", page=None, confidence=0.95)}
    vision.fill_missing(doc, [card], res2, JobContext())
    assert res2["deposit"].value is None
