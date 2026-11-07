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
