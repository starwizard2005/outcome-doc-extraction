"""Offline check: golden specs -> executor -> validator against the answer keys. No LLM, no network."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.parse import load_document
from app.extract.sheet_exec import run_spec, check_result

S = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/claude/assignment/sample")
sys.path.insert(0, str(S.parent / "tools"))
import validate as V

CASES = [("boxscore_001", "boxscore.csv", "boxscore", "boxscore"),
         ("t12_001", "t12_001.xlsx", "operating_statement_lines", "t12"),
         ("rentroll_001", "rentroll_001.xlsx", "rent_roll_units", "rentroll")]
for doc_id, fname, schema_name, spec_name in CASES:
    fname = {"boxscore.csv": "boxscore_001.csv"}.get(fname, fname)
    doc = load_document(S / "sample-data-documents" / fname)
    schema = json.loads((S / "schemas" / f"{schema_name}.json").read_text())
    keys = list(schema["fields"]); primary = next(k for k, v in schema["fields"].items() if v.get("primary"))
    spec = json.loads(Path(__file__).with_name("golden_specs").joinpath(f"{spec_name}.json").read_text())
    res = run_spec(doc, spec, keys, primary)
    entries = [{k: {"value": c.value, "citation": "", "confidence": 1.0} for k, c in r.items()} for r in res.rows]
    truth = json.loads((S / "sample-data-output-validated" / f"{doc_id}.json").read_text())
    rep = V.score({"entries": entries}, truth, schema)
    print(f"{doc_id:14s} score={rep['score']:.4f} rows {rep['rows']['predicted']}/{rep['rows']['truth']} issues={check_result(res, keys, primary, spec)} warn={res.warnings}")
    for k, v in sorted(rep["per_field"].items(), key=lambda kv: kv[1]):
        if v < 1: print(f"     {k:28s} {v:.3f}")
    if rep["hallucinations"]: print("     halluc:", rep["hallucinations"][:6])
    if rep["misses"]: print("     misses:", rep["misses"][:6])
