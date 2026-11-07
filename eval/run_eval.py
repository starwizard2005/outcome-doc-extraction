"""Run the pipeline in-process over the sample documents with the REAL gateway and score against the
validated answers. Writes:
  outputs/<doc_id>.json        submission files (with the top-level "schema" key)
  eval_out/report.md           per-document score, cost, latency, weakest fields
  eval_out/diffs_<doc>.txt     every wrong cell: expected vs got + the model's citation (for tuning)
  eval_out/summary.json
Usage (inside the container):  python -m eval.run_eval [--docs lease_001,om_001] [--parallel 4] [--no-cache]
Models can be overridden per run with MODEL_SPEC / MODEL_RECORD / MODEL_STRONG / MODEL_VISION / MODEL_CHEAP env vars.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def find_assignment() -> Path:
    for c in (ROOT / "assignment", ROOT):
        if (c / "sample" / "manifest.csv").exists():
            return c
    sys.exit("sample/manifest.csv not found (expected ./assignment/sample or ./sample)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="", help="comma-separated document ids (default: all)")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    if a.no_cache:
        os.environ["LLM_CACHE"] = "0"

    from app.config import get_settings
    from app.logging_setup import setup_logging
    setup_logging()
    os.environ.setdefault("LOG_LEVEL", "WARNING")
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    s = get_settings()
    if not (s.llm_base_url and s.llm_api_key):
        sys.exit("LLM_BASE_URL / LLM_API_KEY not set (put them in .env)")

    base = find_assignment()
    sample = base / "sample"
    sys.path.insert(0, str(base / "tools"))
    import validate as V
    from app.extract.ctx import JobContext
    from app.extract.pipeline import run_extraction
    from app.llm import LEDGER, CostTracker

    rows = list(csv.DictReader(open(sample / "manifest.csv", encoding="utf-8")))
    want = {x for x in a.docs.split(",") if x}
    rows = [r for r in rows if not want or r["document_id"] in want]
    out_dir, ev = ROOT / a.out, ROOT / "eval_out"
    out_dir.mkdir(exist_ok=True)
    ev.mkdir(exist_ok=True)

    def run(r: dict) -> dict:
        doc_id, schema_name = r["document_id"], r["schema_name"]
        schema = json.loads((sample / "schemas" / f"{schema_name}.json").read_text())
        ctx = JobContext(job_id=doc_id, tracker=CostTracker(), deadline=time.time() + 1500)
        t0 = time.time()
        try:
            result, status = run_extraction(sample / "sample-data-documents" / r["file"], schema, ctx)
        except Exception as e:  # noqa: BLE001
            import traceback
            return {"doc": doc_id, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
        dt = time.time() - t0
        (out_dir / f"{doc_id}.json").write_text(json.dumps({"schema": schema_name, **result}, indent=2, ensure_ascii=False))
        truth = json.loads((sample / "sample-data-output-validated" / f"{doc_id}.json").read_text())
        errs = V.validate_contract(result, schema)
        rep = V.score(result, truth, schema)
        # per-cell diffs
        fields, primary = V.schema_fields(schema)
        lines = []
        for pe, te, label in V.align_rows(result["entries"], truth.get("entries", []), primary):
            if pe is None or te is None:
                lines.append(f"ROW {'MISSING' if pe is None else 'EXTRA'}: {label}")
                continue
            for k in fields:
                tv, pv = V.value_of(te, k), V.value_of(pe, k)
                if V.cell_match(pv, tv) < 1.0:
                    cit = (pe.get(k) or {}).get("citation") if isinstance(pe.get(k), dict) else None
                    conf = (pe.get(k) or {}).get("confidence") if isinstance(pe.get(k), dict) else None
                    lines.append(f"[{label}] {k}\n    expected: {json.dumps(tv, ensure_ascii=False)[:300]}\n"
                                 f"    got     : {json.dumps(pv, ensure_ascii=False)[:300]}  (conf={conf})\n"
                                 f"    citation: {(cit or '')[:200]!r}")
        (ev / f"diffs_{doc_id}.txt").write_text("\n".join(lines) or "no differences")
        md = result["metadata"]
        return {"doc": doc_id, "schema": schema_name, "status": status, "score": rep["score"], "contract_errors": errs[:5],
                "rows": rep["rows"], "halluc": len(rep["hallucinations"]), "misses": len(rep["misses"]),
                "seconds": round(dt, 1), "cost_usd": md["cost_usd"], "calls": md["model_calls"],
                "weak": {k: v for k, v in rep["per_field"].items() if v < 1.0}, "warnings": md.get("warnings", [])[:8],
                "models": md.get("models")}

    t0 = time.time()
    with ThreadPoolExecutor(max(1, a.parallel)) as ex:
        results = list(ex.map(run, rows))
    wall = time.time() - t0

    ok = [r for r in results if "score" in r]
    lines = ["# Eval report", "", f"models: spec={s.model_spec} record={s.model_record} strong={s.model_strong} vision={s.model_vision}", "",
             "| doc | status | score | rows | halluc | miss | sec | $ | calls |", "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if "score" in r:
            lines.append(f"| {r['doc']} | {r['status']} | {r['score']:.3f} | {r['rows']['predicted']}/{r['rows']['truth']} | {r['halluc']} | "
                         f"{r['misses']} | {r['seconds']} | {r['cost_usd']:.4f} | {r['calls']} |")
        else:
            lines.append(f"| {r['doc']} | ERROR | - | - | - | - | - | - | {r['error'][:80]} |")
    if ok:
        lines += ["", f"mean score {sum(r['score'] for r in ok)/len(ok):.4f}; total cost ${sum(r['cost_usd'] for r in ok):.4f}; wall {wall:.0f}s; "
                      f"ledger ${LEDGER.total():.4f}"]
    for r in results:
        lines += ["", f"## {r['doc']}"]
        if "error" in r:
            lines += [r["error"], "```", r.get("trace", ""), "```"]
            continue
        lines += [f"weak fields: {r['weak']}", f"warnings: {r['warnings']}", f"contract errors: {r['contract_errors']}"]
    (ev / "report.md").write_text("\n".join(lines))
    (ev / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    print("\n".join(lines[:4 + len(results) + 3]))
    print(f"\nFull report: eval_out/report.md   Per-cell diffs: eval_out/diffs_*.txt   Outputs: {a.out}/")


if __name__ == "__main__":
    main()
