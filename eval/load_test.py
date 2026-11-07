"""Load + contract test against a RUNNING service (docker compose up -d).

  docker compose exec app python -m eval.load_test [--base http://localhost:8080] [--n 10]

Submits N documents at once (the 8 samples, then repeats under a different X-Organization-Id so they are not
deduplicated), polls every 2 s like the grading harness, and reports per-job latency, throughput, scores
against the validated answers, plus checks: /health latency under load, tenant isolation (404), duplicate
submission returns the same job, and validation errors (400). Costs real money when LLM_CACHE=0.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8080")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    base_dir = next((c for c in (ROOT / "assignment", ROOT) if (c / "sample" / "manifest.csv").exists()), None)
    if base_dir is None:
        sys.exit("sample data not found")
    sample = base_dir / "sample"
    sys.path.insert(0, str(base_dir / "tools"))
    import validate as V

    rows = list(csv.DictReader(open(sample / "manifest.csv", encoding="utf-8")))
    jobs = [(rows[i % len(rows)], f"org-{i // len(rows)}") for i in range(a.n)]
    cli = httpx.Client(base_url=a.base, timeout=60)

    t = time.time()
    assert cli.get("/health").status_code == 200
    print(f"health ok in {(time.time() - t) * 1000:.0f} ms")

    def submit(item):
        r, org = item
        schema = (sample / "schemas" / f"{r['schema_name']}.json").read_text()
        t0 = time.time()
        with open(sample / "sample-data-documents" / r["file"], "rb") as fh:
            resp = cli.post("/jobs", files={"file": (r["file"], fh)}, data={"schema": schema}, headers={"X-Organization-Id": org})
        return r, org, resp, time.time() - t0

    t_start = time.time()
    with ThreadPoolExecutor(len(jobs)) as ex:
        subs = list(ex.map(submit, jobs))
    ack = [s[3] for s in subs]
    assert all(s[2].status_code == 202 for s in subs), [s[2].status_code for s in subs]
    print(f"{len(subs)} submissions acknowledged (202) in max {max(ack):.2f}s, median {statistics.median(ack):.2f}s")

    pending = {s[2].json()["job_id"]: s for s in subs}
    done: dict[str, tuple[dict, float]] = {}
    health_ms = []
    while pending:
        for jid, (r, org, _, _) in list(pending.items()):
            g = cli.get(f"/jobs/{jid}", headers={"X-Organization-Id": org})
            if g.status_code == 200 and g.json()["status"] in ("SUCCESS", "PARTIAL", "FAILED"):
                done[jid] = (g.json(), time.time() - t_start)
                pending.pop(jid)
        h0 = time.time()
        cli.get("/health")
        health_ms.append((time.time() - h0) * 1000)
        time.sleep(2)
        if time.time() - t_start > 1700:
            break
    wall = time.time() - t_start

    print("\n| doc | org | status | seconds | score | $ |\n|---|---|---|---|---|---|")
    lat = []
    for jid, (body, secs) in done.items():
        r, org, _, _ = pending.get(jid) or next(s for s in subs if s[2].json()["job_id"] == jid)
        score, cost = "-", "-"
        if body["status"] != "FAILED":
            truth = json.loads((sample / "sample-data-output-validated" / f"{r['document_id']}.json").read_text())
            schema = json.loads((sample / "schemas" / f"{r['schema_name']}.json").read_text())
            score = f"{V.score(body['result'], truth, schema)['score']:.3f}"
            cost = f"{body['result']['metadata'].get('cost_usd', 0):.4f}"
        else:
            score = body["error"]["code"]
        lat.append(secs)
        print(f"| {r['document_id']} | {org} | {body['status']} | {secs:.0f} | {score} | {cost} |")
    if pending:
        print(f"\nNOT FINISHED: {len(pending)} job(s)")
    if lat:
        lat.sort()
        print(f"\nwall {wall:.0f}s for {len(done)} docs ({len(done) / wall * 60:.1f} docs/min); latency p50 {statistics.median(lat):.0f}s,"
              f" p95 {lat[min(len(lat) - 1, int(len(lat) * 0.95))]:.0f}s, max {lat[-1]:.0f}s")
    print(f"/health during load: median {statistics.median(health_ms):.0f} ms, max {max(health_ms):.0f} ms")

    # ---- contract checks
    first_id = subs[0][2].json()["job_id"]
    r0, org0 = subs[0][0], subs[0][1]
    assert cli.get(f"/jobs/{first_id}", headers={"X-Organization-Id": "someone-else"}).status_code == 404, "tenant isolation"
    schema = (sample / "schemas" / f"{r0['schema_name']}.json").read_text()
    with open(sample / "sample-data-documents" / r0["file"], "rb") as fh:
        dup = cli.post("/jobs", files={"file": (r0["file"], fh)}, data={"schema": schema}, headers={"X-Organization-Id": org0})
    assert dup.json()["job_id"] == first_id, "duplicate submission should return the same job"
    assert cli.post("/jobs", files={"file": ("x.pdf", b"not a pdf")}, data={"schema": schema}).status_code == 400
    assert cli.post("/jobs", files={"file": ("x.csv", b"a,b\n1,2")}, data={"schema": "{oops"}).status_code == 400
    print("contract checks passed: tenant isolation 404, dedupe returns same job, 400 on bad file / bad schema")
    bad = [j for j, (b, _) in done.items() if b["status"] == "FAILED"]
    sys.exit(1 if bad or pending else 0)


if __name__ == "__main__":
    main()
