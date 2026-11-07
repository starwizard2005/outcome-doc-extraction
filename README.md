# Document Extraction Service

**Role applied for: AI/ML Developer.** Candidate: Achyuth Harikrishna Warrier, roll no. 230052.

A service that takes one document (PDF, DOCX, XLSX or CSV) and a field list, and returns the extracted values,
with citations and confidence, in the contract format. See [DESIGN.md](DESIGN.md) for the decisions and trade-offs.

## Run it

```bash
cp .env.example .env          # then set LLM_BASE_URL and LLM_API_KEY  (or export them in your shell)
docker compose up --build     # service on http://localhost:8080, /health answers within seconds
```

```bash
curl localhost:8080/health
curl -X POST localhost:8080/jobs -H "X-Organization-Id: acme" \
     -F file=@lease.pdf -F schema=@schema.json          # -> 202 {"job_id": ..., "status": "PENDING"}
curl localhost:8080/jobs/<job_id> -H "X-Organization-Id: acme"   # PENDING | RUNNING | SUCCESS | PARTIAL | FAILED
```

Only the models on the assignment's allow-list can be called (enforced in `app/llm.py`), and only through
`LLM_BASE_URL` / `LLM_API_KEY`. Nothing is hard-coded.

### Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `WORKERS` | 6 | background worker threads per container (more workers = more documents in parallel) |
| `LLM_CONCURRENCY` | 12 | concurrent model calls, process-wide (a 429 from the provider pauses all callers) |
| `JOB_DEADLINE_S` | 1560 | per-job budget counted from submission, so queue time counts (limit is 30 min) |
| `MAX_ATTEMPTS` / `LEASE_S` | 3 / 120 | retry budget; a job not heartbeating for `LEASE_S` is re-queued |
| `MAX_PENDING`, `MAX_UPLOAD_MB` | 500, 50 | back-pressure (429 + Retry-After) and upload cap (413) |
| `MODEL_SPEC`, `MODEL_RECORD`, `MODEL_STRONG`, `MODEL_VISION` | haiku-5.5, haiku-5.5, sonnet-5.5, glm-5.3-flash | model routing |
| `MAX_SPEND_USD` | 0 (off) | development spend cap; the service refuses further model calls past it |
| `LLM_CACHE` | 1 | disk cache of model responses (set 0 to measure real cost) |

## Scores on the sample set

`python tools/validate.py outputs/ --schema-dir sample/schemas --truth-dir sample/sample-data-output-validated`
prints `VALID` for all 8 files in `outputs/`. Produced by `eval/run_eval.py` with the cache off, so the cost figures are real:

| document | kind | score | rows | cost (USD) | time |
|---|---|---|---|---|---|
| boxscore_001 | CSV table | 1.000 | 10/10 | 0.0020 | 11 s |
| budget_001 | PDF table | 0.999 | 55/55 | 0.0103 | 26 s |
| t12_001 | XLSX table | 0.997 | 119/119 | 0.0026 | 17 s |
| rentroll_001 | XLSX table | 1.000 | 325/325 | 0.0028 | 37 s |
| lease_001 | scanned PDF record | 0.850 | 1/1 | 0.0991 | 118 s |
| lease_002 | DOCX record | 0.850 | 1/1 | 0.0724 | 40 s |
| lease_003 | scanned PDF record | 0.850 | 1/1 | 0.1289 | 217 s |
| om_001 | PDF record | 0.966 | 1/1 | 0.0384 | 23 s |
| **mean** | | **0.939** | | **0.357 total (0.045 per document)** | |

Zero hallucinations (a value where the answer is null) on every document. Single-record scores vary by about
±0.03 between runs (lease_001 scored 0.85 to 0.90, lease_003 0.78 to 0.85 across runs); the table documents are stable.
Most remaining misses are free-text fields where the key is a particular wording of the clause (see DESIGN.md).

Load test (`eval/load_test.py`, 10 documents submitted at once, cache off): all 10 reached SUCCESS in 277 s wall,
p50 32 s, p95 275 s (the two scanned leases, which are OCR-bound), `/health` max 11 ms under load.

## Tests and tooling

```bash
pip install -r requirements.txt && python -m pytest tests -q        # 14 tests; those needing the sample package skip if it is absent
docker compose -f docker-compose.yml -f docker-compose.eval.yml run --rm app python -m eval.run_eval --no-cache   # real model run + scoring
docker compose -f docker-compose.yml -f docker-compose.eval.yml up -d
docker compose -f docker-compose.yml -f docker-compose.eval.yml exec app python -m eval.load_test --n 10        # load + contract checks
```

The offline tests drive the whole pipeline with a scripted stand-in for the model. They check plumbing, parsing,
normalisation, output shape, API behaviour (tenant isolation, dedupe, 400/404, terminal states, lease recovery),
not model quality; model quality is measured by `eval/run_eval.py`.

## Layout

```
app/main.py, worker.py, store.py     HTTP API, worker pool + reaper + webhooks, SQLite job store
app/llm.py                           the only place that talks to the gateway (allow-list, retries, cache, cost ledger)
app/parse/                           PDF (text layer + tesseract OCR), XLSX, CSV, DOCX -> common document model
app/extract/pipeline.py              routing: table vs record, assembling the contract output
app/extract/sheet_exec.py, spec_gen.py   spreadsheet tables: model writes a declarative spec, deterministic executor runs it
app/extract/text_tables.py           PDF/DOCX tables: plan, deterministic row segmentation, batched extraction
app/extract/records.py, verify.py, derive.py, vision.py   single-record documents: extract, verify, re-read, derive, look at the page
eval/                                eval runner (scores against the validated answers) and load test
tests/                               offline pipeline tests with a scripted fake model; API/store tests
```
