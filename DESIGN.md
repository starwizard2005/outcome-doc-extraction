# DESIGN

## What it does and the one idea behind it
The field list decides the shape: a schema with a `primary` field is a **table** (one entry per row), otherwise a
**record** (one entry). Models are used where judgement is needed; everything that can be exact is code.

**Tables in spreadsheets (XLSX/CSV).** The model never reads rows. It sees a compact "skeleton" of the sheet (headers,
a few sample rows, merged cells, block structure; about 1-2k tokens) and writes a small declarative spec (columns,
row filters, block layout). A deterministic executor applies it to every row and checks the result. One cheap
Haiku call per document (~$0.003) and the 325-row rent roll and 119-row operating statement score 1.000 / 0.997.
**Tables in PDFs/DOCX.** Haiku plans the row anchor; code segments rows (including wrapped commentary), Haiku
converts batches of segments; a chunked fallback exists if the plan fails.
**Records (leases, offering memoranda).** Pass 1: Haiku extracts all fields at once with a verbatim quote and
confidence each. Deterministic checks then run on every field: is the quote really in the text (fuzzy), is the value in
the quote, does it coerce to the declared type. Only failing or low-confidence fields go to pass 2, where Sonnet re-reads
the relevant pages. Derived fields (term in months, total base rent, rent per RSF, price per unit) are recomputed in code
from the extracted inputs using conventions I inferred from the answer keys. For scanned PDFs (tesseract, open source) a
cheap vision model (GLM-5.3-flash) checks cited values against the page image and fills fields OCR left empty.
Null discipline: no quote means lower confidence, an unsupported value is re-read or nulled, never kept on faith.
Field descriptions are not trusted: types are inferred from them and the prompts add the conventions
(null over a guess, verbatim text, numbers as printed). Nothing depends on a particular field list.

## Models and spend
Haiku 5.5 for spec writing, planning, pass 1 and batch conversion; Sonnet 5.5 only for doubtful record fields and as
fallback; GLM-5.3-flash for page images. Cost per document, cache off: tables $0.003-0.010, leases $0.07-0.13,
OM $0.04; **$0.045 average, $0.357 for the 8 samples**. Total spend of the $10 across all development runs: **$2.03 of the $10** (the service has a spend cap and a persisted ledger; cached re-runs are free, which is what made iterating cheap).
I did not compare models on every document; routing was chosen by field difficulty and checked against the validator.

## Service
`POST /jobs` stores the file, returns 202 and enqueues. Jobs live in SQLite (WAL) on a volume. Workers claim a job with a
lease and heartbeat; a reaper re-queues expired leases (a crashed worker loses nothing, after `MAX_ATTEMPTS` the job
fails with a reason). Terminal states are immutable (checked in SQL). Identical file + field list for one organisation
returns the same job. `X-Organization-Id` scopes every read (404 across tenants). The LLM client retries with backoff,
honours Retry-After, and pauses all callers on a 429, so throttling slows jobs instead of failing them. The per-job deadline
counts from submission. Structured JSON logs carry the job id, stage, cost and failure reason. Scaling: `WORKERS`
per container; OCR concurrency is capped at the CPU count so scanned documents stay CPU-bound but do not thrash.

## What broke
- **OCR timeouts.** First real run: tesseract's OpenMP threads multiplied by parallel pages and jobs made pages exceed
  240 s and the two scanned leases lost most of their text (scores 0.80 / 0.23, 20 minutes each). Fixed with one thread
  per process, a process-wide cap and a retry: those documents now take 2-3.5 minutes.
- **Type inference bug.** "aligned with the start dates" made the rent list type "dates", so every rent value was
  discarded (empty lists with correct quotes). I only found it by reading the per-cell diffs. The re-read trigger now also
  fires on "cited evidence but no value".
- **Malformed JSON** from the cheap model forced an expensive fallback once; added a repair step.
- **Answer-key disagreement.** budget_001 line 7650-0000: the PDF prints the commentary in sequence, the key moves it.
  I follow the document. It costs 0.001 and I left it.

## Known gaps (honest)
- Free-text fields (right to sublet, electricity, late fee, termination fee) are scored on exact wording, and the keys
  paraphrase the clause. These are most of the remaining lease misses, and I did not tune prompts to three leases.
- lease_001's handwritten security deposit (2150) is still missed: OCR garbles it and the vision fill did not recover it.
- Single-record scores move about ±0.03 between runs (temperature 0 does not make the providers deterministic).
- SQLite on one volume means one node; the schema is ready for Postgres plus a shared queue but I did not build it.

## With another week
Per-schema description refinement driven by the eval loop (learn the wording style of the keys, then check on held-out
documents); a self-consistency vote on low-confidence record fields; vision for table pages that have no text layer;
a PDF-native layout step for tables with merged cells; Postgres + a queue for multi-node scale; a metrics endpoint
(cost, latency per stage, 429 rate); a larger set of my own labelled documents so I am not tuning on eight.
