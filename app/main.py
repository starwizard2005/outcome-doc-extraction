"""HTTP API (contract/openapi.yaml): POST /jobs, GET /jobs/{id}, GET /health."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from .config import get_settings
from .logging_setup import setup_logging
from .parse import UnsupportedFile, sniff_kind
from .store import JobStore, to_api
from .worker import WorkerPool

setup_logging()
log = logging.getLogger("api")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict | None = None):
        self.status, self.code, self.message, self.headers = status, code, message, headers or {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    (s.data_dir / "uploads").mkdir(parents=True, exist_ok=True)
    app.state.store = JobStore(s.data_dir / "jobs.db")
    app.state.pool = None
    if os.environ.get("EMBED_WORKERS", "1") not in ("0", "false"):
        app.state.pool = WorkerPool(app.state.store, s)
        app.state.pool.start()
    log.info("service ready", extra={"version": s.version, "data_dir": str(s.data_dir),
                                     "llm_configured": bool(s.llm_base_url and s.llm_api_key)})
    yield
    if app.state.pool:
        app.state.pool.stop()


app = FastAPI(title="Document Extraction Service", version=get_settings().version, lifespan=lifespan)


@app.exception_handler(ApiError)
async def _api_error(_: Request, e: ApiError):
    return JSONResponse({"code": e.code, "message": e.message}, status_code=e.status, headers=e.headers)


def _org(x_organization_id: str | None) -> str:
    return (x_organization_id or "").strip()


def _validate_schema(raw: str) -> dict:
    try:
        schema = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ApiError(400, "invalid_schema", f"schema is not valid JSON: {e}")
    fields = schema.get("fields") if isinstance(schema, dict) else None
    if not isinstance(fields, dict) or not fields:
        raise ApiError(400, "invalid_schema", "schema must be an object with a non-empty 'fields' map")
    if sum(1 for v in fields.values() if isinstance(v, dict) and v.get("primary") is True) > 1:
        raise ApiError(400, "invalid_schema", "at most one field may be marked primary")
    return schema


@app.get("/health")
async def health():
    return {"status": "ok", "version": get_settings().version}


@app.post("/jobs", status_code=202)
async def create_job(request: Request, file: UploadFile = File(...), schema: str = Form(...),
                     webhook_url: str | None = Form(None),
                     x_organization_id: str | None = Header(None)):
    s = get_settings()
    store: JobStore = request.app.state.store
    org = _org(x_organization_id)
    schema_obj = _validate_schema(schema)

    # stream the upload to disk with a size cap, hashing as we go
    h = hashlib.sha256()
    tmp = s.data_dir / "uploads" / f".tmp-{os.getpid()}-{id(file)}"
    size = 0
    try:
        with open(tmp, "wb") as out:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > s.max_upload_bytes:
                    raise ApiError(413, "file_too_large", f"file exceeds {s.max_upload_bytes // (1 << 20)} MB")
                h.update(chunk)
                out.write(chunk)
        if size == 0:
            raise ApiError(400, "empty_file", "uploaded file is empty")
        try:
            kind = await run_in_threadpool(sniff_kind, tmp, file.filename or "")
        except UnsupportedFile as e:
            raise ApiError(400, "unsupported_file_type", str(e))
        ext = {"pdf": ".pdf", "docx": ".docx", "xlsx": ".xlsx", "csv": ".csv"}[kind]
        digest = h.hexdigest()
        final = s.data_dir / "uploads" / f"{digest}{ext}"
        if not final.exists():
            tmp.replace(final)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)

    canonical = json.dumps(schema_obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    dedupe_key = hashlib.sha256((digest + "|" + canonical).encode()).hexdigest()
    if await run_in_threadpool(store.count, "PENDING") >= s.max_pending:
        raise ApiError(429, "queue_full", "too many pending jobs; retry shortly", {"Retry-After": "5"})
    job, created = await run_in_threadpool(
        lambda: store.create_or_get(org=org, dedupe_key=dedupe_key, filename=re.sub(r"[^\w.\- ]", "_", file.filename or "upload"),
                                    file_path=str(final), schema_json=canonical, webhook_url=webhook_url))
    log.info("job accepted" if created else "duplicate submission", extra={"job_id": job["id"], "org": org, "kind": kind, "bytes": size})
    return to_api(job)


@app.get("/jobs/{job_id}")
async def get_job(request: Request, job_id: str, x_organization_id: str | None = Header(None)):
    job = await run_in_threadpool(request.app.state.store.get, job_id, _org(x_organization_id))
    if not job:
        raise ApiError(404, "not_found", "job not found")
    return to_api(job)
