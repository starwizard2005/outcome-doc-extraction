"""SQLite-backed job store. This is the source of truth for job state.

Design points (see DESIGN.md):
* One connection per thread, WAL mode, explicit BEGIN IMMEDIATE for every write, so
  several worker threads (or several containers sharing the volume) never double-claim.
* Terminal states are immutable: finish() only updates rows that are still PENDING/RUNNING.
* Crash safety: a RUNNING job carries a lease that its worker keeps extending. If the worker
  dies, the lease expires and requeue_expired() puts the job back (or fails it after
  MAX_ATTEMPTS, so a poison document cannot loop forever).
* Idempotency: (org, dedupe_key = sha256(file) + sha256(canonical schema)) is unique among
  non-FAILED jobs, so resubmitting returns the existing job and never redoes the work.
* Tenancy: every read is scoped by org; a mismatch looks exactly like "not found".
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

TERMINAL = ("SUCCESS", "PARTIAL", "FAILED")

_DDL = """
CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    org          TEXT NOT NULL DEFAULT '',
    dedupe_key   TEXT NOT NULL,
    status       TEXT NOT NULL,
    stage        TEXT,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    filename     TEXT,
    file_path    TEXT NOT NULL,
    schema_json  TEXT NOT NULL,
    webhook_url  TEXT,
    result_json  TEXT,
    error_json   TEXT,
    attempts     INTEGER NOT NULL DEFAULT 0,
    lease_until  REAL,
    worker       TEXT,
    webhook_done INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_jobs_claim ON jobs(status, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_dedupe ON jobs(org, dedupe_key) WHERE status != 'FAILED';
"""


class JobStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self._conn() as c:
            c.executescript(_DDL)

    # ------------------------------------------------------------ plumbing
    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.db_path, timeout=30, isolation_level=None, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA busy_timeout=30000")
            self._local.conn = c
        return c

    class _Tx:
        def __init__(self, conn: sqlite3.Connection):
            self.c = conn

        def __enter__(self) -> sqlite3.Connection:
            self.c.execute("BEGIN IMMEDIATE")
            return self.c

        def __exit__(self, et, ev, tb) -> None:
            self.c.execute("ROLLBACK" if et else "COMMIT")

    def _tx(self) -> "JobStore._Tx":
        return JobStore._Tx(self._conn())

    # ------------------------------------------------------------ writes
    def create_or_get(self, *, org: str, dedupe_key: str, filename: str, file_path: str,
                      schema_json: str, webhook_url: str | None) -> tuple[dict, bool]:
        """Insert a PENDING job unless an equivalent non-failed job already exists."""
        now = time.time()
        with self._tx() as c:
            row = c.execute("SELECT * FROM jobs WHERE org=? AND dedupe_key=? AND status != 'FAILED'",
                            (org, dedupe_key)).fetchone()
            if row:
                return dict(row), False
            jid = uuid.uuid4().hex
            c.execute(
                "INSERT INTO jobs (id, org, dedupe_key, status, stage, created_at, updated_at, filename, file_path,"
                " schema_json, webhook_url) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (jid, org, dedupe_key, "PENDING", None, now, now, filename, file_path, schema_json, webhook_url))
            return dict(c.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()), True

    def claim_next(self, worker: str, lease_s: float) -> dict | None:
        now = time.time()
        with self._tx() as c:
            row = c.execute("SELECT id FROM jobs WHERE status='PENDING' ORDER BY created_at LIMIT 1").fetchone()
            if not row:
                return None
            c.execute("UPDATE jobs SET status='RUNNING', stage='STARTING', attempts=attempts+1, worker=?,"
                      " lease_until=?, updated_at=? WHERE id=? AND status='PENDING'",
                      (worker, now + lease_s, now, row["id"]))
            return dict(c.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone())

    def heartbeat(self, job_id: str, worker: str, lease_s: float) -> bool:
        now = time.time()
        with self._tx() as c:
            cur = c.execute("UPDATE jobs SET lease_until=?, updated_at=? WHERE id=? AND status='RUNNING' AND worker=?",
                            (now + lease_s, now, job_id, worker))
            return cur.rowcount == 1

    def set_stage(self, job_id: str, stage: str) -> None:
        with self._tx() as c:
            c.execute("UPDATE jobs SET stage=?, updated_at=? WHERE id=? AND status='RUNNING'",
                      (stage, time.time(), job_id))

    def finish(self, job_id: str, status: str, *, result: dict | None = None, error: dict | None = None) -> bool:
        """Move to a terminal state. Returns False (and changes nothing) if already terminal."""
        assert status in TERMINAL
        with self._tx() as c:
            cur = c.execute(
                "UPDATE jobs SET status=?, stage=NULL, result_json=?, error_json=?, lease_until=NULL, updated_at=?"
                " WHERE id=? AND status IN ('PENDING','RUNNING')",
                (status, json.dumps(result) if result is not None else None,
                 json.dumps(error) if error is not None else None, time.time(), job_id))
            return cur.rowcount == 1

    def requeue(self, job_id: str) -> bool:
        with self._tx() as c:
            cur = c.execute("UPDATE jobs SET status='PENDING', stage=NULL, lease_until=NULL, worker=NULL, updated_at=?"
                            " WHERE id=? AND status='RUNNING'", (time.time(), job_id))
            return cur.rowcount == 1

    def requeue_expired(self, max_attempts: int) -> tuple[int, int]:
        """Recover jobs whose worker vanished. Returns (requeued, failed)."""
        now = time.time()
        requeued = failed = 0
        with self._tx() as c:
            rows = c.execute("SELECT id, attempts FROM jobs WHERE status='RUNNING' AND lease_until < ?", (now,)).fetchall()
            for r in rows:
                if r["attempts"] >= max_attempts:
                    c.execute("UPDATE jobs SET status='FAILED', stage=NULL, lease_until=NULL, updated_at=?, error_json=?"
                              " WHERE id=?", (now, json.dumps({"code": "worker_lost",
                                                                "message": f"worker lost {r['attempts']} times; giving up"}), r["id"]))
                    failed += 1
                else:
                    c.execute("UPDATE jobs SET status='PENDING', stage=NULL, lease_until=NULL, worker=NULL, updated_at=?"
                              " WHERE id=?", (now, r["id"]))
                    requeued += 1
        return requeued, failed

    def mark_webhook_done(self, job_id: str) -> None:
        with self._tx() as c:
            c.execute("UPDATE jobs SET webhook_done=1 WHERE id=?", (job_id,))

    # ------------------------------------------------------------ reads
    def get(self, job_id: str, org: str) -> dict | None:
        row = self._conn().execute("SELECT * FROM jobs WHERE id=? AND org=?", (job_id, org)).fetchone()
        return dict(row) if row else None

    def get_any(self, job_id: str) -> dict | None:
        row = self._conn().execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def count(self, status: str) -> int:
        return self._conn().execute("SELECT COUNT(*) FROM jobs WHERE status=?", (status,)).fetchone()[0]

    def pending_webhooks(self) -> list[dict]:
        rows = self._conn().execute("SELECT * FROM jobs WHERE webhook_url IS NOT NULL AND webhook_done=0 AND status IN"
                                    " ('SUCCESS','PARTIAL','FAILED')").fetchall()
        return [dict(r) for r in rows]


def to_api(job: dict, include_result: bool = True) -> dict[str, Any]:
    """Row -> the JobStatus object from openapi.yaml."""
    from datetime import datetime, timezone

    def iso(t: float) -> str:
        return datetime.fromtimestamp(t, tz=timezone.utc).isoformat().replace("+00:00", "Z")

    out: dict[str, Any] = {
        "job_id": job["id"],
        "status": job["status"],
        "stage": job["stage"],
        "created_at": iso(job["created_at"]),
        "updated_at": iso(job["updated_at"]),
    }
    if job["status"] in ("SUCCESS", "PARTIAL") and include_result and job["result_json"]:
        out["result"] = json.loads(job["result_json"])
    if job["status"] == "FAILED":
        out["error"] = json.loads(job["error_json"]) if job["error_json"] else {"code": "failed", "message": "job failed"}
    return out
