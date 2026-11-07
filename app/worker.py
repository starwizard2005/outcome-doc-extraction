"""Background workers. Embedded in the API process (threads) and also runnable standalone
(`python -m app.worker`) so extra worker containers can share the same data volume.

Failure model
* Pipeline raises ValueError/UnsupportedFile      -> FAILED (bad input; retrying cannot help)
* BudgetExceeded / DeadlineExceeded               -> FAILED with a precise code
* LLMError / any other exception                  -> re-queued with backoff up to MAX_ATTEMPTS, then FAILED
* worker process dies mid-job                     -> lease expires; the reaper re-queues it (or fails it after MAX_ATTEMPTS)
The 30-minute-per-document limit counts queue time, so the pipeline deadline is created_at + JOB_DEADLINE_S.
"""
from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
import traceback
import uuid
from pathlib import Path

import httpx

from .config import Settings, get_settings
from .extract.ctx import JobContext
from .extract.pipeline import run_extraction
from .llm import BudgetExceeded, CostTracker, DeadlineExceeded, LLMError
from .parse import UnsupportedFile
from .store import JobStore, to_api

log = logging.getLogger("worker")


class WorkerPool:
    def __init__(self, store: JobStore, settings: Settings | None = None):
        self.store = store
        self.s = settings or get_settings()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.node = f"{os.uname().nodename}-{os.getpid()}-{uuid.uuid4().hex[:4]}"

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        rq, failed = self.store.requeue_expired(self.s.max_attempts)
        # on a fresh process start nothing can legitimately still be running here, but another node might own
        # the lease: only expired leases are touched.
        if rq or failed:
            log.warning("recovered jobs from a previous run", extra={"requeued": rq, "failed": failed})
        for i in range(self.s.workers):
            self._spawn(self._loop, f"worker-{i}", f"{self.node}/{i}")
        self._spawn(self._reaper, "reaper")
        self._spawn(self._webhooks, "webhooks")
        log.info("worker pool started", extra={"workers": self.s.workers, "node": self.node})

    def stop(self) -> None:
        self._stop.set()

    def _spawn(self, fn, name: str, *args) -> None:
        t = threading.Thread(target=fn, args=args, name=name, daemon=True)
        t.start()
        self._threads.append(t)

    # ------------------------------------------------------------ loops
    def _loop(self, wid: str) -> None:
        while not self._stop.is_set():
            try:
                job = self.store.claim_next(wid, self.s.lease_s)
            except Exception:  # noqa: BLE001
                log.exception("claim failed")
                time.sleep(2)
                continue
            if not job:
                time.sleep(0.4 + random.random() * 0.3)
                continue
            self._process(job, wid)

    def _reaper(self) -> None:
        while not self._stop.wait(15):
            try:
                rq, failed = self.store.requeue_expired(self.s.max_attempts)
                if rq or failed:
                    log.warning("reaper recovered stalled jobs", extra={"requeued": rq, "failed": failed})
            except Exception:  # noqa: BLE001
                log.exception("reaper error")

    def _webhooks(self) -> None:
        while not self._stop.wait(5):
            try:
                for job in self.store.pending_webhooks():
                    if self._post_webhook(job):
                        self.store.mark_webhook_done(job["id"])
            except Exception:  # noqa: BLE001
                log.exception("webhook loop error")

    @staticmethod
    def _post_webhook(job: dict) -> bool:
        url = job["webhook_url"]
        if not url.lower().startswith(("http://", "https://")):
            return True                      # nothing sensible to do; do not retry forever
        try:
            r = httpx.post(url, json=to_api(job), timeout=15)
            ok = r.status_code < 500
            log.info("webhook delivered", extra={"job_id": job["id"], "status": r.status_code})
            return ok
        except Exception as e:  # noqa: BLE001
            log.warning("webhook failed, will retry", extra={"job_id": job["id"], "error": str(e)[:120]})
            return False

    # ------------------------------------------------------------ one job
    def _heartbeat(self, job_id: str, wid: str, stop: threading.Event) -> None:
        while not stop.wait(self.s.lease_s / 3):
            try:
                if not self.store.heartbeat(job_id, wid, self.s.lease_s):
                    log.warning("lost lease", extra={"job_id": job_id})
                    return
            except Exception:  # noqa: BLE001
                log.exception("heartbeat failed", extra={"job_id": job_id})

    def _process(self, job: dict, wid: str) -> None:
        jid = job["id"]
        t0 = time.time()
        log.info("job start", extra={"job_id": jid, "attempt": job["attempts"], "file": job["filename"], "worker": wid})
        hb_stop = threading.Event()
        threading.Thread(target=self._heartbeat, args=(jid, wid, hb_stop), daemon=True).start()
        ctx = JobContext(job_id=jid, tracker=CostTracker(), deadline=job["created_at"] + self.s.job_deadline_s,
                         stage_cb=lambda st: self.store.set_stage(jid, st))
        try:
            schema = json.loads(job["schema_json"])
            result, status = run_extraction(Path(job["file_path"]), schema, ctx)
            if self.store.finish(jid, status, result=result):
                log.info("job done", extra={"job_id": jid, "status": status, "seconds": round(time.time() - t0, 1),
                                            "cost_usd": result["metadata"]["cost_usd"], "calls": result["metadata"]["model_calls"],
                                            "entries": len(result["entries"])})
            else:
                log.warning("job result discarded: already terminal", extra={"job_id": jid})
        except (UnsupportedFile, ValueError, json.JSONDecodeError) as e:
            self._fail(jid, "invalid_input", str(e), traceback.format_exc())
        except BudgetExceeded as e:
            self._fail(jid, "budget_exceeded", str(e), None)
        except DeadlineExceeded as e:
            self._fail(jid, "deadline_exceeded", str(e), None)
        except Exception as e:  # noqa: BLE001  (LLMError and everything unexpected)
            tb = traceback.format_exc()
            kind = "llm_unavailable" if isinstance(e, LLMError) else "internal_error"
            if job["attempts"] < self.s.max_attempts and ctx.time_left() > 90:
                log.warning("job failed, re-queueing", extra={"job_id": jid, "attempt": job["attempts"], "error": str(e)[:300],
                                                              "kind": kind}, exc_info=True)
                time.sleep(min(30, 3 * 2 ** job["attempts"]) * (0.5 + random.random() / 2))
                self.store.requeue(jid)
            else:
                self._fail(jid, kind, str(e), tb)
        finally:
            hb_stop.set()

    def _fail(self, jid: str, code: str, msg: str, tb: str | None) -> None:
        log.error("job failed", extra={"job_id": jid, "code": code, "error": msg[:500], "trace": (tb or "")[-1500:]})
        self.store.finish(jid, "FAILED", error={"code": code, "message": msg[:500]})


def main() -> None:     # standalone worker process
    from .logging_setup import setup_logging
    setup_logging()
    s = get_settings()
    pool = WorkerPool(JobStore(s.data_dir / "jobs.db"), s)
    pool.start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pool.stop()


if __name__ == "__main__":
    main()
