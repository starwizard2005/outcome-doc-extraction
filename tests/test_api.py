"""API + worker + store behaviour with the scripted fake LLM (no network)."""
import json
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SAMPLE = ROOT / "assignment" / "sample"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("WORKERS", "2")
    monkeypatch.setenv("MAX_PENDING", "50")
    from app import config
    config.reset_settings_cache()
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c
    config.reset_settings_cache()


def _wait(client, jid, org="", timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = client.get(f"/jobs/{jid}", headers={"X-Organization-Id": org})
        assert r.status_code == 200
        if r.json()["status"] in ("SUCCESS", "PARTIAL", "FAILED"):
            return r.json()
        time.sleep(0.2)
    raise AssertionError("timeout")


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_bad_requests(client):
    schema = json.dumps({"fields": {"a": {"description": "x"}}})
    assert client.post("/jobs", files={"file": ("a.pdf", b"")}, data={"schema": schema}).status_code == 400
    assert client.post("/jobs", files={"file": ("a.txt", b"hello world")}, data={"schema": schema}).status_code == 400
    assert client.post("/jobs", files={"file": ("a.csv", b"a,b\n1,2\n")}, data={"schema": "{bad"}).status_code == 400
    assert client.post("/jobs", files={"file": ("a.csv", b"a,b\n1,2\n")}, data={"schema": "{}"}).status_code == 400
    assert client.get("/jobs/nope").status_code == 404


def test_tenant_isolation_and_dedupe(client):
    from tests import fake_llm
    fake_llm.install()
    schema = json.dumps({"fields": {"a": {"description": "col a"}}})
    f = {"file": ("a.csv", b"a,b\n1,2\n")}
    r1 = client.post("/jobs", files=f, data={"schema": schema}, headers={"X-Organization-Id": "orgA"})
    assert r1.status_code == 202
    jid = r1.json()["job_id"]
    assert client.get(f"/jobs/{jid}", headers={"X-Organization-Id": "orgB"}).status_code == 404
    assert client.get(f"/jobs/{jid}", headers={"X-Organization-Id": "orgA"}).status_code == 200
    r2 = client.post("/jobs", files={"file": ("a.csv", b"a,b\n1,2\n")}, data={"schema": schema}, headers={"X-Organization-Id": "orgA"})
    assert r2.json()["job_id"] == jid                       # same org + same bytes + same schema -> same job
    r3 = client.post("/jobs", files={"file": ("a.csv", b"a,b\n1,2\n")}, data={"schema": schema}, headers={"X-Organization-Id": "orgB"})
    assert r3.json()["job_id"] != jid                       # other tenant never shares a job


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample data not available")
def test_end_to_end_boxscore(client):
    from tests import fake_llm
    fake_llm.install()
    fake_llm.CURRENT["doc_id"] = "boxscore_001"
    schema = (SAMPLE / "schemas" / "boxscore.json").read_text()
    with open(SAMPLE / "sample-data-documents" / "boxscore_001.csv", "rb") as fh:
        r = client.post("/jobs", files={"file": ("boxscore_001.csv", fh)}, data={"schema": schema})
    assert r.status_code == 202 and r.json()["status"] == "PENDING"
    done = _wait(client, r.json()["job_id"])
    assert done["status"] in ("SUCCESS", "PARTIAL")
    assert set(done["result"]) == {"entries", "metadata"}
    assert len(done["result"]["entries"]) == 10


def test_store_terminal_immutable_and_requeue(tmp_path):
    from app.store import JobStore
    st = JobStore(tmp_path / "j.db")
    job, created = st.create_or_get(org="o", dedupe_key="k", filename="f", file_path="/x", schema_json="{}", webhook_url=None)
    assert created
    c = st.claim_next("w1", lease_s=0.01)
    assert c and c["id"] == job["id"] and c["attempts"] == 1
    time.sleep(0.05)
    assert st.requeue_expired(3) == (1, 0)
    c = st.claim_next("w2", lease_s=60)
    assert c["attempts"] == 2
    assert st.finish(job["id"], "SUCCESS", result={"entries": [], "metadata": {}})
    assert not st.finish(job["id"], "FAILED", error={"code": "x", "message": "y"})   # immutable
    assert st.get(job["id"], "o")["status"] == "SUCCESS"


def test_store_gives_up_after_max_attempts(tmp_path):
    from app.store import JobStore
    st = JobStore(tmp_path / "j.db")
    job, _ = st.create_or_get(org="", dedupe_key="k", filename="f", file_path="/x", schema_json="{}", webhook_url=None)
    for _ in range(2):
        st.claim_next("w", lease_s=0.01)
        time.sleep(0.03)
        st.requeue_expired(2)
    assert st.get(job["id"], "")["status"] == "FAILED"
