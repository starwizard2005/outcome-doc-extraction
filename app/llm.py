"""OpenAI-compatible chat client with the behaviour a production pipeline needs.

* Only models in config.ALLOWED_MODELS can be called (disqualification guard).
* All traffic goes through LLM_BASE_URL / LLM_API_KEY.
* Retries with exponential backoff + jitter on 429 / 5xx / network errors / empty
  or unparseable bodies, honouring Retry-After. A 429 pauses *every* thread for
  the advertised window instead of letting each one hammer the provider.
* A global semaphore bounds in-flight requests across all jobs.
* Responses are cached on disk (key = model + messages + params). A cache hit is
  free and is reported separately, so evals never silently hide real cost.
* Per-job cost accounting from the gateway's own `usage.cost` when present,
  otherwise tokens x listed price.
* Optional hard spend cap (MAX_SPEND_USD) persisted across runs.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .config import ALLOWED_MODELS, PRICES_PER_M, get_settings

log = logging.getLogger("llm")


class LLMError(Exception):
    """Unrecoverable LLM failure (after retries)."""


class BudgetExceeded(LLMError):
    pass


class DeadlineExceeded(LLMError):
    pass


# ---------------------------------------------------------------- accounting

@dataclass
class CostTracker:
    """Accumulates usage for one job. Thread-safe."""
    calls: int = 0
    cache_hits: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    cost_if_uncached_usd: float = 0.0
    per_model: dict[str, dict[str, float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, model: str, tin: int, tout: int, cost: float, cached: bool) -> None:
        with self._lock:
            self.cost_if_uncached_usd += cost
            if cached:
                self.cache_hits += 1
                return
            self.calls += 1
            self.tokens_in += tin
            self.tokens_out += tout
            self.cost_usd += cost
            m = self.per_model.setdefault(model, {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0})
            m["calls"] += 1
            m["tokens_in"] += tin
            m["tokens_out"] += tout
            m["cost_usd"] += cost

    def metadata(self) -> dict[str, Any]:
        with self._lock:
            return {
                "model_calls": self.calls,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "cost_usd": round(self.cost_usd, 6),
                "cache_hits": self.cache_hits,
                "cost_if_uncached_usd": round(self.cost_if_uncached_usd, 6),
                "models": {k: {**v, "cost_usd": round(v["cost_usd"], 6)} for k, v in self.per_model.items()},
            }


class _SpendLedger:
    """Persisted running total of real (non-cached) spend, for the dev budget cap."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def _path(self) -> Path:
        p = get_settings().data_dir / "spend.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def total(self) -> float:
        try:
            return float(json.loads(self._path().read_text()).get("total_usd", 0.0))
        except Exception:  # noqa: BLE001
            return 0.0

    def add(self, usd: float) -> float:
        with self._lock:
            t = self.total() + usd
            try:
                self._path().write_text(json.dumps({"total_usd": round(t, 6)}))
            except Exception:  # noqa: BLE001
                pass
            return t


LEDGER = _SpendLedger()


# ---------------------------------------------------------------- client

_global_pause_until = 0.0          # set when any thread sees a 429
_pause_lock = threading.Lock()
_sem: threading.BoundedSemaphore | None = None
_sem_lock = threading.Lock()


def _semaphore() -> threading.BoundedSemaphore:
    global _sem
    with _sem_lock:
        if _sem is None:
            _sem = threading.BoundedSemaphore(get_settings().llm_concurrency)
        return _sem


def _pause_for(seconds: float) -> None:
    global _global_pause_until
    with _pause_lock:
        _global_pause_until = max(_global_pause_until, time.time() + seconds)


def _wait_if_paused(deadline: float | None) -> None:
    while True:
        with _pause_lock:
            wait = _global_pause_until - time.time()
        if wait <= 0:
            return
        if deadline is not None and time.time() + wait > deadline:
            raise DeadlineExceeded("job deadline reached while waiting out provider rate limit")
        time.sleep(min(wait, 5.0))


_client: httpx.Client | None = None
_client_lock = threading.Lock()


def _http() -> httpx.Client:
    global _client
    with _client_lock:
        if _client is None:
            s = get_settings()
            _client = httpx.Client(
                timeout=httpx.Timeout(s.llm_timeout_s, connect=15.0),
                limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
            )
        return _client


def extract_json(text: str) -> Any:
    """Parse the JSON object/array a model returned, tolerating code fences and chatter."""
    if text is None:
        raise ValueError("empty response")
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    try:
        return json.loads(t)
    except Exception:  # noqa: BLE001
        pass
    # take the outermost {...} or [...]
    starts = [i for i in (t.find("{"), t.find("[")) if i != -1]
    if not starts:
        raise ValueError("no JSON found in response")
    i = min(starts)
    closer = "}" if t[i] == "{" else "]"
    j = t.rfind(closer)
    if j <= i:
        raise ValueError("unterminated JSON in response")
    return json.loads(t[i : j + 1])


def _cache_key(model: str, messages: list[dict], params: dict) -> str:
    h = hashlib.sha256()
    h.update(json.dumps([model, messages, params], sort_keys=True, default=str).encode())
    return h.hexdigest()


def _price(model: str, tin: int, tout: int) -> float:
    pin, pout = PRICES_PER_M.get(model, (2.0, 10.0))
    return (tin * pin + tout * pout) / 1_000_000


@dataclass
class LLMResult:
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    cached: bool
    latency_ms: int


def chat(
    model: str,
    messages: list[dict],
    *,
    tracker: CostTracker | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    deadline: float | None = None,
    cache: bool = True,
    tag: str = "",
    max_attempts: int = 8,
) -> LLMResult:
    """One chat completion with retries. Raises LLMError subclasses on hard failure."""
    if model not in ALLOWED_MODELS:
        raise LLMError(f"model {model!r} is not on the allowed list")
    s = get_settings()
    if not s.llm_base_url or not s.llm_api_key:
        raise LLMError("LLM_BASE_URL / LLM_API_KEY are not set")

    params = {"max_tokens": max_tokens, "temperature": temperature}
    key = _cache_key(model, messages, params)
    cpath = s.cache_dir / key[:2] / f"{key}.json"
    if cache and s.use_cache and cpath.exists():
        try:
            c = json.loads(cpath.read_text())
            if tracker:
                tracker.add(model, c["tokens_in"], c["tokens_out"], c["cost_usd"], cached=True)
            return LLMResult(c["text"], model, c["tokens_in"], c["tokens_out"], c["cost_usd"], True, 0)
        except Exception:  # noqa: BLE001  (corrupt cache entry: ignore and refetch)
            pass

    if s.max_spend_usd and LEDGER.total() >= s.max_spend_usd:
        raise BudgetExceeded(f"spend cap ${s.max_spend_usd:.2f} reached (spent ${LEDGER.total():.2f})")

    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "usage": {"include": True}}
    headers = {"Authorization": f"Bearer {s.llm_api_key}", "Content-Type": "application/json"}
    url = f"{s.llm_base_url}/chat/completions"

    last_err = "unknown"
    for attempt in range(1, max_attempts + 1):
        if deadline is not None and time.time() > deadline:
            raise DeadlineExceeded(f"deadline reached before LLM call succeeded ({last_err})")
        _wait_if_paused(deadline)
        t0 = time.time()
        retry_after: float | None = None
        try:
            with _semaphore():
                r = _http().post(url, headers=headers, json=body)
            if r.status_code == 429 or r.status_code in (408, 409, 425) or r.status_code >= 500:
                ra = r.headers.get("retry-after")
                try:
                    retry_after = float(ra) if ra else None
                except ValueError:
                    retry_after = None
                last_err = f"HTTP {r.status_code}"
                if r.status_code == 429:
                    _pause_for(retry_after if retry_after else min(2 ** attempt, 30))
                raise _Retry()
            if r.status_code in (401, 402, 403):
                raise LLMError(f"HTTP {r.status_code} from gateway: {r.text[:200]}")
            if r.status_code >= 400:
                raise LLMError(f"HTTP {r.status_code} from gateway: {r.text[:300]}")
            data = r.json()
            if "error" in data and not data.get("choices"):
                # OpenRouter reports upstream failures as 200 + {"error": ...}
                last_err = f"upstream error: {str(data['error'])[:200]}"
                raise _Retry()
            choice = (data.get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content") or ""
            if not text.strip():
                last_err = f"empty completion (finish_reason={choice.get('finish_reason')})"
                raise _Retry()
            usage = data.get("usage") or {}
            tin = int(usage.get("prompt_tokens") or 0)
            tout = int(usage.get("completion_tokens") or 0)
            cost = usage.get("cost")
            cost = float(cost) if isinstance(cost, (int, float)) else _price(model, tin, tout)
            lat = int((time.time() - t0) * 1000)
            log.info("llm call ok", extra={"model": model, "tag": tag, "tokens_in": tin, "tokens_out": tout,
                                           "cost_usd": round(cost, 6), "latency_ms": lat, "attempt": attempt})
            if tracker:
                tracker.add(model, tin, tout, cost, cached=False)
            LEDGER.add(cost)
            if cache and s.use_cache:
                try:
                    cpath.parent.mkdir(parents=True, exist_ok=True)
                    tmp = cpath.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
                    tmp.write_text(json.dumps({"text": text, "tokens_in": tin, "tokens_out": tout, "cost_usd": cost,
                                               "model": model}))
                    tmp.replace(cpath)
                except Exception:  # noqa: BLE001
                    pass
            return LLMResult(text, model, tin, tout, cost, False, lat)
        except _Retry:
            pass
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError, json.JSONDecodeError) as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt == max_attempts:
            break
        delay = retry_after if retry_after is not None else min(1.5 * (2 ** (attempt - 1)), 40.0)
        delay = delay * (0.75 + random.random() * 0.5)
        if deadline is not None and time.time() + delay > deadline:
            raise DeadlineExceeded(f"deadline reached while retrying LLM call ({last_err})")
        log.warning("llm call retry", extra={"model": model, "tag": tag, "attempt": attempt, "reason": last_err,
                                             "sleep_s": round(delay, 1)})
        time.sleep(delay)
    raise LLMError(f"LLM call failed after {max_attempts} attempts: {last_err}")


class _Retry(Exception):
    pass


def chat_json(
    model: str,
    messages: list[dict],
    *,
    tracker: CostTracker | None = None,
    max_tokens: int = 4096,
    deadline: float | None = None,
    tag: str = "",
    json_retries: int = 2,
) -> tuple[Any, LLMResult]:
    """chat() + robust JSON parsing. If the model returns unusable JSON, re-ask once or twice
    (without the cache, which would just replay the bad answer)."""
    msgs = list(messages)
    last: Exception | None = None
    for i in range(json_retries + 1):
        res = chat(model, msgs, tracker=tracker, max_tokens=max_tokens, deadline=deadline, tag=tag,
                   cache=(i == 0))
        try:
            return extract_json(res.text), res
        except Exception as e:  # noqa: BLE001
            last = e
            log.warning("llm returned unparseable JSON", extra={"model": model, "tag": tag, "error": str(e)[:120]})
            msgs = list(messages) + [
                {"role": "assistant", "content": res.text[:2000]},
                {"role": "user", "content": "That was not valid JSON. Reply again with ONLY the JSON object, "
                                            "no commentary, no code fences."},
            ]
    raise LLMError(f"model returned unparseable JSON after {json_retries + 1} tries: {last}")
