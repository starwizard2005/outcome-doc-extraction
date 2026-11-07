"""Runtime configuration. Everything is read from the environment at call time.

Nothing here hardcodes a key or an endpoint: all model traffic goes through
LLM_BASE_URL / LLM_API_KEY (see contract/README.md).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# The only models this service may ever call (assignment section 3). Enforced in
# llm.py: asking for anything else raises before a request is made.
ALLOWED_MODELS = (
    "google/gemini-3.8-flash",
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-haiku-5.5",
    "anthropic/claude-sonnet-5.5",
    "openai/gpt-6-luna",
    "openai/gpt-6.1-sol",
    "z-ai/glm-5.3-flash",
    "qwen/qwen3-vl-235b-a22b-instruct",
    "deepseek/deepseek-v3.2",
)

# USD per 1M tokens (input, output), taken from the OpenRouter /models listing.
# Used only when a response carries no `usage.cost`; the gateway's own figure wins.
PRICES_PER_M = {
    "google/gemini-3.8-flash": (0.825, 4.125),
    "google/gemini-3.1-pro-preview": (2.0, 12.0),
    "anthropic/claude-haiku-5.5": (0.10, 0.50),
    "anthropic/claude-sonnet-5.5": (2.0, 10.0),
    "openai/gpt-6-luna": (0.10, 0.50),
    "openai/gpt-6.1-sol": (2.0, 10.0),
    "z-ai/glm-5.3-flash": (0.15, 0.50),
    "qwen/qwen3-vl-235b-a22b-instruct": (0.21, 1.90),
    "deepseek/deepseek-v3.2": (0.26, 0.38),
}


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_api_key: str
    data_dir: Path
    cache_dir: Path
    workers: int
    max_pending: int
    max_upload_bytes: int
    job_deadline_s: float       # hard budget per job, counted from submission
    lease_s: float              # a RUNNING job not heartbeating for this long is re-queued
    max_attempts: int
    llm_concurrency: int        # simultaneous in-flight LLM requests across all jobs
    llm_timeout_s: float
    max_spend_usd: float        # 0 = unlimited. Dev safety net for the $10 budget.
    use_cache: bool
    # model routing (all overridable, all validated against ALLOWED_MODELS)
    model_spec: str             # writes table-extraction specs
    model_record: str           # first-pass record extraction
    model_strong: str           # escalation for low-confidence fields
    model_vision: str           # page-image verification
    model_cheap: str            # small helper calls (header fields, row batches)
    version: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_dir = Path(os.environ.get("DATA_DIR", "./data")).resolve()
    return Settings(
        llm_base_url=os.environ.get("LLM_BASE_URL", "").rstrip("/"),
        llm_api_key=os.environ.get("LLM_API_KEY", ""),
        data_dir=data_dir,
        cache_dir=Path(os.environ.get("CACHE_DIR", str(data_dir / "llm_cache"))),
        workers=_i("WORKERS", 6),
        max_pending=_i("MAX_PENDING", 500),
        max_upload_bytes=_i("MAX_UPLOAD_MB", 50) * 1024 * 1024,
        job_deadline_s=_f("JOB_DEADLINE_S", 1560),   # 26 min: leaves margin under the 30 min limit
        lease_s=_f("LEASE_S", 120),
        max_attempts=_i("MAX_ATTEMPTS", 3),
        llm_concurrency=_i("LLM_CONCURRENCY", 12),
        llm_timeout_s=_f("LLM_TIMEOUT_S", 240),
        max_spend_usd=_f("MAX_SPEND_USD", 0),
        use_cache=os.environ.get("LLM_CACHE", "1") not in ("0", "false", "no"),
        model_spec=os.environ.get("MODEL_SPEC", "anthropic/claude-haiku-5.5"),
        model_record=os.environ.get("MODEL_RECORD", "anthropic/claude-haiku-5.5"),
        model_strong=os.environ.get("MODEL_STRONG", "anthropic/claude-sonnet-5.5"),
        model_vision=os.environ.get("MODEL_VISION", "z-ai/glm-5.3-flash"),
        model_cheap=os.environ.get("MODEL_CHEAP", "anthropic/claude-haiku-5.5"),
        version=os.environ.get("APP_VERSION", "dev"),
    )


def reset_settings_cache() -> None:
    get_settings.cache_clear()
