"""Per-job context threaded through the pipeline."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from ..config import Settings, get_settings
from ..llm import CostTracker

log = logging.getLogger("pipeline")


@dataclass
class JobContext:
    job_id: str = "local"
    settings: Settings = field(default_factory=get_settings)
    tracker: CostTracker = field(default_factory=CostTracker)
    deadline: float | None = None
    stage_cb: Callable[[str], None] | None = None
    warnings: list[str] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)      # reasons the result is PARTIAL rather than SUCCESS
    started: float = field(default_factory=time.time)

    def stage(self, name: str) -> None:
        log.info("stage", extra={"job_id": self.job_id, "stage": name})
        if self.stage_cb:
            try:
                self.stage_cb(name)
            except Exception:  # noqa: BLE001
                pass

    def warn(self, msg: str) -> None:
        log.warning(msg, extra={"job_id": self.job_id})
        self.warnings.append(msg)

    def degrade(self, reason: str) -> None:
        self.degraded.append(reason)
        self.warn("DEGRADED: " + reason)

    def time_left(self) -> float:
        return float("inf") if self.deadline is None else self.deadline - time.time()
