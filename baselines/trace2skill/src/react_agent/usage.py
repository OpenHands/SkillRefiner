"""Process-wide token/cost accounting for Trace2Skill LLM calls.

Both the error-analysis (ReAct, per-instance ``OpenAIClient``) and success-analysis
(raw ``openai.OpenAI``) runners parallelize over instances with a ThreadPoolExecutor,
so a single lock-guarded module-level accumulator captures the whole run's usage.

Token counts come from ``response.usage`` (always present). Cost comes from the
eval-proxy ``x-litellm-response-cost`` response header when available (read via
``with_raw_response``); it stays 0.0 for endpoints that don't return it (e.g. Modal),
in which case price the ``total_tokens`` yourself.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class UsageTracker:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    api_calls: int = 0
    cache_hits: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, *, prompt: int = 0, completion: int = 0,
               total: int = 0, cost: float = 0.0) -> None:
        p, c = int(prompt or 0), int(completion or 0)
        with self._lock:
            self.prompt_tokens += p
            self.completion_tokens += c
            self.total_tokens += int(total or (p + c))
            self.cost_usd += float(cost or 0.0)
            self.api_calls += 1

    def record_cache_hit(self) -> None:
        with self._lock:
            self.cache_hits += 1

    def summary(self) -> dict:
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "cost_usd": round(self.cost_usd, 4),
                "api_calls": self.api_calls,
                "cache_hits": self.cache_hits,
            }

    def reset(self) -> None:
        with self._lock:
            self.prompt_tokens = self.completion_tokens = self.total_tokens = 0
            self.cost_usd = 0.0
            self.api_calls = self.cache_hits = 0


#: Shared accumulator for a single process's run.
GLOBAL = UsageTracker()


def extract_cost(headers) -> float:
    """Best-effort read of the eval-proxy per-response cost header (USD)."""
    if not headers:
        return 0.0
    try:
        return float(headers.get("x-litellm-response-cost", 0) or 0)
    except Exception:
        return 0.0


def record_response(response, headers=None) -> None:
    """Record one API response's usage + optional cost header into GLOBAL."""
    usage = getattr(response, "usage", None)
    GLOBAL.record(
        prompt=getattr(usage, "prompt_tokens", 0) if usage else 0,
        completion=getattr(usage, "completion_tokens", 0) if usage else 0,
        total=getattr(usage, "total_tokens", 0) if usage else 0,
        cost=extract_cost(headers),
    )
