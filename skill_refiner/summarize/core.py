"""Core types for raw-trace summarization."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from skill_refiner.trace.context import EvaluationContext
from skill_refiner.trace.store import Trace


@dataclass(frozen=True)
class TraceSummary:
    """Per-trace summary of what the agent actually did. No PR fields, no labels."""

    trace_id: str
    summary: str
    observations: tuple[str, ...]
    prompt: str = field(default="", compare=False)
    """The full prompt sent to the LLM for this trace (excluded from equality)."""
    technique: str = field(default="unknown", compare=False)
    """Summarization technique that produced this summary."""
    created_at: str | None = field(default=None, compare=False)
    """UTC timestamp when this summary was produced."""
    summarization_cost: float | None = field(default=None, compare=False)
    """LLM cost for producing this summary when available."""


class TraceSummarizer(ABC):
    """Strategy interface for producing one summary for one trace."""

    @property
    def technique(self) -> str:
        """Stable name for the summarization technique."""
        return self.__class__.__name__

    @abstractmethod
    async def summarize(
        self,
        trace_id: str,
        trace: Trace,
        ctx: EvaluationContext,
    ) -> TraceSummary | None:
        """Return a summary for ``trace`` or None to skip it."""
