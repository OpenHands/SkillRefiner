"""Batch orchestration for raw-trace summarizers."""

import asyncio

from tqdm.asyncio import tqdm

from skill_refiner.summarize.core import TraceSummarizer, TraceSummary
from skill_refiner.summarize.llm import LLMTraceSummarizer
from skill_refiner.trace.context import EvaluationContext
from skill_refiner.trace.store import Trace


class TraceSummarizationRunner:
    """Runs a trace summarizer over many traces with bounded concurrency."""

    def __init__(
        self,
        concurrency: int = 16,
        summarizer: TraceSummarizer | None = None,
    ) -> None:
        self._concurrency = concurrency
        self._summarizer = summarizer or LLMTraceSummarizer()

    @property
    def technique(self) -> str:
        return self._summarizer.technique

    async def summarize(
        self,
        traces: list[tuple[str, Trace]],
        ctx: EvaluationContext,
    ) -> list[TraceSummary]:
        sem = asyncio.Semaphore(self._concurrency)

        async def _limited(tid: str, traj: Trace) -> TraceSummary | None:
            async with sem:
                result = await self._summarizer.summarize(tid, traj, ctx)
                bar.update(1)
                return result

        tasks = [_limited(tid, traj) for tid, traj in traces]
        with tqdm(total=len(tasks), desc=f"summarizing ({self.technique})", unit="trace") as bar:
            results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
