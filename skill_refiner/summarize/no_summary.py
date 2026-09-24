"""Pass-through 'summarizer' for the w/o-Summary ablation.

Every downstream stage of the SkillRefiner pipeline reads a trace only through
``TraceSummary.summary`` / ``.observations``: the embedding step
(``SummaryTextExtractor``), the per-cluster proposal prompt (``_summary_block``),
and the negative-cluster evidence gate. Removing the summary component therefore
means handing those stages the trace's own serialized raw text instead of an
LLM-written account of it.

This class is that substitution and nothing else. It makes no LLM call, invents
no structure, and applies no bespoke compression: the only transformation is the
same ``_budget.truncate_to_tokens`` call the real summarizer already applied to
its own input, against the same budget. ``observations`` is empty because there
is no summarizer to derive observations from -- the existing prompt builders
already render an empty list as ``(none)``.
"""

from skill_refiner.propose import budget as _budget
from skill_refiner.summarize.core import TraceSummarizer, TraceSummary
from skill_refiner.summarize.llm import PolarityLiteral, utc_now_iso
from skill_refiner.trace.context import EvaluationContext
from skill_refiner.trace.serialization import serialize_trace
from skill_refiner.trace.store import Trace


class NoSummaryTraceSummarizer(TraceSummarizer):
    """Emits the raw trace text in place of a summary. Zero LLM calls.

    ``polarity`` and ``skill_name`` are accepted so this stage is constructed
    exactly like every other summarizer, but they are unused: both only ever fed
    the summarizer's prompt header, and there is no prompt any more. Polarity is
    not lost from the pipeline -- the partitions still split on it, and the
    positive/negative cluster-proposal prompts state PASSED/FAILED themselves.
    """

    def __init__(
        self,
        polarity: PolarityLiteral | None = None,
        skill_name: str | None = None,
    ) -> None:
        self._polarity = polarity
        self._skill_name = skill_name

    @property
    def technique(self) -> str:
        # Distinct from every real summarizer's technique so the driver's summary
        # cache can never serve a summarized trace to this arm, or vice versa.
        return "no-summary"

    async def summarize(
        self,
        trace_id: str,
        trace: Trace,
        ctx: EvaluationContext,
    ) -> TraceSummary | None:
        content = serialize_trace(trace)
        if not content.strip():
            print(f"  WARNING: trace {trace_id} has no raw content after filtering — skipped")
            return None

        model = getattr(ctx.llm, "model", "") or ""
        # The budget the real summarizer had for this same trace: everything the
        # model could have read. Downstream prompts do their own packing on top.
        available = _budget.context_window(model) - _budget.OUTPUT_RESERVE
        content = _budget.truncate_to_tokens(content, available, model)
        if not content.strip():
            print(
                f"  WARNING: trace {trace_id} content truncated to empty "
                "(context budget exhausted) — skipped"
            )
            return None

        return TraceSummary(
            trace_id=trace_id,
            summary=content,
            observations=(),
            prompt="",  # no prompt was issued
            technique=self.technique,
            created_at=utc_now_iso(),
            summarization_cost=0.0,
        )
