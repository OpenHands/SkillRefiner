"""LLM-based raw-trace summarization strategies and shared helpers."""

import json
import re
from datetime import UTC, datetime
from typing import Any, Literal

from openhands.sdk.llm.message import Message, TextContent

from skill_refiner.propose import budget as _budget
from skill_refiner.summarize.core import TraceSummarizer, TraceSummary
from skill_refiner.trace.context import EvaluationContext
from skill_refiner.trace.serialization import serialize_trace
from skill_refiner.trace.store import Trace

PolarityLiteral = Literal["positive", "negative"]

def _check_polarity(polarity: str | None) -> None:
    if polarity is not None and polarity not in ("positive", "negative"):
        raise ValueError(f"polarity must be 'positive', 'negative', or None, got {polarity!r}")


def _skill_run_header(
    polarity: PolarityLiteral | None,
    skill_name: str | None,
) -> str:
    """Provenance header stating which skill the agent invoked and, when the
    polarity is known, whether the run passed or failed downstream evaluation.

    ``skill_name`` is the skill the agent used for this run — e.g. "PR review"
    for the PR-review experiment, or the relevant SkillsBench skill name. When
    it is None the header falls back to a generic "a skill" phrasing.
    """
    used = f"invoked the `{skill_name}` skill" if skill_name else "invoked a skill"
    if polarity == "positive":
        outcome = " and then PASSED its downstream evaluation"
    elif polarity == "negative":
        outcome = " and then FAILED its downstream evaluation"
    else:
        outcome = ""
    return f"This trace comes from an AI agent run that {used}{outcome}."


class LLMTraceSummarizer(TraceSummarizer):
    """Generates skill-blind raw-trace summaries with one LLM call per trace."""

    def __init__(
        self,
        polarity: PolarityLiteral | None = None,
        skill_name: str | None = None,
    ) -> None:
        _check_polarity(polarity)
        self._polarity = polarity
        self._skill_name = skill_name

    @property
    def technique(self) -> str:
        return "llm-full-trace"

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
        fixed_tokens = _budget.count_tokens(
            _build_prompt("", self._polarity, self._skill_name), model
        )
        available = _budget.context_window(model) - fixed_tokens - _budget.OUTPUT_RESERVE
        content = _budget.truncate_to_tokens(content, available, model)
        if not content.strip():
            print(
                f"  WARNING: trace {trace_id} content truncated to empty "
                "(context budget exhausted) — skipped"
            )
            return None

        prompt = _build_prompt(content, self._polarity, self._skill_name)
        try:
            response = await ctx.llm.acompletion(
                messages=[Message(role="user", content=[TextContent(text=prompt)])]
            )
        except Exception as exc:
            print(f"  WARNING: summarizer LLM call failed for trace {trace_id}: {exc} — skipped")
            return None

        return _summary_from_parsed(
            trace_id,
            _parse_summary_response(_response_text(response)),
            prompt,
            technique=self.technique,
            summarization_cost=response_cost(response),
        )


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def response_cost(response: Any) -> float | None:
    for path in (
        ("cost",),
        ("response_cost",),
        ("_hidden_params", "response_cost"),
        ("raw_response", "_hidden_params", "response_cost"),
    ):
        value = _nested_value(response, path)
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def _response_text(response: Any) -> str:
    text_parts = []
    for content_item in response.message.content:
        text_item = getattr(content_item, "text", None)
        if isinstance(text_item, str):
            text_parts.append(text_item)
    return "".join(text_parts)


def _nested_value(obj: Any, path: tuple[str, ...]) -> Any:
    value = obj
    for key in path:
        value = value.get(key) if isinstance(value, dict) else getattr(value, key, None)
        if value is None:
            return None
    return value


def _parse_summary_response(text: str) -> dict[str, object]:
    try:
        parsed = json.loads(_extract_json(text))
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    return {"summary": text.strip()[:2000], "observations": []}


def _summary_from_parsed(
    trace_id: str,
    parsed: dict[str, object],
    prompt: str,
    *,
    technique: str,
    summarization_cost: float | None,
) -> TraceSummary:
    observations = parsed.get("observations", [])
    if not isinstance(observations, list):
        observations = []
    summary = parsed.get("summary", "")
    return TraceSummary(
        trace_id=trace_id,
        summary=summary if isinstance(summary, str) else "",
        observations=tuple(item for item in observations if isinstance(item, str)),
        prompt=prompt,
        technique=technique,
        created_at=utc_now_iso(),
        summarization_cost=summarization_cost,
    )


def _extract_json(text: str) -> str:
    """Strip <think> blocks and markdown fences, returning raw JSON text."""
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        return "\n".join(inner)
    return stripped


def _build_prompt(
    trace_content: str,
    polarity: PolarityLiteral | None = None,
    skill_name: str | None = None,
) -> str:
    header = _skill_run_header(polarity, skill_name)
    intro = (
        "Below is a cleaned execution trace of an AI agent run. Each section is one "
        "agent turn or tool result. Internal model reasoning has been stripped — you "
        "are seeing only tool calls, tool results, and agent outputs.\n\n"
        f"--- TRACE START ---\n{trace_content}\n--- TRACE END ---\n\n"
    )
    if polarity == "positive":
        return intro + (
            f"{header} Reconstruct "
            "the behavioral recipe that produced success.\n\n"
            "Analyze this trace and answer:\n"
            "1. What did the agent do? List the key tool calls (name + what it was "
            "trying to accomplish) and what each returned.\n"
            "2. Which decisions, orderings, or checks were instrumental to the "
            "successful outcome?\n"
            "3. What is the behavioral recipe? State the reusable rules that made "
            "this run succeed.\n\n"
            "Reply as JSON (no markdown wrapper):\n"
            '{"summary": "<3-5 sentence account of what the agent did and why it succeeded>", '
            '"observations": ["<codifiable behavior worth reinforcing>", "..."]}\n\n'
            "observations must be codifiable behaviors — rules you could write into "
            'a skill so the pattern repeats, e.g. "agent verified the diff against '
            'the files manifest before commenting" not "agent was careful".'
        )
    if polarity == "negative":
        return intro + (
            f"{header} Diagnose the "
            "failure chain.\n\n"
            "Analyze this trace and answer:\n"
            "1. What did the agent do? List the key tool calls (name + what it was "
            "trying to accomplish) and what each returned.\n"
            "2. Diagnose the failure chain: what was the earliest decision that led "
            "the run astray, and what would a correct trace have done there?\n"
            "3. What instruction or guardrail would have prevented this failure?\n\n"
            "Reply as JSON (no markdown wrapper):\n"
            '{"summary": "<3-5 sentence account of what the agent did and where it went wrong>", '
            '"observations": ["<failure mode phrased as a preventable rule>", "..."]}\n\n'
            "observations must be failure modes phrased as preventable rules, e.g. "
            '"agent skipped reading the diff before commenting — require a diff read '
            'first" not "agent made mistakes".'
        )
    body = (
        "Analyze this trace and answer:\n"
        "1. What did the agent do? List the key tool calls (name + what it was trying "
        "to accomplish) and what each returned.\n"
        "2. What was the agent's final output or conclusion?\n"
        "3. What stands out behaviorally? Look for: repeated commands, unnecessary "
        "steps, missing steps, commands that returned nothing useful, or patterns "
        "that suggest the agent was confused or working around a gap in its instructions.\n\n"
        "Reply as JSON (no markdown wrapper):\n"
        '{"summary": "<3-5 sentence account of what the agent did and concluded>", '
        '"observations": ["<specific behavioral pattern worth noting>", "..."]}\n\n'
        "observations should be concrete and reusable — e.g. "
        '"agent ran `date` before every CVE check" not "agent checked something".'
    )
    if skill_name is None:
        # No polarity (handled above) and no skill_name: nothing to attribute this
        # trace to, so stay fully skill-blind rather than emitting a generic
        # "invoked a skill" header with no actual skill behind it.
        return intro + body
    return intro + f"{header}\n\n" + body
