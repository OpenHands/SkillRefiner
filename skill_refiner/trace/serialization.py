"""Raw trace serialization helpers for skill-blind summarization."""

import json

from openhands.sdk.event import Event, LLMConvertibleEvent

from skill_refiner.trace.event_normalization import normalize_event
from skill_refiner.trace.store import Trace

# Derived artifact event kinds — excluded structurally so no suggestion
# breakdowns, categories, reflection labels, or PR metadata reach the LLM.
# NOTE: this is a denylist — any NEW derived/artifact event kind added to a
# trace store must be listed here or it will leak into the prompt.
_DERIVED_KINDS = frozenset({"pr_review_context", "precomputed_reflection"})

# Top-level keys in raw span JSON output that are internal model reasoning /
# metadata — they dominate token budgets but add no behavioral signal.
_NOISY_SPAN_KEYS = frozenset(
    {
        "reasoning_content",
        "thinking_blocks",
        "provider_specific_fields",
        "refusal",
    }
)
_RAW_SPAN_KINDS = frozenset({"raw_span"})
_SECTION_SEPARATOR = "\n---\n"


def _clean_span_output(text: str) -> str:
    """Strip internal model reasoning fields from a raw span output string.

    The output of an agent turn is often a JSON array (one object per turn).
    Each object may contain ``reasoning_content``, ``thinking_blocks``, and
    ``provider_specific_fields`` which are internal extended-thinking data.
    These fields consume the vast majority of tokens but carry no behavioral
    signal for skill learning — only the ``content`` and ``tool_calls`` matter.
    """
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text

    def _clean(obj: object) -> object:
        if isinstance(obj, dict):
            result = {k: _clean(v) for k, v in obj.items() if k not in _NOISY_SPAN_KEYS}
            # Drop signature blobs inside content items (thinking block artifacts)
            if "signature" in obj and isinstance(obj.get("type"), str):
                result.pop("signature", None)
            return result
        if isinstance(obj, list):
            return [_clean(item) for item in obj]
        return obj

    cleaned = _clean(parsed)
    return json.dumps(cleaned, ensure_ascii=False)


def serialize_trace(trace: Trace) -> str:
    """Render a trace's raw content as text.

    Raw span outputs are cleaned to strip internal reasoning fields before
    inclusion; derived artifact kinds are dropped entirely; any other event is
    JSON-dumped.
    """
    return _SECTION_SEPARATOR.join(_serialize_trace_sections(trace))


def deserialize_trace(trace: Trace) -> list[LLMConvertibleEvent]:
    """Deserialize raw trace event dicts into typed SDK ``Event`` objects.

    Only ``LLMConvertibleEvent`` instances (system prompts, actions,
    observations, messages) are returned — other event kinds (e.g.
    ``TokenEvent``, ``StreamingDeltaEvent``) are silently dropped.
    """
    events: list[LLMConvertibleEvent] = []
    for raw in trace:
        normalized = normalize_event(raw)
        try:
            ev = Event.model_validate(normalized)
        except Exception:
            continue
        if isinstance(ev, LLMConvertibleEvent):
            events.append(ev)
    return events


def _serialize_trace_sections(trace: Trace) -> list[str]:
    parts: list[str] = []
    for event in trace:
        kind = event.get("kind", "")
        if kind in _DERIVED_KINDS:
            continue
        if kind in _RAW_SPAN_KINDS:
            output = event.get("output", "")
            if output:
                parts.append(_clean_span_output(output))
        else:
            parts.append(json.dumps(event, sort_keys=True))
    return [part for part in parts if part.strip()]
