"""EvaluationContext — runtime dependencies passed through the evaluation interface."""

from dataclasses import dataclass

from openhands.sdk.llm import LLM


@dataclass(frozen=True)
class EvaluationContext:
    """Runtime dependencies provided by the runner."""

    llm: LLM
    secrets: dict[str, str]
