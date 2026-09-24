"""RefinementContext: the runtime dependencies passed to every refinement stage."""

from dataclasses import dataclass

from openhands.sdk.llm import LLM


@dataclass(frozen=True)
class RefinementContext:
    """Runtime dependencies provided to the refiner."""

    llm: LLM
