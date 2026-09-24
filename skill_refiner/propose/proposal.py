"""RefinementProposal — a proposed change to a skill."""

from pydantic import BaseModel


class RefinementProposal(BaseModel, frozen=True):
    """A proposed change to a skill, with rationale."""

    skill_name: str
    """Which skill this proposal targets."""

    original_content: str
    """The skill content before refinement."""

    proposed_content: str
    """The proposed new content."""

    rationale: str
    """Why this change should help — human-readable explanation."""

    confidence: float
    """How confident the refiner is (0.0 to 1.0)."""

    failure_patterns: list[str] = []
    """Patterns observed in failed evaluations that drove this proposal."""

    success_patterns: list[str] = []
    """Patterns observed in successful evaluations to preserve."""

    observations: list[str] = []
    """Recurring behaviors observed in raw traces that drove this proposal."""

    n_traces: int = 0
    """Number of trace summaries included in the refinement prompt."""

    refiner_prompt: str = ""
    """The full prompt sent to the LLM for this refinement call."""
