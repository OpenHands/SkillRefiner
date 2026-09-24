"""Cross-polarity merge: one revised skill from both partitions' proposals (paper §2.6).

The prompt is exactly the paper's merge prompt (§A.5.4): the current skill, the
clusters, ``_MERGE_FRAMING_SAS``, and the JSON contract. The framing carries the
soften-at-source conflict rule: a negative-cluster lesson that limits a positively
endorsed behavior qualifies that behavior where it is stated, rather than being
appended alongside it.
"""

from __future__ import annotations

import json

from openhands.sdk.skills import Skill  # noqa: TC002

from skill_refiner.propose.clustered import _extract_json, _str_list
from skill_refiner.propose.core import RefinementContext  # noqa: TC001
from skill_refiner.propose.proposal import RefinementProposal
from skill_refiner.propose.protocols import ClusterProposal  # noqa: TC001

_MERGE_FRAMING_SAS = (
    "Clusters marked polarity=positive come from PASSING traces — validated "
    "behaviors to preserve and strengthen. Clusters marked polarity=negative come "
    "from FAILING traces — failure modes that ACTUALLY occurred, and must be weighed "
    "as seriously as the positive signal (a failure is at least as strong evidence as "
    "a success pattern of equal cluster_size). Validated behaviors are the default, "
    "NOT sacrosanct. Conflict rule: when a negative cluster shows that a behavior a "
    "positive cluster endorses is being over-applied, over-trusted, or misused, do NOT "
    "keep the positive text loud and unchanged — qualify, scope, or condition that "
    "behavior AT ITS SOURCE so the caveat sits together with the behavior it limits, "
    "editing the existing text down rather than appending a distant caveat. Delete a "
    "validated behavior outright only when the failure evidence directly contradicts "
    "it; otherwise narrow it to the cases where it still holds."
)


def _cluster_block(p: ClusterProposal) -> str:
    pol = f", polarity={p.polarity}" if p.polarity else ""
    return (
        f"[Cluster {p.cluster_id} — {p.theme} "
        f"(cluster_size={p.cluster_size}, shown_to_llm={p.n_traces}{pol})]\n"
        f"Reinforce: {p.reinforce}\n"
        f"Soften: {p.soften}\n"
        f"Suggested edit: {p.suggested_edit}"
    )


def _merge_prompt(proposals: list[ClusterProposal], skill: Skill, *, total_traces: int) -> str:
    """The merge prompt, paper §A.5.4 (mirrored in prompts/merge.md)."""
    clusters_block = "\n\n".join(_cluster_block(p) for p in proposals)
    return (
        f"Current skill content:\n```markdown\n{skill.content}\n```\n\n"
        f"The following {len(proposals)} cluster(s) of agent behavior were identified "
        f"and analyzed across {total_traces} traces:\n\n"
        f"{clusters_block}\n\n"
        f"{_MERGE_FRAMING_SAS}\n\n"
        "Reply as JSON (no markdown wrapper):\n"
        '{"proposed_content": "<full revised skill, placeholders intact>", '
        '"rationale": "<1-3 sentence explanation of what changed and why>", '
        '"key_observations": ["<recurring behavior that drove a change>", "..."], '
        '"confidence": <self-assessed confidence, 0.0 to 1.0>}'
    )


async def merge_polarity_proposals(
    positive: list[ClusterProposal],
    negative: list[ClusterProposal],
    skill: Skill,
    ctx: RefinementContext,
    *,
    total_traces: int,
) -> RefinementProposal | None:
    """Merge the positive and evidence-gated negative proposals into the initial skill.

    One LLM call. Returns None when the reply is not a usable revised skill.
    """
    from openhands.sdk.llm.message import Message, TextContent

    prompt = _merge_prompt(positive + negative, skill, total_traces=total_traces)
    try:
        response = await ctx.llm.acompletion(
            messages=[Message(role="user", content=[TextContent(text=prompt)])]
        )
        text = "".join(c.text for c in response.message.content if hasattr(c, "text"))
        parsed = json.loads(_extract_json(text))
    except Exception:
        return None

    if not isinstance(parsed, dict):
        return None

    proposed_content = parsed.get("proposed_content", "")
    if not isinstance(proposed_content, str) or not proposed_content.strip():
        return None

    try:
        confidence = min(1.0, max(0.0, float(parsed.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5

    return RefinementProposal(
        skill_name=skill.name,
        original_content=skill.content,
        proposed_content=proposed_content,
        rationale=parsed.get("rationale", "")
        if isinstance(parsed.get("rationale"), str)
        else "",
        confidence=confidence,
        observations=_str_list(parsed.get("key_observations", [])),
        n_traces=total_traces,
    )
