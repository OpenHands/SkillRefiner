"""Polarity-specialized pipeline stages for partitioned (positive/negative) runs.

PolarityClusterRefiner  — reinforce-only prompt for positive clusters,
                          soften/guardrail-only prompt for negative clusters.

Polarity is partition-based: the caller declares it at construction; nothing
here infers polarity from scores.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from openhands.sdk.skills import Skill  # noqa: TC002

from skill_refiner.propose import budget as _budget
from skill_refiner.propose.clustered import _extract_json, _str_list, _summary_block
from skill_refiner.propose.core import RefinementContext  # noqa: TC001
from skill_refiner.propose.protocols import ClusterProposal, Polarity

if TYPE_CHECKING:
    from skill_refiner.summarize.core import TraceSummary

_VALID_POLARITIES = ("positive", "negative")


def _check_polarity(polarity: str) -> None:
    if polarity not in _VALID_POLARITIES:
        raise ValueError(
            f"polarity must be one of {_VALID_POLARITIES}, got {polarity!r}"
        )


def _positive_cluster_prompt(skill_content: str, cluster_id: int,
                             n: int, summaries_block: str) -> str:
    return (
        f"Current skill content:\n```markdown\n{skill_content}\n```\n\n"
        f"The following {n} agent trace summaries form a semantic cluster "
        f"(cluster {cluster_id}). All traces in this cluster come from runs that "
        "PASSED their evaluation — they represent validated, successful behavior.\n\n"
        f"{summaries_block}\n\n"
        "Based on this cluster of successful traces:\n"
        "1. Identify the common theme across these traces.\n"
        "2. Identify the specific behaviors that drove success and should be "
        "REINFORCED in the skill — patterns to codify so they repeat reliably.\n"
        "3. Propose one concrete, targeted edit to the skill that codifies the "
        "winning pattern.\n\n"
        "Do NOT propose removals or warnings — these traces succeeded; your job "
        "is to lock in what worked.\n\n"
        "Reply as JSON (no markdown wrapper):\n"
        '{"theme": "<what these traces have in common>", '
        '"reinforce": ["<behavior to strengthen>", "..."], '
        '"suggested_edit": "<one concrete minimal edit that codifies the winning pattern>"}'
    )


def _negative_cluster_prompt(skill_content: str, cluster_id: int,
                             n: int, summaries_block: str) -> str:
    return (
        f"Current skill content:\n```markdown\n{skill_content}\n```\n\n"
        f"The following {n} agent trace summaries form a semantic cluster "
        f"(cluster {cluster_id}). All traces in this cluster come from runs that "
        "FAILED their evaluation — they represent a recurring failure mode.\n\n"
        f"{summaries_block}\n\n"
        "Based on this cluster of failing traces:\n"
        "1. Identify the common theme across these traces.\n"
        "2. Identify the specific behaviors that caused failure and should be "
        "SOFTENED or removed — instructions that misled the agent, missing "
        "guardrails, or friction that wasted effort.\n"
        "3. Propose one concrete, targeted edit to the skill — a guardrail to add "
        "or a misleading instruction to remove — that would prevent this failure "
        "mode from repeating.\n\n"
        "Do NOT invent praise — these traces failed; your job is to stop the "
        "failure from repeating.\n\n"
        "Reply as JSON (no markdown wrapper):\n"
        '{"theme": "<what these traces have in common>", '
        '"soften": ["<behavior to remove or de-emphasise>", "..."], '
        '"suggested_edit": "<one concrete minimal edit that prevents this failure mode>"}'
    )


def _verify_prompt(cluster_id: int, proposal: ClusterProposal,
                   summaries_block: str) -> str:
    """Judge prompt: is a negative cluster's proposed lesson supported by evidence?

    The evidence-verifier pass-gate (Trace2Skill-style, adapted to skill-lab's
    cluster-proposal architecture): rather than validating a per-trace fix against
    ground truth, an LLM judge checks whether the proposed failure-mode lesson is
    actually borne out by the cluster's own trace evidence. Unsupported proposals
    are dropped before they reach the skill, trading recall for precision.
    """
    soften = "\n".join(f"  - {s}" for s in proposal.soften) or "  (none)"
    return (
        f"You are auditing a proposed skill edit for cluster {cluster_id}. The edit "
        "was inferred from a cluster of FAILING agent traces and is meant to prevent "
        "a recurring failure mode. Your job is to decide whether the proposed lesson "
        "is actually SUPPORTED by the trace evidence below — not merely plausible.\n\n"
        f"Proposed theme:\n  {proposal.theme}\n\n"
        f"Proposed behaviors to soften/remove:\n{soften}\n\n"
        f"Proposed edit:\n  {proposal.suggested_edit}\n\n"
        f"Trace evidence (the failing traces this was inferred from):\n\n"
        f"{summaries_block}\n\n"
        "A proposal is SUPPORTED only if the traces show concrete evidence that the "
        "named behavior actually occurred AND plausibly caused the failures. Reject "
        "proposals that generalize beyond the evidence, blame a cause the traces do "
        "not show, or restate a generic best-practice not grounded in these traces.\n\n"
        "Reply as JSON (no markdown wrapper):\n"
        '{"supported": true/false, "reason": "<one sentence citing the evidence>"}'
    )


class PolarityClusterRefiner:
    """ClusterRefiner for single-polarity partitions.

    Positive clusters get a reinforce-only prompt (soften=[] always);
    negative clusters get a soften/guardrail-only prompt (reinforce=[] always).
    """

    def __init__(self, polarity: Polarity, *, verify: bool = False) -> None:
        _check_polarity(polarity)
        self._polarity: Polarity = polarity
        # Evidence-verifier pass-gate; only meaningful on the negative path
        # (positive clusters come from PASSING traces, already ground-truth valid).
        self._verify: bool = verify and polarity == "negative"
        # Proposals the evidence gate rejected. They are dropped from the skill by
        # design, but discarding the RECORD too makes the gate unauditable: the run
        # log keeps only a 120-char reason and no theme line, so a rejected proposal
        # cannot be quoted or re-examined afterwards. Kept in memory for the caller
        # to persist alongside cluster_proposals.json.
        self.rejected: list[dict] = []

    async def propose(
        self,
        cluster_id: int,
        items: list[Any],
        skill: Skill,
        ctx: RefinementContext,
        model: str,
    ) -> ClusterProposal | None:
        from openhands.sdk.llm.message import Message, TextContent

        summaries: list[TraceSummary] = items
        builder = (
            _positive_cluster_prompt if self._polarity == "positive"
            else _negative_cluster_prompt
        )

        blocks = [_summary_block(s) for s in summaries]
        fixed_tokens = _budget.count_tokens(
            builder(skill.content, cluster_id, 0, ""), model
        )
        available = _budget.context_window(model) - fixed_tokens - _budget.OUTPUT_RESERVE
        n = _budget.pack_sequential(blocks, available, model)
        summaries_block = "\n".join(blocks[:n])
        prompt = builder(skill.content, cluster_id, n, summaries_block)

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

        proposal = ClusterProposal(
            cluster_id=cluster_id,
            theme=str(parsed.get("theme", "")),
            reinforce=(
                _str_list(parsed.get("reinforce", []))
                if self._polarity == "positive" else []
            ),
            soften=(
                _str_list(parsed.get("soften", []))
                if self._polarity == "negative" else []
            ),
            suggested_edit=str(parsed.get("suggested_edit", "")),
            n_traces=n,
            cluster_size=len(summaries),
            polarity=self._polarity,
        )

        if self._verify and not await self._is_supported(
            cluster_id, proposal, summaries_block, ctx
        ):
            return None  # pass-gate: drop lessons the evidence does not support

        return proposal

    async def _is_supported(
        self,
        cluster_id: int,
        proposal: ClusterProposal,
        summaries_block: str,
        ctx: RefinementContext,
    ) -> bool:
        """LLM evidence-verifier gate. Fail-open: only an explicit False drops.

        A verifier error (LLM exception, unparseable reply, missing field) keeps the
        proposal — an infrastructure failure is not evidence of unsupportedness, and
        must not silently discard signal.
        """
        from openhands.sdk.llm.message import Message, TextContent

        prompt = _verify_prompt(cluster_id, proposal, summaries_block)
        try:
            response = await ctx.llm.acompletion(
                messages=[Message(role="user", content=[TextContent(text=prompt)])]
            )
            text = "".join(c.text for c in response.message.content if hasattr(c, "text"))
            parsed = json.loads(_extract_json(text))
        except Exception:
            print(f"  [verify] cluster {cluster_id}: verifier error — keeping (fail-open)")
            return True

        if not isinstance(parsed, dict) or "supported" not in parsed:
            print(f"  [verify] cluster {cluster_id}: no verdict — keeping (fail-open)")
            return True

        supported = bool(parsed.get("supported"))
        full_reason = str(parsed.get("reason", ""))
        verdict = "kept" if supported else "DROPPED"
        print(f"  [verify] cluster {cluster_id}: {verdict} — {full_reason[:120]}")
        if not supported:
            self.rejected.append({
                "cluster_id": cluster_id,
                "theme": proposal.theme,
                "soften": list(proposal.soften),
                "suggested_edit": proposal.suggested_edit,
                "n_traces": proposal.n_traces,
                "cluster_size": proposal.cluster_size,
                "verdict": "dropped",
                "reason": full_reason,
            })
        return supported
