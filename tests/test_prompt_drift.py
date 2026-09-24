"""prompts/*.md must match what the code actually builds.

The prompts folder is documentation. This test is what stops it becoming fiction.
"""

from pathlib import Path

import pytest
from openhands.sdk.skills import Skill

from skill_refiner.merge.synthesize import _MERGE_FRAMING_SAS, _merge_prompt
from skill_refiner.propose.polarity import (
    _negative_cluster_prompt,
    _positive_cluster_prompt,
    _verify_prompt,
)
from skill_refiner.propose.protocols import ClusterProposal
from skill_refiner.summarize.llm import _build_prompt, _skill_run_header

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"

_SUMMARIES_BLOCK = "- trace t1: did X\n  observations: obs1; obs2"

_CLUSTER_PROPOSAL = ClusterProposal(
    cluster_id=0,
    theme="agent skipped validating the diff before commenting",
    reinforce=[],
    soften=["commented without re-reading the diff"],
    suggested_edit="Require a diff re-read immediately before writing any comment.",
    n_traces=5,
    cluster_size=7,
    polarity="negative",
)

CASES = {
    "skill_run_header.md": lambda: _skill_run_header("negative", "xlsx"),
    "summarize_positive.md": lambda: _build_prompt("<trace content>", "positive", "xlsx"),
    "summarize_negative.md": lambda: _build_prompt("<trace content>", "negative", "xlsx"),
    "evidence_verify_gate.md": lambda: _verify_prompt(0, _CLUSTER_PROPOSAL, _SUMMARIES_BLOCK),
    "positive_cluster.md": lambda: _positive_cluster_prompt("SKILL", 0, 3, "SUMMARIES"),
    "negative_cluster.md": lambda: _negative_cluster_prompt("SKILL", 0, 3, "SUMMARIES"),
    "merge_framing.md": lambda: _MERGE_FRAMING_SAS,
    "merge.md": lambda: _merge_prompt(
        [_CLUSTER_PROPOSAL], Skill(name="xlsx", content="SKILL"), total_traces=7,
    ),
}


@pytest.mark.parametrize("name,builder", CASES.items())
def test_prompt_file_matches_runtime_string(name, builder):
    assert (PROMPTS / name).read_text() == builder(), (
        f"prompts/{name} has drifted from the code that builds it"
    )


# Every prompts/ file is a mirror of a string the code builds; nothing here is
# read at runtime.
def test_every_prompt_file_is_covered_or_listed():
    on_disk = {p.name for p in PROMPTS.glob("*.md")} - {"README.md"}
    assert on_disk >= set(CASES), "a documented prompt has no drift test"
    assert on_disk <= set(CASES), f"a prompts/ file has no drift test: {on_disk - set(CASES)}"


def test_every_shipped_framing_constant_is_mirrored():
    """A framing constant with no prompts/ file is a silent documentation gap."""
    import skill_refiner.merge.synthesize as s

    shipped = {n for n in vars(s) if "FRAMING" in n and not n.startswith("__")}
    mirrored_sources = {"_MERGE_FRAMING_SAS"}
    assert shipped == mirrored_sources, (
        f"framing constants without a prompts/ mirror: {shipped - mirrored_sources}; "
        f"mirrored but no longer shipped: {mirrored_sources - shipped}"
    )
