"""End-to-end coverage for the DEFAULT configuration (the configuration of record).

tests/test_pipeline_end_to_end.py selects ``cluster_method="single"``, so this module
covers the ``umap_hdbscan`` default path end to end. It guards against wiring defects
where a declared default stays correct but the value never arrives, for example:

  * ``method="single"`` being forced into ``_run_partition``;
  * ``min_cluster_size`` not reaching the clusterer;
  * ``UmapHdbscanClusterer.cluster`` dropping ``**self._kwargs``;
  * ``_make_summarizer`` receiving ``polarity=None`` (polarity steering silently off).

So this module asserts on what ARRIVES. The clusterer, the summarizer factory and the merge prompt
are SPIED — wrapped and called through, never replaced — and the assertions are on the
arguments they actually received. Only the two real network boundaries are stubbed, the
same two as the sibling module: the refinement LLM and the embedding client. UMAP and
HDBSCAN run for real.

It also pins the merge stage's skill loading (see ``test_merge_stage_*`` below).
"""

import json
import random
import re

import pytest

from skill_refiner import pipeline as pipeline_mod
from skill_refiner.cluster.umap_hdbscan import EmbeddingClient
from skill_refiner.merge import synthesize as synthesize_mod
from skill_refiner.pipeline import RefineConfig, refine
from skill_refiner.propose import clustered as clustered_mod
from tests.test_pipeline_end_to_end import FakeLLM, _FakeResponse

# ---------------------------------------------------------------------------
# Corpus. Two well-separated behavioral groups per polarity, so real UMAP +
# HDBSCAN has something to find. Everything is content-derived and seeded, so
# the clustering is reproducible run to run.
# ---------------------------------------------------------------------------

N_PER_POLARITY = 14

SKILL_FRONTMATTER = (
    "---\n"
    "name: xlsx\n"
    "description: Use this skill when creating, editing, analyzing or verifying "
    "spreadsheet files.\n"
    "---\n"
)
SKILL_BODY = "\n# xlsx\n\nPlaceholder skill body for the default-configuration test.\n"
SKILL_MD = SKILL_FRONTMATTER + SKILL_BODY

_TRACE_ID_RE = re.compile(r"trace-(?:pos|neg)-(\d+)")


class GroupedFakeLLM(FakeLLM):
    """FakeLLM, but summaries differ by trace so the embeddings are not degenerate.

    The base FakeLLM answers every summarize call with the same string, which would
    embed 28 identical points and make UMAP's output meaningless. Here each summary
    carries its trace's group label, so the two groups separate.
    """

    async def acompletion(self, *, messages, **_kwargs):
        prompt = messages[0].content[0].text
        is_summarize = (
            "auditing a proposed skill edit" not in prompt
            and "form a semantic cluster" not in prompt
            and "cluster(s) of agent behavior were identified" not in prompt
        )
        if not is_summarize:
            return await super().acompletion(messages=messages, **_kwargs)
        self.prompts_seen.append(prompt)
        match = _TRACE_ID_RE.search(prompt)
        index = int(match.group(1)) if match else 0
        group = "A" if index % 2 == 0 else "B"
        return _FakeResponse(
            json.dumps(
                {
                    "summary": (
                        f"group {group} behavior: the agent followed the {group} "
                        f"path on trace {index}"
                    ),
                    "observations": [f"group {group} observation {index}"],
                }
            )
        )


_GROUP_RE = re.compile(r"group ([AB]) behavior.*?trace (\d+)", re.S)


def _fake_embed_one(self, text: str) -> list[float]:
    """Two well-separated but non-degenerate clusters, derived from the summary text.

    Seeded per summary, so the vectors are byte-stable across runs and across
    machines -- the clustering this drives is reproducible, which a clustering
    test has to be. Identical vectors would collapse each group to one point and
    HDBSCAN would return pure noise, so each point gets its own small offset.
    """
    match = _GROUP_RE.search(text)
    group, index = (match.group(1), match.group(2)) if match else ("A", "0")
    rng = random.Random(f"{group}-{index}")
    axis = [1.0, 0.0] if group == "A" else [0.0, 1.0]
    return [a + rng.uniform(-0.05, 0.05) for a in axis] + [
        rng.uniform(-0.05, 0.05) for _ in range(6)
    ]


def _trace_line(trace_id: str) -> str:
    return json.dumps(
        {
            "trace_id": trace_id,
            "spans": [
                {
                    "span_id": "s1",
                    "name": "tool_call",
                    "input_text": "",
                    "output_text": f"USER: do the task\nAGENT: work performed for {trace_id}",
                    "start_time": "",
                    "end_time": "",
                }
            ],
        }
    )


@pytest.fixture
def corpus(tmp_path):
    positive = [f"trace-pos-{i}" for i in range(N_PER_POLARITY)]
    negative = [f"trace-neg-{i}" for i in range(N_PER_POLARITY)]

    raw_jsonl = tmp_path / "traces.jsonl"
    raw_jsonl.write_text(
        "\n".join(_trace_line(t) for t in positive + negative) + "\n", encoding="utf-8"
    )

    rewards = tmp_path / "binary_rewards.json"
    rewards.write_text(
        json.dumps({t: True for t in positive} | {t: False for t in negative}),
        encoding="utf-8",
    )

    skill_file = tmp_path / "base_true.md"
    skill_file.write_text(SKILL_MD, encoding="utf-8")

    return {
        "raw_jsonl": raw_jsonl,
        "binary_rewards": rewards,
        "skill_file": skill_file,
        "output_dir": tmp_path / "out",
    }


@pytest.fixture
def spies(monkeypatch):
    """Wrap-and-call-through recorders on the three wiring points under test."""
    record = {"cluster": [], "summarizer": [], "merge": []}

    real_cluster = clustered_mod.cluster_by_umap_hdbscan

    def spy_cluster(vectors, **kwargs):
        record["cluster"].append({"n_items": len(vectors), "kwargs": dict(kwargs)})
        return real_cluster(vectors, **kwargs)

    monkeypatch.setattr(clustered_mod, "cluster_by_umap_hdbscan", spy_cluster)

    real_make_summarizer = pipeline_mod._make_summarizer

    def spy_make_summarizer(name, polarity, skill_name):
        summarizer = real_make_summarizer(name, polarity, skill_name)
        record["summarizer"].append(
            {
                "name": name,
                "polarity_arg": polarity,
                "polarity_on_instance": getattr(summarizer, "_polarity", "<absent>"),
                "skill_name": skill_name,
                "type": type(summarizer).__name__,
            }
        )
        return summarizer

    monkeypatch.setattr(pipeline_mod, "_make_summarizer", spy_make_summarizer)

    real_merge_prompt = synthesize_mod._merge_prompt

    def spy_merge_prompt(proposals, skill, **kwargs):
        record["merge"].append(
            {"n_proposals": len(proposals), "skill_content": skill.content}
        )
        return real_merge_prompt(proposals, skill, **kwargs)

    monkeypatch.setattr(synthesize_mod, "_merge_prompt", spy_merge_prompt)
    return record


@pytest.fixture
def fake_llm(monkeypatch):
    llm = GroupedFakeLLM()
    monkeypatch.setattr(pipeline_mod, "_build_llm", lambda config: llm)
    return llm


@pytest.fixture
def fake_embedding(monkeypatch):
    monkeypatch.setattr(EmbeddingClient, "embed_one", _fake_embed_one)


@pytest.fixture
async def default_run(tmp_path, monkeypatch, corpus, spies, fake_llm, fake_embedding):
    """One refine() at the shipped defaults. Nothing about the config is overridden."""
    monkeypatch.chdir(tmp_path)  # embedding + summary caches use relative paths
    config = RefineConfig(
        skill_file=corpus["skill_file"],
        skill_name="xlsx",
        raw_jsonl=corpus["raw_jsonl"],
        binary_rewards=corpus["binary_rewards"],
        output_dir=corpus["output_dir"],
    )
    proposal = await refine(config)
    return {"proposal": proposal, "config": config, "spies": spies, "llm": fake_llm}


# ---------------------------------------------------------------------------
# The defaults are what actually runs.
# ---------------------------------------------------------------------------


async def test_the_default_run_reaches_a_merged_proposal(default_run):
    assert default_run["proposal"] is not None
    combined = default_run["config"].output_dir / "combined" / "proposed_skill.md"
    assert combined.is_file()


async def test_umap_hdbscan_actually_runs_on_both_partitions(default_run):
    """Guards against: forcing method="single" into _run_partition.

    With 'single' selected, cluster_by_umap_hdbscan is never called at all --
    SingleClusterer puts every item in cluster 0 without touching it.
    """
    calls = default_run["spies"]["cluster"]
    assert len(calls) == 2, f"expected one clustering pass per polarity, got {len(calls)}"
    assert [c["n_items"] for c in calls] == [N_PER_POLARITY, N_PER_POLARITY]


async def test_the_configured_min_cluster_size_arrives_at_the_clusterer(default_run):
    """Guards against: hardcoding hdbscan_min_cluster_size=99, and dropping **self._kwargs.

    RefineConfig.min_cluster_size is 2 (not HDBSCAN's default of 3); it travels
    through _run_partition -> UmapHdbscanClusterer(hdbscan_min_cluster_size=...) ->
    self._kwargs -> cluster_by_umap_hdbscan. Only the far end proves the wiring:
    UmapHdbscanClusterer.cluster ignoring **self._kwargs leaves this empty.
    """
    for call in default_run["spies"]["cluster"]:
        assert call["kwargs"] == {"hdbscan_min_cluster_size": 2}, call["kwargs"]


async def test_both_polarities_get_a_polarity_steered_summarizer(default_run):
    """Guards against: _make_summarizer(polarity=None).

    Polarity steering is the artifact's central mechanism -- a success is asked to
    reconstruct the recipe, a failure to diagnose the chain. Passing None silently
    swaps both for the framework's generic behavior prompt.
    """
    calls = default_run["spies"]["summarizer"]
    assert len(calls) == 2, f"expected one summarizer per partition, got {len(calls)}"
    assert sorted(c["polarity_arg"] for c in calls) == ["negative", "positive"]
    # ...and it has to survive into the object, not just the call.
    assert sorted(c["polarity_on_instance"] for c in calls) == ["negative", "positive"]
    assert {c["name"] for c in calls} == {"full_trace"}
    assert {c["type"] for c in calls} == {"LLMTraceSummarizer"}
    assert {c["skill_name"] for c in calls} == {"xlsx"}


async def test_the_polarity_specific_prompts_are_the_ones_that_get_sent(default_run):
    """The other half of the above: both steered prompt bodies reach the LLM."""
    prompts = default_run["llm"].prompts_seen
    assert any("PASSED their evaluation" in p for p in prompts)
    assert any("FAILED their evaluation" in p for p in prompts)


async def test_the_merge_is_the_only_synthesis_call(default_run):
    """Paper §2: proposals go straight from the clusters into one merge.

    Exactly one merge prompt is built, it
    carries both partitions' proposals, and no partition writes its own skill.
    """
    calls = default_run["spies"]["merge"]
    assert len(calls) == 1, f"expected exactly one merge, got {len(calls)}"
    merge_prompts = [
        p for p in default_run["llm"].prompts_seen
        if "cluster(s) of agent behavior were identified" in p
    ]
    assert len(merge_prompts) == 1
    assert "polarity=positive" in merge_prompts[0]
    assert "polarity=negative" in merge_prompts[0]
    out = default_run["config"].output_dir
    assert not (out / "positive" / "proposed_skill.md").exists()
    assert not (out / "negative" / "proposed_skill.md").exists()


async def test_the_evidence_verify_gate_runs_on_the_negative_partition(default_run):
    """verify_negative_clusters defaults True; the gate is negative-only."""
    prompts = default_run["llm"].prompts_seen
    verify_prompts = [p for p in prompts if "auditing a proposed skill edit" in p]
    assert verify_prompts, "the evidence-verify gate never ran"


async def test_the_merge_prompt_is_the_papers_a54_prompt(default_run):
    """Paper §A.5.4: skill, clusters, merge framing, JSON contract -- nothing else."""
    merge_prompt = [
        p
        for p in default_run["llm"].prompts_seen
        if "cluster(s) of agent behavior were identified" in p
    ][-1]
    assert "CRITICAL constraints" not in merge_prompt
    assert "Synthesise these cluster proposals" not in merge_prompt
    assert f"{synthesize_mod._MERGE_FRAMING_SAS}\n\nReply as JSON" in merge_prompt


# ---------------------------------------------------------------------------
# C1: the two stages load the base skill two different ways, on purpose.
# ---------------------------------------------------------------------------


async def test_merge_stage_sees_the_raw_skill_including_its_frontmatter(default_run):
    """The merge input must be the RAW file, frontmatter included.

    Skill.load parses the YAML frontmatter off into fields and rewrites the header.
    The merge prompt embeds skill.content verbatim and the reply is written straight
    out as the proposed skill, so loading it here would strip the `description:`
    line the agent harness triggers the skill on.
    """
    merge_call = default_run["spies"]["merge"][-1]
    assert merge_call["skill_content"] == SKILL_MD
    assert merge_call["skill_content"].startswith("---\nname: xlsx\n")
    assert "description: Use this skill when" in merge_call["skill_content"]


async def test_the_merge_prompt_sent_to_the_llm_carries_the_frontmatter(default_run):
    merge_prompts = [
        p
        for p in default_run["llm"].prompts_seen
        if "cluster(s) of agent behavior were identified" in p
    ]
    assert merge_prompts
    assert "description: Use this skill when" in merge_prompts[-1]


async def test_refine_stage_keeps_the_parsed_skill_load_form(default_run):
    """The per-cluster proposal prompts quote the Skill.load form (frontmatter parsed
    off), so the two stages legitimately differ. Pinning both halves stops either
    one being changed by accident."""
    per_cluster_prompts = [
        p for p in default_run["llm"].prompts_seen if "form a semantic cluster" in p
    ]
    assert per_cluster_prompts
    assert all("description: Use this skill when" not in p for p in per_cluster_prompts)


def test_a_merge_skill_built_from_a_file_without_frontmatter_is_unchanged(tmp_path):
    """The raw read is a straight passthrough, not a frontmatter injector."""
    from openhands.sdk.skills import Skill

    plain = tmp_path / "plain.md"
    plain.write_text("# xlsx\n\nno frontmatter here\n", encoding="utf-8")
    skill = Skill(name=plain.stem, content=plain.read_text(encoding="utf-8"))
    assert skill.content == "# xlsx\n\nno frontmatter here\n"


def test_output_dir_is_where_the_proposal_lands(corpus):
    """Sanity: the fixture's output_dir is not silently shared with the sibling test."""
    assert not corpus["output_dir"].exists()
