"""End-to-end coverage for skill_refiner.pipeline.refine.

tests/test_pipeline_smoke.py only checks that refine is a coroutine function
and that RefineConfig holds its fields. This module drives refine() through real
code: trace loading, per-polarity summarization, clustering, per-cluster proposal
(plus the negative-only evidence gate), and the cross-polarity merge. Only the two
genuine network boundaries are stubbed — the refinement LLM (``_build_llm``) and the
embedding client
(``EmbeddingClient.embed_one``). No other production code is patched,
monkeypatched, or bypassed.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skill_refiner import pipeline as pipeline_mod
from skill_refiner.cluster.umap_hdbscan import EmbeddingClient
from skill_refiner.pipeline import RefineConfig, refine
from skill_refiner.propose import RefinementProposal

# ---------------------------------------------------------------------------
# Stubs for the two real network boundaries.
# ---------------------------------------------------------------------------


class _FakeTextContent:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeMessage:
    def __init__(self, content: list[_FakeTextContent]) -> None:
        self.content = content


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.message = _FakeMessage([_FakeTextContent(text)])


class FakeLLM:
    """Stands in for openhands.sdk.llm.LLM. Never touches the network.

    Dispatches on markers unique to each of the four prompt shapes the
    pipeline builds (trace summary, per-cluster proposal, evidence-verify,
    synthesis/merge) so every call site gets JSON it can parse, without
    importing any prompt-builder internals — the marker strings are read
    straight off the literals in llm.py / polarity.py / clustered.py.
    """

    def __init__(self) -> None:
        self.model = "fake/test-model"
        self.base_url = "http://fake-llm.invalid"
        self.metrics = SimpleNamespace(
            accumulated_cost=0.0, accumulated_token_usage=None, costs=[]
        )
        self.prompts_seen: list[str] = []

    async def acompletion(self, *, messages, **_kwargs):
        prompt = messages[0].content[0].text
        self.prompts_seen.append(prompt)
        if "auditing a proposed skill edit" in prompt:
            payload = {"supported": True, "reason": "fake: evidence supports it"}
        elif "form a semantic cluster" in prompt:
            payload = {
                "theme": "fake cluster theme",
                "reinforce": ["fake reinforce behavior"],
                "soften": ["fake soften behavior"],
                "suggested_edit": "fake suggested edit",
            }
        elif "cluster(s) of agent behavior were identified" in prompt:
            payload = {
                "proposed_content": "# fake skill\n\nfake revised body\n",
                "rationale": "fake rationale",
                "key_observations": ["fake observation"],
                "confidence": 0.75,
            }
        else:
            payload = {"summary": "fake summary", "observations": ["fake observation"]}
        return _FakeResponse(json.dumps(payload))


def _fake_embed_one(self, text: str) -> list[float]:
    # Deterministic, content-derived: identical text embeds identically. The
    # values themselves don't matter for cluster_method="single" (it ignores
    # vector content), but a stub for the real network call should still
    # behave like a real embedding call would.
    h = abs(hash(text)) % 1000
    return [float(h), 1.0, 0.0, 0.0]


SKILL_MD = "# xlsx\n\nPlaceholder skill body for end-to-end pipeline testing.\n"


def _write_skill(tmp_path: Path) -> Path:
    path = tmp_path / "xlsx.md"
    path.write_text(SKILL_MD, encoding="utf-8")
    return path


def _trace_line(trace_id: str, text: str) -> str:
    return json.dumps(
        {
            "trace_id": trace_id,
            "spans": [
                {
                    "span_id": "s1",
                    "name": "tool_call",
                    "input_text": "",
                    "output_text": text,
                    "start_time": "",
                    "end_time": "",
                }
            ],
        }
    )


def _write_traces(tmp_path: Path, trace_ids: list[str]) -> Path:
    path = tmp_path / "traces.jsonl"
    lines = [
        _trace_line(tid, f"USER: do the task\nAGENT: did something for {tid}")
        for tid in trace_ids
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_binary_rewards(tmp_path: Path, positive: list[str], negative: list[str]) -> Path:
    path = tmp_path / "binary_rewards.json"
    payload = {tid: True for tid in positive} | {tid: False for tid in negative}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def fake_llm(monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr(pipeline_mod, "_build_llm", lambda config: llm)
    return llm


@pytest.fixture
def fake_embedding(monkeypatch):
    monkeypatch.setattr(EmbeddingClient, "embed_one", _fake_embed_one)


async def test_refine_reaches_merge_with_stubbed_llm_and_embeddings(
    tmp_path, monkeypatch, fake_llm, fake_embedding
):
    """Drives refine() through both partitions, clustering, proposal, and merge.

    cluster_method="single" (not the umap_hdbscan default) is deliberate: with
    stubbed embedding vectors and only 3 traces per partition, UMAP+HDBSCAN's
    behavior on tiny N is dominated by noise (cluster/umap_hdbscan.py documents
    this itself: row-order/eigengap instability at N<=6) and would make this an
    unreliable instrument. 'single' is a real, shipped ClusterMethod — it
    exercises the identical propose -> verify -> merge code path
    deterministically every run.
    """
    monkeypatch.chdir(tmp_path)  # embedding + summary caches use relative paths

    positive_ids = ["pos-1", "pos-2", "pos-3"]
    negative_ids = ["neg-1", "neg-2", "neg-3"]
    raw_jsonl = _write_traces(tmp_path, positive_ids + negative_ids)
    binary_rewards = _write_binary_rewards(tmp_path, positive_ids, negative_ids)
    skill_file = _write_skill(tmp_path)
    output_dir = tmp_path / "out"

    config = RefineConfig(
        skill_file=skill_file,
        skill_name="xlsx",
        raw_jsonl=raw_jsonl,
        binary_rewards=binary_rewards,
        output_dir=output_dir,
        cluster_method="single",
    )

    result = await refine(config)

    assert isinstance(result, RefinementProposal)
    assert result.proposed_content == "# fake skill\n\nfake revised body\n"
    assert result.skill_name == "xlsx"

    # The merge only runs once both partitions produced a surviving cluster —
    # confirm both partitions actually ran, not just that *a* proposal came back.
    assert (output_dir / "positive" / "cluster_proposals.json").is_file()
    assert (output_dir / "negative" / "cluster_proposals.json").is_file()
    # The merge is the only call that writes a skill.
    assert not (output_dir / "positive" / "proposed_skill.md").exists()
    assert not (output_dir / "negative" / "proposed_skill.md").exists()
    assert (output_dir / "combined" / "proposed_skill.md").is_file()
    assert (output_dir / "combined" / "manifest.json").is_file()

    combined_manifest = json.loads((output_dir / "combined" / "manifest.json").read_text())
    assert combined_manifest["n_positive_clusters"] == 1
    assert combined_manifest["n_negative_clusters"] == 1

    # 6 summarize calls (3 traces x 2 polarities) + 2 propose calls (1 cluster x
    # 2 polarities) + 1 verify call (negative-only; verify_negative_clusters
    # defaults True) + 1 cross-polarity merge = 10.
    assert len(fake_llm.prompts_seen) == 10


async def test_refine_raises_when_binary_rewards_is_missing(tmp_path, monkeypatch, fake_llm):
    """binary_rewards pointing at a nonexistent file is a real failure mode.

    _load_binary_scores returns None for a missing path (and {} for a
    valid-but-empty file, which is equally falsy),
    so split_trace_groups falls back to the polarity-blind "all" partition.
    refine() has no merge path for that partition shape, so it must fail loudly
    instead of silently running a polarity-blind refinement.
    """
    raw_jsonl = _write_traces(tmp_path, ["only-1"])
    skill_file = _write_skill(tmp_path)
    missing_rewards = tmp_path / "does_not_exist.json"

    config = RefineConfig(
        skill_file=skill_file,
        skill_name="xlsx",
        raw_jsonl=raw_jsonl,
        binary_rewards=missing_rewards,
        output_dir=tmp_path / "out",
    )

    with pytest.raises(RuntimeError, match="positive/negative split"):
        await refine(config)


async def test_refine_raises_when_binary_rewards_is_empty_json(tmp_path, monkeypatch, fake_llm):
    """An existing-but-empty {} rewards file hits the same guard via a different
    branch: _score_from_binary_rewards({}) parses fine but returns {}, which
    _load_binary_scores treats as falsy at the call site in refine() the same
    way None is — so this must raise identically to the missing-file case.
    """
    raw_jsonl = _write_traces(tmp_path, ["only-1"])
    skill_file = _write_skill(tmp_path)
    empty_rewards = tmp_path / "empty_rewards.json"
    empty_rewards.write_text("{}", encoding="utf-8")

    config = RefineConfig(
        skill_file=skill_file,
        skill_name="xlsx",
        raw_jsonl=raw_jsonl,
        binary_rewards=empty_rewards,
        output_dir=tmp_path / "out",
    )

    with pytest.raises(RuntimeError, match="positive/negative split"):
        await refine(config)
