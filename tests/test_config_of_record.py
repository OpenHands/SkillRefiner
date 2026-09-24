"""The shipped defaults ARE the configuration of record, with one recorded exception.

Source: skillrefiner_main.md (2026-09-22) section 2 — the 81.0% configuration.
If any assertion here fails, the artifact no longer ships what the paper measured.

Deliberate divergence: the 81.0% run also passed a 1280-char hand-written,
SpreadsheetBench-specific merge guardrail via --extra-constraints-file. That
mechanism has been removed from the technique (see the test below), so the
shipped merge prompt is the configuration of record MINUS that block. §6 of
skillrefiner_main.md reports the block was ignored by the model and credits
--soften-at-source with the gain, but that was measured with the block present;
the 81.0% number has not been re-measured without it.
"""

import inspect
from dataclasses import fields
from pathlib import Path

from skill_refiner.cluster import cluster_by_umap_hdbscan
from skill_refiner.merge import merge_polarity_proposals
from skill_refiner.pipeline import RefineConfig

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


def _config():
    return RefineConfig(
        skill_file=Path("s.md"), skill_name="xlsx", raw_jsonl=Path("t.jsonl"),
        binary_rewards=Path("b.json"), output_dir=Path("out"),
    )


def test_summarizer_is_full_trace_not_hierarchical():
    """hierarchical scored 78.0% and has weaker polarity steering."""
    assert _config().summarizer == "full_trace"


def test_clustering_is_umap_hdbscan_at_min_cluster_size_two():
    cfg = _config()
    assert cfg.cluster_method == "umap_hdbscan"
    assert cfg.min_cluster_size == 2
    assert inspect.signature(
        cluster_by_umap_hdbscan
    ).parameters["hdbscan_min_cluster_size"].default == 2


def test_evidence_verify_gate_is_on():
    assert _config().verify_negative_clusters is True


def test_no_hand_written_merge_guardrail_mechanism_exists():
    """The merge-stage guardrail block is removed from the technique entirely.

    skillrefiner_main.md §2 records that the 81.0% run passed a 1280-char
    hand-written block via --extra-constraints-file; §6 reports the model
    *ignored* it and credits --soften-at-source with the gain. The block was
    SpreadsheetBench-specific (formula recalculation / `#NAME?`) yet sat in the
    merge prompt for every benchmark, so it is gone: no config field, no
    parameter, no shipped file.
    """
    import skill_refiner.pipeline as pipeline
    from skill_refiner.merge.synthesize import _merge_prompt

    assert not hasattr(pipeline, "DEFAULT_EXTRA_CONSTRAINTS_FILE")
    assert "extra_constraints_file" not in {f.name for f in fields(RefineConfig)}
    assert "extra_constraints" not in inspect.signature(_merge_prompt).parameters
    assert "extra_constraints" not in inspect.signature(
        merge_polarity_proposals).parameters
    assert not (PROMPTS / "extra_constraints.md").exists()


def test_both_polarities_are_on():
    cfg = _config()
    assert cfg.use_positive is True
    assert cfg.use_negative is True


def test_soften_at_source_is_not_electable():
    assert "soften_at_source" not in inspect.signature(merge_polarity_proposals).parameters


def test_only_one_summarizer_ships():
    import skill_refiner.summarize as s
    assert not hasattr(s, "HierarchicalTraceSummarizer")
