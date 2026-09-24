"""The shipped defaults ARE the configuration of record (paper §2, §3, Appendix A.2).

If any assertion here fails, the artifact no longer ships the configuration the
paper describes.
"""

import inspect
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


def test_summarizer_is_full_trace():
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


def test_both_polarities_are_on():
    cfg = _config()
    assert cfg.use_positive is True
    assert cfg.use_negative is True


def test_soften_at_source_is_not_electable():
    assert "soften_at_source" not in inspect.signature(merge_polarity_proposals).parameters
