"""SkillRefiner ablations. Each ablation removes exactly one stage of the technique.

Held constant across all five: ground-truth injection, merge prompt, held-out eval
harness and judges. The base configuration is a plain ``RefineConfig``, so the base
run and the ablation arms agree by construction.
"""

import argparse
import asyncio
from dataclasses import replace
from pathlib import Path

from skill_refiner.pipeline import RefineConfig, refine

ABLATIONS = {
    "summary": "full_trace summarizer -> NoSummaryTraceSummarizer (raw trace text)",
    "clustering": "UmapHdbscanClusterer -> SingleClusterer (one cluster per partition)",
    "positive": "drop the positive partition before summarization",
    "negative": "drop the negative partition before summarization",
    "gating": "evidence-verify gate off; every negative lesson reaches the skill",
}


def build_base_config(
    *, skill_file: Path, skill_name: str, raw_jsonl: Path, binary_rewards: Path,
    output_dir: Path, ground_truth_file: Path | None = None, limit: int | None = None,
) -> RefineConfig:
    """The ablation runner's base config: plain `RefineConfig`, no divergence."""
    return RefineConfig(
        skill_file=skill_file, skill_name=skill_name, raw_jsonl=raw_jsonl,
        binary_rewards=binary_rewards, output_dir=output_dir,
        ground_truth_file=ground_truth_file, limit=limit,
    )


def build_config(base: RefineConfig, ablate: str) -> RefineConfig:
    """Return `base` with exactly one stage swapped out."""
    if ablate not in ABLATIONS:
        raise ValueError(f"unknown ablation {ablate!r}; choose from {sorted(ABLATIONS)}")
    if ablate == "summary":
        return replace(base, summarizer="none")
    if ablate == "clustering":
        return replace(base, cluster_method="single")
    if ablate == "positive":
        return replace(base, use_positive=False)
    if ablate == "negative":
        return replace(base, use_negative=False)
    return replace(base, verify_negative_clusters=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ablate", required=True, choices=sorted(ABLATIONS))
    p.add_argument("--skill-file", type=Path, required=True)
    p.add_argument("--skill-name", required=True)
    p.add_argument("--raw-jsonl", type=Path, required=True)
    p.add_argument("--binary-rewards", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--ground-truth-file", type=Path)
    p.add_argument("--limit", type=int)
    args = p.parse_args()

    base = build_base_config(
        skill_file=args.skill_file, skill_name=args.skill_name,
        raw_jsonl=args.raw_jsonl, binary_rewards=args.binary_rewards,
        output_dir=args.output_dir, ground_truth_file=args.ground_truth_file,
        limit=args.limit,
    )
    asyncio.run(refine(build_config(base, args.ablate)))


if __name__ == "__main__":
    main()
