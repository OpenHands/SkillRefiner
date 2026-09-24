"""Run SkillRefiner end to end on the bundled minimal corpus.

One command, no dataset download:

  uv run python examples/minimal/run.py

It refines ``seed_skill.md`` against the 12 traces in ``traces.jsonl`` /
``binary_rewards.json`` using the defaults on ``RefineConfig`` -- the
configuration of record -- and prints the path of the refined skill.

Prerequisites (run ``make doctor`` to check all of them at once):

  * an API key in ``$LLM_API_KEY`` or ``$OPENAI_API_KEY``
  * an LLM base URL in ``$SKILL_REFINER_BASE_URL_EVAL_PROXY`` or
    ``artifact.local.json``'s ``llm_base_urls.eval_proxy``
  * an embedding endpoint serving ``qwen3-embedding:4b`` at
    ``http://localhost:11434/v1`` (override with ``$EMBEDDING_BASE_URL`` /
    ``$EMBEDDING_MODEL``)

Output goes to ``examples/minimal/output/`` (gitignored). The run costs roughly
20 LLM calls: one summary per trace, one proposal per cluster, the evidence gate on
the negative clusters, and the merge.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from skill_refiner.pipeline import RefineConfig, refine

HERE = Path(__file__).resolve().parent


def build_config(output_dir: Path) -> RefineConfig:
    """The configuration of record, pointed at the bundled corpus.

    Every stage field is left at its ``RefineConfig`` default on purpose: this
    example is meant to exercise the shipped configuration, not a variant of it.
    ``ground_truth_file`` is the one field of the record the example cannot set --
    see the README's note on grader feedback.
    """
    return RefineConfig(
        skill_file=HERE / "seed_skill.md",
        skill_name="xlsx",
        raw_jsonl=HERE / "traces.jsonl",
        binary_rewards=HERE / "binary_rewards.json",
        output_dir=output_dir,
        embedding_base_url=os.environ.get(
            "EMBEDDING_BASE_URL", RefineConfig.embedding_base_url
        ),
        embedding_model=os.environ.get("EMBEDDING_MODEL", RefineConfig.embedding_model),
    )


def main() -> int:
    output_dir = Path(os.environ.get("EXAMPLE_OUTPUT_DIR", HERE / "output"))
    proposal = asyncio.run(refine(build_config(output_dir)))
    if proposal is None:
        print(
            "\nNo proposal: no cluster survived on either polarity. "
            "That is a real outcome, not a crash -- rerun, or widen the corpus.",
            file=sys.stderr,
        )
        return 1
    refined = output_dir / "combined" / "proposed_skill.md"
    print("\n" + "=" * 72)
    print(f"Refined skill : {refined}")
    print(f"Manifest      : {output_dir / 'combined' / 'manifest.json'}")
    print(f"Confidence    : {proposal.confidence}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
