"""Build the grader-diff ground-truth map the configuration of record injects.

``RefineConfig.ground_truth_file`` takes a ``{trace_id: text}`` JSON object.
``refine()`` appends each entry to its trace as a SCORING REFERENCE event
*before* summarization, so the (skill-blind) summarizer of a failing run sees
the grader's own diagnosis of what the run got wrong rather than having to
infer it from the transcript. This is the ``--ground-truth-file
.../ground_truth_from_grader_diff.json`` in the README's reproduction recipe.

This script produces that file for SpreadsheetBench from the official
evaluator's ``eval_official_results.json`` (written by
``spreadsheetbench_agent_runner.py`` at the end of a rollout): for every
FAILING instance it takes the first failing test case's message -- the
evaluator's ``expected 'X', got 'Y'`` cell diff. Passing instances are omitted,
so only the negative partition receives an injected reference.

The map is keyed to a particular rollout's trace ids, so generate it from the
rollout you are refining from.

Usage:
  uv run python scripts/benchmarks/spreadsheetbench/build_grader_diff_ground_truth.py \\
      --results results/spreadsheetbench/train/eval_official_results.json \\
      --out results/spreadsheetbench/train/ground_truth_from_grader_diff.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_ground_truth_from_grader_diff(results_path: Path, out_path: Path) -> dict[str, str]:
    """{instance_id: official evaluator's failing-cell message} for FAILING instances.

    ``hard_score`` is the official evaluator's all-cells-match flag; an instance
    with ``hard_score == 1`` passed and is skipped.
    """
    data = json.loads(results_path.read_text(encoding="utf-8"))
    results = data.get("results", [])
    ground_truth: dict[str, str] = {}
    n_fail = 0
    for result in results:
        if int(result.get("hard_score") or 0) == 1:
            continue
        n_fail += 1
        message = next(
            (
                case.get("message", "")
                for case in result.get("test_cases", [])
                if not case.get("passed")
            ),
            "",
        )
        ground_truth[str(result["id"])] = message
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")
    print(
        f"Grader-diff ground truth: {n_fail} failing / {len(results)} total "
        f"instances -> {out_path}"
    )
    return ground_truth


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results", type=Path, required=True, metavar="FILE",
        help="The official evaluator's eval_official_results.json from a rollout",
    )
    parser.add_argument(
        "--out", type=Path, required=True, metavar="FILE",
        help="Where to write the {trace_id: grader message} JSON map",
    )
    args = parser.parse_args(argv)
    if not args.results.is_file():
        print(f"ERROR: no such results file: {args.results}", file=sys.stderr)
        return 2
    ground_truth = build_ground_truth_from_grader_diff(args.results, args.out)
    if not ground_truth:
        print(
            "WARNING: every instance passed, so the map is empty. refine() rejects an "
            "empty ground_truth_file -- omit --ground-truth-file instead of passing this.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
