"""Assemble the parametric rollout's training signal for both evolution engines.

Three artifacts, all keyed by the rollout's trace_id:
  suggestion_learning_traces.jsonl -- the corpus both engines read (--raw-jsonl)
  binary_rewards.json              -- per-trace polarity (--binary-rewards)
  reflection_feedback.json         -- per-trace ground truth (--ground-truth-file)

The polarity rule is deliberately LOOSE (partially_reflected counts as reflected,
byte-identical to build_reflection_rewards.py's set and to
skill_lab.evaluation.trace_quality_scorer.REFLECTED_LABELS) while the miss-list from
build_miss_list.py is STRICT. They answer different questions: "did this review mostly
land?" versus "is this definitely a real issue it should have caught?".

Note: skill_lab.monitoring.traces.artifact_store._REFLECTED_LABELS is a THIRD,
four-label copy of this set that additionally includes "strictly_reflected". This
module's three-label set is a strict subset of it, not a mismatch to reconcile: the
upstream judge (reflect_incorporated_suggestions.py) only ever emits
reflected_directly / reflected_by_alternative_fix / partially_reflected / not_reflected
/ unclear -- it never emits "strictly_reflected" -- so the two sets agree on every
label this pipeline can actually produce. Do not add "strictly_reflected" here.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path

# suggestion_learning_format.py ships alongside this file.
_DATA_PREP_DIR = Path(__file__).resolve().parent

REFLECTED_LABELS = frozenset(
    {"reflected_directly", "reflected_by_alternative_fix", "partially_reflected"}
)
SCORE_THRESHOLD = 0.5
UNCLEAR_LABEL = "unclear"


def trace_passes(labels: list[str]) -> bool | None:
    """True/False if this trace's reward can be scored, None if it can't.

    "unclear" (reflect_incorporated_suggestions.py's fallback when a PR dataset
    can't be fetched -- a network failure, not a verdict) must not silently count
    as "not reflected": that would flip a trace to a negative reward purely because
    a fetch failed, and tell the skill every suggestion it made was NOT CONFIRMED
    when nobody ever actually judged them. So unclear rows are dropped from the
    denominator entirely; only if EVERY label on the trace is unclear -- meaning
    there is no real signal at all -- does this return None, telling the caller to
    drop the trace rather than score it 0.
    """
    scored = [label for label in labels if label != UNCLEAR_LABEL]
    if not scored:
        return None
    hits = sum(1 for label in scored if label in REFLECTED_LABELS)
    return (hits / len(scored)) > SCORE_THRESHOLD


def render_feedback(ai_items: list[dict], missed: list[dict]) -> str:
    lines = [
        "FEEDBACK ON THIS REVIEW (from the actual merged PR outcome, unknown to the "
        "reviewer at review time):",
        "",
        "This review's own suggestions, checked against what actually merged:",
    ]
    for item in ai_items:
        mark = "CONFIRMED" if item.get("reflection_label") in REFLECTED_LABELS else "NOT CONFIRMED"
        lines.append(f"- [{mark}] {item.get('body', '').strip()}")
    if missed:
        lines += [
            "",
            "Issues the merged PR confirmed as real that this review did not raise:",
        ]
        for issue in missed:
            where = issue.get("path") or ""
            line = issue.get("line")
            loc = f" ({where}:{line})" if where and line else (f" ({where})" if where else "")
            lines.append(f"- [MISSED]{loc} {issue.get('body', '').strip()}")
    return "\n".join(lines)


def _load_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _import_data_prep_module(name: str):
    data_prep_dir = str(_DATA_PREP_DIR)
    if data_prep_dir not in sys.path:
        sys.path.insert(0, data_prep_dir)
    return importlib.import_module(name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--reflected-suggestions", type=Path, required=True,
        help="reflected_suggestions.jsonl -- rollout_suggestions.jsonl rows plus a "
        "reflection_label each",
    )
    parser.add_argument(
        "--miss-list", type=Path, required=True,
        help="miss_list.json from build_miss_list.py: trace_id -> [{body, path, line, origin}]",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--drop-zero-findings",
        action="store_true",
        help="Drop traces where the agent produced no review at all (every row "
             "flagged zero_findings). Such a trace is a harness/agent failure, not "
             "review-quality signal, but it carries reflection_label=not_reflected "
             "on every row, so it scores NEGATIVE and clusters into 'your reviews "
             "were empty' lessons that have nothing to do with review quality -- 206 "
             "of gpt-5.4-mini's 487 negative traces in the parametric bootstrap. "
             "Off by default so existing runs reproduce byte-for-byte.",
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="trace manifest ({'cases': [{'trace_id': ...}, ...]}). When given, this "
             "run's trace count is reconciled against len(manifest['cases']) and a "
             "mismatch aborts the run instead of silently reporting a partial corpus "
             "as if it were the whole 638/639-PR train set.",
    )
    args = parser.parse_args(argv)

    rows = _load_jsonl(args.reflected_suggestions)
    miss_list: dict[str, list[dict]] = json.loads(args.miss_list.read_text(encoding="utf-8"))

    by_trace: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        trace_id = row.get("trace_id")
        if not trace_id:
            continue
        by_trace[trace_id].append(row)

    # Reconciliation counts every case the rollout actually produced, so it must be
    # taken BEFORE the zero-findings drop -- otherwise a clean run aborts because the
    # dropped empty reviews are still cases in the manifest.
    n_rolled_out = len(by_trace)

    n_dropped_zero_findings = 0
    if args.drop_zero_findings:
        empty_traces = {
            trace_id
            for trace_id, items in by_trace.items()
            if all(item.get("zero_findings") for item in items)
        }
        n_dropped_zero_findings = len(empty_traces)
        for trace_id in empty_traces:
            del by_trace[trace_id]
        # The corpus the refiner reads is written from `rows`, not `by_trace`, so it
        # has to be filtered too -- dropping a trace from the rewards but leaving it
        # in suggestion_learning_traces.jsonl would still feed it to the clusterer.
        rows = [row for row in rows if row.get("trace_id") not in empty_traces]

    binary_rewards: dict[str, bool] = {}
    reflection_feedback: dict[str, str] = {}
    n_dropped_unclear = 0
    for trace_id, items in by_trace.items():
        labels = [item.get("reflection_label") for item in items]
        passes = trace_passes(labels)
        if passes is None:
            n_dropped_unclear += 1
            continue
        binary_rewards[trace_id] = passes
        reflection_feedback[trace_id] = render_feedback(items, miss_list.get(trace_id, []))

    if args.manifest is not None:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        expected = len(manifest.get("cases") or [])
        actual = n_rolled_out
        if actual != expected:
            raise ValueError(
                f"trace-count reconciliation failed: {actual} traces in "
                f"{args.reflected_suggestions} vs {expected} cases in {args.manifest} -- "
                f"a silent gap here means the reported result is scored against fewer "
                f"than the full train set without anyone noticing."
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    suggestion_learning_format = _import_data_prep_module("suggestion_learning_format")
    suggestion_learning_format.write_learning_artifacts(rows, args.output_dir, split_test_size=None)

    (args.output_dir / "binary_rewards.json").write_text(
        json.dumps(binary_rewards, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "reflection_feedback.json").write_text(
        json.dumps(reflection_feedback, indent=2, sort_keys=True), encoding="utf-8"
    )

    n_traces = len(binary_rewards)
    n_positive = sum(1 for v in binary_rewards.values() if v)
    n_with_miss_list = sum(1 for tid in binary_rewards if miss_list.get(tid))
    print(
        f"traces={n_traces} positive={n_positive} negative={n_traces - n_positive} "
        f"with_miss_list={n_with_miss_list} dropped_all_unclear={n_dropped_unclear} "
        f"dropped_zero_findings={n_dropped_zero_findings}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
