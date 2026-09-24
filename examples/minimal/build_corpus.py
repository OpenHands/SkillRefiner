"""Regenerate this example's corpus from the golden fixtures.

The corpus checked in beside this file (``traces.jsonl`` + ``binary_rewards.json``)
is derived, not invented: every trace body is real captured SpreadsheetBench agent
text from ``tests/fixtures/golden/{positive,negative}/*/raw_content.txt`` -- the
same already-anonymized traces ``tests/test_golden_prompts.py`` pins the summarizer
prompt against. Running this script rewrites those two files in place; it exists so
the derivation is auditable and rerunnable rather than a claim in a README.

Two reductions keep the example fast and the artifact small:

  * SELECTION -- the ``N_PER_POLARITY`` smallest fixtures of each polarity, by
    ``raw_content.txt`` size. Six a side, because ``_run_partition`` skips any
    partition with fewer than three summaries and a three-trace partition leaves
    HDBSCAN nothing to separate.
  * TRUNCATION -- a trace longer than ``HEAD_CHARS + TAIL_CHARS`` keeps its head
    and its tail with an explicit marker between them. The head holds the task
    prompt and the tail holds the outcome, which is where a failure actually shows
    up; cutting only the head would hand the negative summarizer a trace with no
    visible failure.

Each trace becomes a single JSONL span, so ``serialize_trace`` reproduces the
fixture text (minus the cut) verbatim: ``_clean_span_output`` only rewrites spans
whose output parses as JSON, and these do not.

Usage:
  uv run python examples/minimal/build_corpus.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GOLDEN = HERE.parents[1] / "tests" / "fixtures" / "golden"

N_PER_POLARITY = 6
HEAD_CHARS = 3000
TAIL_CHARS = 3000
TRUNCATION_MARKER = "\n\n... [middle of trace elided for the bundled minimal example] ...\n\n"


def _truncate(text: str) -> str:
    if len(text) <= HEAD_CHARS + TAIL_CHARS:
        return text
    return text[:HEAD_CHARS] + TRUNCATION_MARKER + text[-TAIL_CHARS:]


def _selected(polarity: str) -> list[Path]:
    candidates = [
        d for d in sorted((GOLDEN / polarity).iterdir())
        if (d / "raw_content.txt").is_file()
    ]
    candidates.sort(key=lambda d: ((d / "raw_content.txt").stat().st_size, d.name))
    return candidates[:N_PER_POLARITY]


def build(out_dir: Path = HERE) -> tuple[Path, Path]:
    records: list[dict] = []
    rewards: dict[str, bool] = {}
    for polarity, reward in (("positive", True), ("negative", False)):
        for case in _selected(polarity):
            body = _truncate((case / "raw_content.txt").read_text(encoding="utf-8"))
            records.append(
                {
                    "trace_id": case.name,
                    "spans": [
                        {
                            "span_id": f"{case.name}-s1",
                            "name": "agent_rollout",
                            "input_text": "",
                            "output_text": body,
                            "start_time": "",
                            "end_time": "",
                        }
                    ],
                }
            )
            rewards[case.name] = reward

    traces_path = out_dir / "traces.jsonl"
    rewards_path = out_dir / "binary_rewards.json"
    with traces_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    rewards_path.write_text(json.dumps(rewards, indent=2) + "\n", encoding="utf-8")

    n_pos = sum(1 for v in rewards.values() if v)
    print(f"Wrote {len(records)} traces ({n_pos} positive, {len(rewards) - n_pos} negative)")
    print(f"  {traces_path}  ({traces_path.stat().st_size / 1024:.0f} KB)")
    print(f"  {rewards_path}")
    return traces_path, rewards_path


if __name__ == "__main__":
    build()
