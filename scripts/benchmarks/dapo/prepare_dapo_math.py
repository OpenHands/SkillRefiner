"""Prepare DAPO-Math samples in Trace2Skill and summarizer-ablation formats.

The HuggingFace dataset ``BytedTsinghua-SIA/DAPO-Math-17k`` is a supervised math
prompt dataset, not an agent-rollout dataset. This script therefore emits
"oracle" successful trajectories: each sampled item becomes a task plus the
reward-model ground-truth answer. That shape is useful for exercising the
Trace2Skill and ablation pipelines, but it is not a substitute for real model
failure/success rollouts if polarity analysis is required.

Default output under ``results/dapo_math_17k``:

  splits/train.jsonl
  splits/eval.jsonl
      Normalized source examples.

  trace2skill/train/logs/*.md
  trace2skill/train/work/<trace_id>_init/{problem.txt,answer.txt,metadata.json}
  trace2skill/eval/logs/*.md
  trace2skill/eval/work/<trace_id>_init/{problem.txt,answer.txt,metadata.json}
      Markdown logs named ``*_<trace_id>_SUCCEED.md`` plus per-instance work dirs.

  ablation/train/traces.jsonl
  ablation/train/binary_rewards.json
  ablation/train/answers.json
  ablation/eval/traces.jsonl
  ablation/eval/binary_rewards.json
  ablation/eval/answers.json
      ``RefineConfig(raw_jsonl=..., binary_rewards=...)`` compatible traces and reward maps.

Usage:
  uv run python scripts/benchmarks/dapo/prepare_dapo_math.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DATASET_NAME = "BytedTsinghua-SIA/DAPO-Math-17k"
DATASET_CONFIG = "default"
DATASET_SPLIT = "train"
DATASETS_SERVER = "https://datasets-server.huggingface.co"
DEFAULT_OUTPUT_DIR = Path("results/dapo_math_17k")
DEFAULT_TRAIN_SIZE = 400
DEFAULT_EVAL_SIZE = 200  # the paper's held-out DAPO-Math set
DEFAULT_PAGE_SIZE = 100


@dataclass(frozen=True)
class DapoExample:
    """Normalized DAPO-Math item used by both output renderers."""

    split: str
    trace_id: str
    source_dataset: str
    source_row_idx: int
    data_source: str
    ability: str
    prompt: list[dict[str, str]]
    ground_truth: str
    reward_style: str
    hf_index: str
    raw: dict[str, Any]


def _get_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "skill-lab-dapo-prep/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        message = f"HuggingFace request failed: {exc.code} {exc.reason}: {detail}"
        raise RuntimeError(message) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"HuggingFace request failed: {exc.reason}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object from {url}")
    return payload


def fetch_dataset_rows(
    *,
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    page_size: int,
) -> list[dict[str, Any]]:
    """Fetch rows from the HuggingFace datasets-server API without extra deps."""
    if length <= 0:
        return []
    rows: list[dict[str, Any]] = []
    while len(rows) < length:
        page_length = min(page_size, length - len(rows))
        params = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset + len(rows),
                "length": page_length,
            }
        )
        payload = _get_json(f"{DATASETS_SERVER}/rows?{params}")
        page_rows = payload.get("rows", [])
        if not isinstance(page_rows, list):
            raise RuntimeError("HuggingFace rows response did not contain a rows list")
        if not page_rows:
            total = payload.get("num_rows_total", "unknown")
            raise RuntimeError(
                f"Only fetched {len(rows)} of {length} requested rows "
                f"from offset {offset}; dataset total is {total}"
            )
        rows.extend(page_rows)
    return rows


def normalize_example(row_payload: dict[str, Any], split: str, dataset: str) -> DapoExample:
    """Normalize one datasets-server row payload into stable local fields."""
    row = row_payload.get("row", {})
    if not isinstance(row, dict):
        raise ValueError(f"Row payload missing dict row: {row_payload!r}")

    source_row_idx = int(row_payload.get("row_idx", -1))
    reward_model = row.get("reward_model", {}) if isinstance(row.get("reward_model"), dict) else {}
    extra_info = row.get("extra_info", {}) if isinstance(row.get("extra_info"), dict) else {}
    prompt = row.get("prompt", [])
    if not isinstance(prompt, list):
        prompt = []

    normalized_prompt: list[dict[str, str]] = []
    for message in prompt:
        if not isinstance(message, dict):
            continue
        normalized_prompt.append(
            {
                "role": str(message.get("role", "user")),
                "content": str(message.get("content", "")),
            }
        )

    return DapoExample(
        split=split,
        trace_id=f"dapo-math-{source_row_idx:07d}",
        source_dataset=dataset,
        source_row_idx=source_row_idx,
        data_source=str(row.get("data_source", "")),
        ability=str(row.get("ability", "")),
        prompt=normalized_prompt,
        ground_truth=str(reward_model.get("ground_truth", "")),
        reward_style=str(reward_model.get("style", "")),
        hf_index=str(extra_info.get("index", "")),
        raw=row,
    )


def prompt_text(example: DapoExample) -> str:
    """Render the chat-style prompt as readable text."""
    rendered: list[str] = []
    for message in example.prompt:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        if content.strip():
            rendered.append(f"{role}: {content}")
    return "\n\n".join(rendered)


def example_metadata(example: DapoExample) -> dict[str, Any]:
    """Small metadata record shared by manifests and work dirs."""
    return {
        "trace_id": example.trace_id,
        "split": example.split,
        "source_dataset": example.source_dataset,
        "source_row_idx": example.source_row_idx,
        "hf_index": example.hf_index,
        "data_source": example.data_source,
        "ability": example.ability,
        "reward_style": example.reward_style,
    }


def render_trace2skill_markdown(example: DapoExample) -> str:
    """Render one supervised example as a Trace2Skill-style markdown trajectory."""
    action = {
        "name": "solve_math_problem",
        "arguments": {
            "ability": example.ability,
            "data_source": example.data_source,
            "reward_style": example.reward_style,
        },
    }
    metadata = example_metadata(example)
    lines = [
        f"# Agent Trajectory: {example.trace_id} (SUCCEED)",
        "",
        "## Task",
        f"Task: {prompt_text(example)}",
        "",
        "### Step 1",
        "Thought: Solve the math problem and provide the final answer in the requested format.",
        "Action:",
        "```json",
        json.dumps(action, ensure_ascii=False),
        "```",
        "",
        f"Observation: Reward model ground-truth answer: {example.ground_truth}",
        "",
        f"Final: Answer: {example.ground_truth}",
        "",
        "## Metadata",
        "```json",
        json.dumps(metadata, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


def ablation_record(example: DapoExample) -> dict[str, Any]:
    """Build one ``RefineConfig.raw_jsonl`` compatible record."""
    prompt = prompt_text(example)
    return {
        "trace_id": example.trace_id,
        "spans": [
            {
                "span_id": f"{example.trace_id}-task",
                "name": "conversation.send_message",
                "input_text": json.dumps({"message": prompt}, ensure_ascii=False),
                "output_text": f"TASK:\n{prompt}",
                "start_time": "",
                "end_time": "",
            },
            {
                "span_id": f"{example.trace_id}-oracle-answer",
                "name": "oracle.reward_model",
                "input_text": "",
                "output_text": (
                    f"GROUND TRUTH ANSWER ({example.reward_style}):\nAnswer: {example.ground_truth}"
                ),
                "start_time": "",
                "end_time": "",
            },
        ],
        "metadata": example_metadata(example),
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def write_trace2skill_split(
    output_dir: Path,
    split: str,
    examples: list[DapoExample],
) -> dict[str, str]:
    split_dir = output_dir / "trace2skill" / split
    logs_dir = split_dir / "logs"
    work_dir = split_dir / "work"
    logs_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    for example in examples:
        log_path = logs_dir / f"dapo_math_agent_{example.trace_id}_SUCCEED.md"
        log_path.write_text(render_trace2skill_markdown(example), encoding="utf-8")

        instance_work_dir = work_dir / f"{example.trace_id}_init"
        instance_work_dir.mkdir(parents=True, exist_ok=True)
        problem = prompt_text(example) + "\n"
        (instance_work_dir / "problem.txt").write_text(problem, encoding="utf-8")
        (instance_work_dir / "answer.txt").write_text(
            f"Answer: {example.ground_truth}\n", encoding="utf-8"
        )
        write_json(instance_work_dir / "metadata.json", example_metadata(example))

    return {
        "logs_dir": str(logs_dir),
        "work_dir": str(work_dir),
    }


def write_ablation_split(
    output_dir: Path,
    split: str,
    examples: list[DapoExample],
) -> dict[str, str]:
    split_dir = output_dir / "ablation" / split
    traces_path = split_dir / "traces.jsonl"
    rewards_path = split_dir / "binary_rewards.json"
    answers_path = split_dir / "answers.json"

    write_jsonl(traces_path, [ablation_record(example) for example in examples])
    write_json(rewards_path, {example.trace_id: True for example in examples})
    write_json(answers_path, {example.trace_id: example.ground_truth for example in examples})

    return {
        "traces_jsonl": str(traces_path),
        "binary_rewards": str(rewards_path),
        "answers": str(answers_path),
    }


def write_split_records(output_dir: Path, split: str, examples: list[DapoExample]) -> str:
    path = output_dir / "splits" / f"{split}.jsonl"
    write_jsonl(path, [asdict(example) for example in examples])
    return str(path)


def build_dataset(
    *,
    output_dir: Path,
    dataset: str,
    config: str,
    source_split: str,
    offset: int,
    train_size: int,
    eval_size: int,
    page_size: int,
    overwrite: bool,
) -> dict[str, Any]:
    """Fetch, normalize, and write DAPO-Math train/eval samples."""
    prepare_output_dir(output_dir, overwrite)
    total_size = train_size + eval_size
    rows = fetch_dataset_rows(
        dataset=dataset,
        config=config,
        split=source_split,
        offset=offset,
        length=total_size,
        page_size=page_size,
    )

    train_rows = rows[:train_size]
    eval_rows = rows[train_size:]
    split_examples = {
        "train": [normalize_example(row, "train", dataset) for row in train_rows],
        "eval": [normalize_example(row, "eval", dataset) for row in eval_rows],
    }

    manifest: dict[str, Any] = {
        "dataset": dataset,
        "config": config,
        "source_split": source_split,
        "offset": offset,
        "train_size": train_size,
        "eval_size": eval_size,
        "note": (
            "DAPO-Math is supervised prompt data. Trace2Skill logs and ablation traces "
            "are oracle SUCCEED examples using reward_model.ground_truth; binary rewards "
            "are therefore true for every emitted trace."
        ),
        "splits": {},
    }

    for split, examples in split_examples.items():
        split_manifest = {
            "count": len(examples),
            "first_trace_id": examples[0].trace_id if examples else None,
            "last_trace_id": examples[-1].trace_id if examples else None,
            "records_jsonl": write_split_records(output_dir, split, examples),
            "trace2skill": write_trace2skill_split(output_dir, split, examples),
            "ablation": write_ablation_split(output_dir, split, examples),
        }
        manifest["splits"][split] = split_manifest

    write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--config", default=DATASET_CONFIG)
    parser.add_argument("--source-split", default=DATASET_SPLIT)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--eval-size", type=int, default=DEFAULT_EVAL_SIZE)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace --output-dir if it exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_size < 0 or args.eval_size < 0:
        print("ERROR: --train-size and --eval-size must be non-negative", file=sys.stderr)
        sys.exit(2)
    if args.offset < 0:
        print("ERROR: --offset must be non-negative", file=sys.stderr)
        sys.exit(2)
    if args.page_size <= 0:
        print("ERROR: --page-size must be positive", file=sys.stderr)
        sys.exit(2)

    try:
        manifest = build_dataset(
            output_dir=args.output_dir,
            dataset=args.dataset,
            config=args.config,
            source_split=args.source_split,
            offset=args.offset,
            train_size=args.train_size,
            eval_size=args.eval_size,
            page_size=args.page_size,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    train = manifest["splits"]["train"]
    eval_split = manifest["splits"]["eval"]
    print(f"Wrote DAPO-Math dataset to {args.output_dir}")
    print(
        f"  train: {train['count']} samples ({train['first_trace_id']}..{train['last_trace_id']})"
    )
    print(
        f"  eval:  {eval_split['count']} samples "
        f"({eval_split['first_trace_id']}..{eval_split['last_trace_id']})"
    )
    print(f"  manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
