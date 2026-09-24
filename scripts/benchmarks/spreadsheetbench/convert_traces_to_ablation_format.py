"""Convert SpreadsheetBench openhands-sdk rollouts into the summarizer ablation's
input format.

Reads the native openhands event dumps written by
``spreadsheetbench_agent_runner.py`` (one JSON file per instance under
``<run-dir>/logs/<instance_id>.json``, each a list of ``event.model_dump()``
dicts) plus the official evaluator's ``eval_official_results.json``, and emits:

  - ``traces.jsonl``      — one line per instance: ``{"trace_id", "spans": [...]}``,
                            the exact schema ``RefineConfig.raw_jsonl``
                            expects (``_jsonl_trace_to_trace`` keeps
                            spans with a non-empty ``output_text``).
  - ``binary_rewards.json`` — ``{instance_id: bool}`` from the evaluator's
                            per-instance ``success`` flag, the schema
                            ``--binary-rewards`` expects.

The ``SystemPromptEvent`` is dropped from every trace: under this runtime the
system prompt carries the full skill text, and leaking it into the (skill-blind)
summarizer would waste tokens and bias summaries toward parroting the skill back
instead of describing execution behavior.

Usage:
  uv run python scripts/benchmarks/spreadsheetbench/convert_traces_to_ablation_format.py \\
      --run-dir results/spreadsheetbench/train \\
      --output-dir results/spreadsheetbench
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Offline guard: skill_refiner/__init__.py sets LITELLM_LOCAL_MODEL_COST_MAP before
# litellm's import-time HTTPS GET to raw.githubusercontent.com. It must run ahead of the
# openhands/litellm imports below, and a plain `import skill_refiner` cannot -- ruff's
# isort sorts `openhands` first, which is precisely how the guard got skipped here.
__import__("skill_refiner")  # noqa: F401 - imported for its import-time side effect

from openhands.sdk.event import SystemPromptEvent  # noqa: E402

from skill_refiner.trace.serialization import deserialize_trace  # noqa: E402
from skill_refiner.trace.store import Trace, TraceEvent  # noqa: E402


def render_message_text(message) -> str:
    """Render an openhands ``Message`` to readable, role-tagged text.

    Joins every text content block and appends any tool calls as
    ``[tool call] <name>(<arguments>)``. Returns ``""`` when the message has no
    renderable content (so the caller can drop empty spans).
    """
    parts: list[str] = []
    for content in message.content or []:
        text = getattr(content, "text", None)
        if text and text.strip():
            parts.append(text)
    for tool_call in message.tool_calls or []:
        name = getattr(tool_call, "name", "") or ""
        arguments = getattr(tool_call, "arguments", "") or ""
        parts.append(f"[tool call] {name}({arguments})")
    body = "\n".join(p for p in parts if p and p.strip())
    if not body.strip():
        return ""
    return f"{message.role.upper()}: {body}"


def event_dicts_to_spans(event_dicts: list[dict]) -> list[dict]:
    """Turn native openhands event dumps into ablation spans.

    Deserializes each dict into a typed SDK event (reusing
    ``skill_refiner.trace.serialization.deserialize_trace``, which keeps only
    ``LLMConvertibleEvent`` and silently drops the rest), drops the
    ``SystemPromptEvent``, renders each remaining event to text, and yields one
    span per non-empty render.
    """
    events = deserialize_trace(Trace([TraceEvent(d) for d in event_dicts]))
    spans: list[dict] = []
    for idx, event in enumerate(events):
        if isinstance(event, SystemPromptEvent):
            continue
        text = render_message_text(event.to_llm_message())
        if not text.strip():
            continue
        spans.append(
            {
                "span_id": getattr(event, "id", "") or f"span_{idx}",
                "name": type(event).__name__,
                "input_text": "",
                "output_text": text,
                "start_time": getattr(event, "timestamp", "") or "",
                "end_time": "",
            }
        )
    return spans


def build_trace_record(trace_id: str, event_dicts: list[dict]) -> dict:
    """Assemble one ``traces.jsonl`` record for an instance."""
    return {"trace_id": trace_id, "spans": event_dicts_to_spans(event_dicts)}


def rewards_from_eval_results(eval_json: dict) -> dict[str, bool]:
    """Extract ``{instance_id: bool}`` from an eval_official_results.json blob.

    Ids are coerced to ``str`` because dataset.json stores numeric ids as ints
    while everything else in the pipeline keys on the string form.
    """
    rewards: dict[str, bool] = {}
    for result in eval_json.get("results", []):
        instance_id = str(result.get("id"))
        rewards[instance_id] = bool(result.get("success", False))
    return rewards


# --------------------------------------------------------------------------- #
# CLI orchestration
# --------------------------------------------------------------------------- #


def _load_event_dicts(log_path: Path) -> list[dict]:
    data = json.loads(log_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{log_path} is not a JSON list of events")
    return data


def convert(run_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    logs_dir = run_dir / "logs"
    eval_path = run_dir / "eval_official_results.json"
    if not logs_dir.is_dir():
        print(f"ERROR: no logs dir at {logs_dir}", file=sys.stderr)
        sys.exit(1)
    if not eval_path.exists():
        print(f"ERROR: no eval_official_results.json at {eval_path}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    traces_path = output_dir / "traces.jsonl"
    rewards_path = output_dir / "binary_rewards.json"

    rewards = rewards_from_eval_results(json.loads(eval_path.read_text(encoding="utf-8")))

    n_written = 0
    with traces_path.open("w", encoding="utf-8") as out:
        for log_path in sorted(logs_dir.glob("*.json")):
            trace_id = log_path.stem
            record = build_trace_record(trace_id, _load_event_dicts(log_path))
            if not record["spans"]:
                print(f"  [skip] {trace_id}: no non-empty spans", file=sys.stderr)
                continue
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    rewards_path.write_text(json.dumps(rewards, indent=2), encoding="utf-8")
    print(f"Wrote {n_written} traces → {traces_path}")
    print(f"Wrote {len(rewards)} rewards → {rewards_path}")
    return traces_path, rewards_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, required=True, metavar="DIR",
                        help="Runner output dir (contains logs/ + eval_official_results.json)")
    parser.add_argument("--output-dir", type=Path, required=True, metavar="DIR",
                        help="Where to write traces.jsonl + binary_rewards.json")
    args = parser.parse_args()
    convert(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
