"""Run agent rollouts on prepared DAPO-Math traces and plot results.

The input is produced by ``scripts/benchmarks/dapo/prepare_dapo_math.py``. This runner
loads ``splits/<split>.jsonl``, asks an agent to solve each prompt, extracts its final
answer, scores it against ``reward_model.ground_truth``, and writes both rollout
artifacts and summary plots. The agent is the OpenHands SDK's built-in ``Agent``, with
no shell by default (pass ``--use-terminal`` to give it one).

Example:
  uv run python scripts/benchmarks/dapo/run_dapo_agent.py \
      --data-dir results/dapo_math_17k \
      --split eval \
      --model gpt-5.4-mini \
      --llm-provider eval_proxy \
      --secrets-file .eval_proxy.secrets.json \
      --skill-file datasets/dapo/seed_skills/parametric_seed.md \
      --limit 200 \
      --concurrency 4 \
      --output-dir results/dapo_math_17k/openhands_gpt54mini_eval
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
import signal
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parents[2]  # scripts/ (holds _llm_config.py)
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Offline guard: skill_refiner/__init__.py sets LITELLM_LOCAL_MODEL_COST_MAP before
# litellm's import-time HTTPS GET to raw.githubusercontent.com. It must run ahead of the
# openhands/litellm imports below, and a plain `import skill_refiner` cannot -- ruff's
# isort sorts `openhands` first, which is precisely how the guard got skipped here.
__import__("skill_refiner")  # noqa: F401 - imported for its import-time side effect

import openhands.tools.preset.default  # noqa: E402,F401 - register built-in tools
from _llm_config import _resolve_llm_config  # noqa: E402
from openhands.sdk import LLM, Agent, Conversation, LocalWorkspace, Tool  # noqa: E402
from pydantic import SecretStr  # noqa: E402

DEFAULT_DATA_DIR = Path("results/dapo_math_17k")
DEFAULT_OUTPUT_DIR = Path("results/dapo_math_17k/openhands_eval")
# The paper's DAPO-Math initial skill is LLM-generated (see datasets/README.md).
DEFAULT_SKILL_FILE = Path("datasets/dapo/seed_skills/parametric_seed.md")
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_SECRETS_FILE = ".eval_proxy.secrets.json"
DEFAULT_MAX_ITER = 12

_SYSTEM_PROMPT = """You are an OpenHands math-solving agent.
Solve the user's math problem carefully. Your final response must end with
exactly one line:

Answer: <final answer>

Do not include the ground-truth answer unless you derived it yourself.
"""

_SKILL_PRELOAD_HEADER = (
    "# Preloaded Skill\n\n"
    "The following skill has been loaded for this DAPO-Math task. Follow its "
    "guidance when solving the problem and formatting the final answer."
)

_USAGE_INT_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "llm_calls",
)


@dataclass(frozen=True)
class DapoTask:
    trace_id: str
    split: str
    prompt: str
    ground_truth: str
    reward_style: str
    source_row_idx: int | None
    hf_index: str
    data_source: str
    ability: str


# --------------------------------------------------------------------------- #
# Input loading
# --------------------------------------------------------------------------- #


def _prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        chunks: list[str] = []
        for item in prompt:
            if isinstance(item, dict):
                role = str(item.get("role", "user")).upper()
                content = str(item.get("content", ""))
                chunks.append(f"{role}: {content}" if role else content)
            else:
                chunks.append(str(item))
        return "\n\n".join(chunk for chunk in chunks if chunk)
    return str(prompt or "")


def load_tasks(
    data_dir: Path,
    split: str,
    limit: int | None,
    include_ids: set[str] | None = None,
) -> list[DapoTask]:
    path = data_dir / "splits" / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"split file not found: {path}")

    tasks: list[DapoTask] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            trace_id = str(raw["trace_id"])
            if include_ids is not None and trace_id not in include_ids:
                continue
            tasks.append(
                DapoTask(
                    trace_id=trace_id,
                    split=str(raw.get("split", split)),
                    prompt=_prompt_text(raw.get("prompt", "")),
                    ground_truth=str(raw.get("ground_truth", "")),
                    reward_style=str(raw.get("reward_style", "")),
                    source_row_idx=raw.get("source_row_idx"),
                    hf_index=str(raw.get("hf_index", "")),
                    data_source=str(raw.get("data_source", "")),
                    ability=str(raw.get("ability", "")),
                )
            )
            # --limit applies AFTER id filtering, so it bounds the requested subset.
            if limit is not None and len(tasks) >= limit:
                break
    if include_ids is not None:
        missing = include_ids - {t.trace_id for t in tasks}
        if missing:
            print(
                f"  WARNING: {len(missing)} --include-ids not found in {split} split: "
                f"{', '.join(sorted(missing))}",
                file=sys.stderr,
            )
    return tasks


def load_include_ids(path: Path) -> set[str]:
    """Load trace_ids from a JSON list or newline-delimited .txt (# comments ignored)."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return {str(x).strip() for x in data if str(x).strip()}
    except json.JSONDecodeError:
        pass
    ids: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.add(line)
    return ids


# --------------------------------------------------------------------------- #
# Answer extraction and scoring
# --------------------------------------------------------------------------- #


def _strip_boxed_once(text: str) -> str:
    match = re.search(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if match:
        return match.group(1)
    return text


def extract_answer(text: str) -> str:
    """Extract the final answer from an agent response."""
    if not text:
        return ""

    answer_matches = re.findall(r"(?im)^\s*answer\s*:\s*(.+?)\s*$", text)
    if answer_matches:
        return answer_matches[-1].strip()

    boxed_matches = re.findall(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if boxed_matches:
        return boxed_matches[-1].strip()

    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return non_empty_lines[-1] if non_empty_lines else ""


def _normalize_latex(text: str) -> str:
    text = _strip_boxed_once(text)
    replacements = {
        r"\left": "",
        r"\right": "",
        r"\,": "",
        r"\;": "",
        r"\!": "",
        r"\cdot": "*",
        r"\times": "*",
        r"\pi": "pi",
        r"\infty": "infinity",
        "−": "-",
        "–": "-",
        "—": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"\1/\2", text)
    text = re.sub(r"\\dfrac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"\1/\2", text)
    text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", text)
    # Angle answers are unit-free in the DAPO ground truth (e.g. `108`), but agents
    # answer `108^\circ` / `108°` / `108 degrees`. Strip degree notation so an
    # otherwise-correct angle answer is not failed on decoration alone. Anchored to
    # the degree markers only, so symbolic answers like `pi`/`e` are untouched.
    text = re.sub(r"\^?\s*\{?\s*\\circ\s*\}?", "", text)
    text = re.sub(r"\\degrees?", "", text)
    text = re.sub(r"(?i)\s*degrees?\b", "", text)
    text = text.replace("°", "")
    return text


def normalize_answer(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"(?is)^\s*answer\s*:\s*", "", text)
    text = _normalize_latex(text)
    text = text.strip().strip(".$")
    text = text.replace("\\(", "").replace("\\)", "")
    text = text.replace("\\[", "").replace("\\]", "")
    text = text.replace("$", "")
    text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)
    return text.lower()


def _to_fraction(text: str) -> Fraction | None:
    value = normalize_answer(text)
    if not value:
        return None
    if re.fullmatch(r"[-+]?\d+", value):
        return Fraction(int(value), 1)
    if re.fullmatch(r"[-+]?\d+/[-+]?\d+", value):
        try:
            return Fraction(value)
        except ZeroDivisionError:
            return None
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)(?:e[-+]?\d+)?", value):
        try:
            return Fraction(Decimal(value))
        except (InvalidOperation, ValueError):
            return None
    return None


def is_correct(prediction: str, ground_truth: str) -> bool:
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(ground_truth)
    if pred_norm == gold_norm:
        return True

    pred_fraction = _to_fraction(prediction)
    gold_fraction = _to_fraction(ground_truth)
    if pred_fraction is not None and gold_fraction is not None:
        return pred_fraction == gold_fraction

    return False


def compose_system_prompt(skill_content: str | None) -> str:
    if skill_content and skill_content.strip():
        return f"{_SYSTEM_PROMPT.rstrip()}\n\n{_SKILL_PRELOAD_HEADER}\n\n{skill_content.strip()}\n"
    return _SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# OpenHands rollout helpers
# --------------------------------------------------------------------------- #


def _disable_interactive_pagers() -> None:
    os.environ.setdefault("PAGER", "cat")
    os.environ.setdefault("GIT_PAGER", "cat")
    os.environ.setdefault("MANPAGER", "cat")
    os.environ.setdefault("LESS", "-F -X -R")
    os.environ["GIT_TERMINAL_PROMPT"] = "0"
    os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")


def _raise_keyboard_interrupt_on_sigterm(signum, frame) -> None:
    """Make SIGTERM unwind like SIGINT does (Python already does this for SIGINT).

    Each task runs in its own spawned worker process. When the harness/OS kills
    a worker for memory pressure it sends SIGTERM (exit code 143), whose default
    disposition terminates the
    process immediately with no ``finally`` blocks run — orphaning that task's
    tmux pane pool (see ``_close_terminals``) instead of releasing it.
    Installing this handler makes SIGTERM raise KeyboardInterrupt the same way
    SIGINT already does by default, so the existing try/finally around
    ``conversation.run()`` still closes the tmux session on the way out.
    """
    raise KeyboardInterrupt("SIGTERM received")


def sweep_stale_pool_sessions(idle_seconds: int = 1800) -> None:
    """Kill ``openhands-pool-*`` tmux sessions idle longer than ``idle_seconds``.

    Defense-in-depth beyond ``_raise_keyboard_interrupt_on_sigterm``: that fix
    only prevents *this process's own* future kills from orphaning a tmux
    session — it can't retroactively clean up sessions left by an uncatchable
    SIGKILL, or by an unrelated process sharing this machine's ``openhands``
    tmux socket (this project's experiments routinely share a memory-
    constrained host with other concurrent sessions). A real in-flight
    rollout's session always has recent activity (every agent turn touches
    its pane), so anything idle past ``idle_seconds`` is safe to assume
    abandoned. Called once per batch run, not per instance — cheap, and
    keeps the tmux server from accumulating sessions across many runs.
    """
    import subprocess
    import time

    list_fmt = "#{session_name} #{session_activity}"
    try:
        result = subprocess.run(
            ["tmux", "-L", "openhands", "list-sessions", "-F", list_fmt],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return
    if result.returncode != 0:
        return  # no tmux server on this socket yet — nothing to sweep

    now = time.time()
    swept = 0
    for line in result.stdout.splitlines():
        name, _, activity = line.rpartition(" ")
        if not name.startswith("openhands-pool-"):
            continue
        try:
            age = now - float(activity)
        except ValueError:
            continue
        if age <= idle_seconds:
            continue
        try:
            subprocess.run(
                ["tmux", "-L", "openhands", "kill-session", "-t", name],
                capture_output=True,
                timeout=10,
            )
            swept += 1
        except (OSError, subprocess.TimeoutExpired):
            pass
    if swept:
        print(f"Swept {swept} stale tmux pool session(s) idle > {idle_seconds}s", file=sys.stderr)


def _close_terminals(agent: Agent) -> None:
    from contextlib import suppress

    try:
        tools = agent.tools_map
    except Exception:
        return
    for tool in tools.values():
        executor = getattr(tool, "executor", None)
        if executor is not None and hasattr(executor, "close"):
            with suppress(Exception):
                executor.close()


def _usage_from_metrics(metrics: Any) -> dict[str, Any]:
    tu = metrics.accumulated_token_usage
    usage = {
        "prompt_tokens": getattr(tu, "prompt_tokens", 0) if tu else 0,
        "completion_tokens": getattr(tu, "completion_tokens", 0) if tu else 0,
        "cache_read_tokens": getattr(tu, "cache_read_tokens", 0) if tu else 0,
        "cache_write_tokens": getattr(tu, "cache_write_tokens", 0) if tu else 0,
        "reasoning_tokens": getattr(tu, "reasoning_tokens", 0) if tu else 0,
        "cost": float(metrics.accumulated_cost or 0.0),
        "llm_calls": len(metrics.costs),
    }
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return usage


def _sum_usages(usages: list[dict[str, Any]]) -> dict[str, Any]:
    total = {k: sum(int(u.get(k, 0) or 0) for u in usages) for k in _USAGE_INT_KEYS}
    total["cost"] = sum(float(u.get("cost", 0.0) or 0.0) for u in usages)
    total["total_tokens"] = total["prompt_tokens"] + total["completion_tokens"]
    return total


def _event_text_parts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return parts
    return []


def final_response_from_events(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        action = event.get("action") or {}
        if isinstance(action, dict) and action.get("kind") == "FinishAction":
            message = action.get("message")
            if isinstance(message, str):
                return message
        if event.get("tool_name") == "finish":
            observation = event.get("observation") or {}
            if isinstance(observation, dict):
                parts = _event_text_parts(observation.get("content"))
                if parts:
                    return "\n".join(parts)
    for event in reversed(events):
        if event.get("source") == "agent":
            llm_message = event.get("llm_message") or {}
            if isinstance(llm_message, dict):
                parts = _event_text_parts(llm_message.get("content"))
                if parts:
                    return "\n".join(parts)
            action = event.get("action") or {}
            if isinstance(action, dict):
                message = action.get("message")
                if isinstance(message, str):
                    return message
            parts = _event_text_parts(event.get("thought"))
            if parts:
                return "\n".join(parts)
    return ""


def _write_task_files(work_dir: Path, task: DapoTask) -> None:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    (work_dir / "problem.txt").write_text(task.prompt, encoding="utf-8")
    # The ground truth is deliberately NOT written into the agent's working
    # directory: an agent with a shell (--use-terminal) would find it with `ls`.


def _task_prompt(task: DapoTask, use_terminal: bool) -> str:
    terminal_note = (
        "You may use Python in the terminal for arithmetic or symbolic checks. "
        if use_terminal
        else "Solve this directly in your response. "
    )
    return (
        f"DAPO-Math trace_id: {task.trace_id}\n\n"
        f"{task.prompt.strip()}\n\n"
        f"{terminal_note}Your final message must end with a line exactly like:\n"
        "Answer: <final answer>"
    )


def _existing_result(output_dir: Path, trace_id: str) -> dict[str, Any] | None:
    """A cached record to reuse under ``--skip-existing``, or None to (re)run this task.

    A record whose ``error`` field is non-empty (the rollout crashed — content-policy
    flag, proxy timeout, etc.) is treated as NOT existing, so it is retried on every
    subsequent invocation instead of being skipped forever. Delete nothing by hand:
    just rerun the same command with ``--skip-existing`` and errored tasks are the
    only ones re-attempted; already-correct/incorrect-but-completed ones stay skipped.
    """
    path = output_dir / "records" / f"{trace_id}.json"
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if record.get("error"):
        return None
    return record


def _run_one_task(
    *,
    task: DapoTask,
    output_dir: Path,
    cfg: Any,
    max_iter: int,
    skip_existing: bool,
    use_terminal: bool,
    system_prompt: str,
) -> dict[str, Any]:
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt_on_sigterm)
    existing = _existing_result(output_dir, task.trace_id) if skip_existing else None
    if existing is not None:
        print(f"  [{task.trace_id}] already complete — skipping")
        return existing

    work_dir = (output_dir / "work" / task.trace_id).resolve()
    events_path = output_dir / "rollouts" / f"{task.trace_id}.json"
    record_path = output_dir / "records" / f"{task.trace_id}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    _write_task_files(work_dir, task)

    llm = LLM(
        model=cfg.model,
        base_url=cfg.base_url,
        api_key=SecretStr(cfg.api_key or "dummy"),
        usage_id=f"dapo_{task.trace_id}",
        drop_params=True,
    )
    tools = [Tool(name="terminal")] if use_terminal else []
    agent = Agent(llm=llm, tools=tools, system_prompt=system_prompt)
    conversation = Conversation(
        agent=agent,
        workspace=LocalWorkspace(working_dir=str(work_dir)),
        max_iteration_per_run=max_iter,
        visualizer=None,
    )

    error = ""
    events: list[dict[str, Any]] = []
    try:
        conversation.send_message(_task_prompt(task, use_terminal))
        conversation.run()
        events = [event.model_dump(mode="json") for event in conversation.state.events]
    except Exception as exc:  # noqa: BLE001 - keep batch going
        error = repr(exc)
        print(f"  [{task.trace_id}] ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
    finally:
        _close_terminals(agent)

    final_response = final_response_from_events(events)
    prediction = extract_answer(final_response)
    correct = is_correct(prediction, task.ground_truth)
    usage = _usage_from_metrics(llm.metrics)

    record = {
        "trace_id": task.trace_id,
        "split": task.split,
        "source_row_idx": task.source_row_idx,
        "hf_index": task.hf_index,
        "data_source": task.data_source,
        "ability": task.ability,
        "reward_style": task.reward_style,
        "prompt": task.prompt,
        "ground_truth": task.ground_truth,
        "final_response": final_response,
        "prediction": prediction,
        "normalized_prediction": normalize_answer(prediction),
        "normalized_ground_truth": normalize_answer(task.ground_truth),
        "correct": correct,
        "error": error,
        "usage": usage,
        "events_path": str(events_path),
    }
    events_path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    status = "correct" if correct else "wrong"
    if error:
        status = "error"
    print(
        f"  [{task.trace_id}] {status}; pred={prediction!r}; gold={task.ground_truth!r}; "
        f"{usage['total_tokens']:,} tok; {usage['llm_calls']} calls"
    )
    return record


# --------------------------------------------------------------------------- #
# Output conversion and plots
# --------------------------------------------------------------------------- #


def _render_trace2skill_log(record: dict[str, Any]) -> str:
    outcome = "SUCCEED" if record.get("correct") else "FAILED"
    metadata = {
        "trace_id": record["trace_id"],
        "split": record.get("split"),
        "prediction": record.get("prediction"),
        "ground_truth": record.get("ground_truth"),
        "correct": record.get("correct"),
        "model_error": record.get("error", ""),
    }
    return (
        f"# Agent Trajectory: {record['trace_id']} ({outcome})\n\n"
        "## Task\n"
        f"{record.get('prompt', '').strip()}\n\n"
        "## Final Response\n"
        f"{record.get('final_response', '').strip()}\n\n"
        "## Extracted Answer\n"
        f"Prediction: {record.get('prediction', '')}\n\n"
        f"Ground truth: {record.get('ground_truth', '')}\n\n"
        "## Metadata\n"
        "```json\n"
        f"{json.dumps(metadata, indent=2, ensure_ascii=False)}\n"
        "```\n"
    )


def write_derived_outputs(output_dir: Path, records: list[dict[str, Any]]) -> None:
    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    answers = {record["trace_id"]: record.get("ground_truth", "") for record in records}
    rewards = {record["trace_id"]: bool(record.get("correct")) for record in records}
    (output_dir / "answers.json").write_text(json.dumps(answers, indent=2), encoding="utf-8")
    (output_dir / "binary_rewards.json").write_text(json.dumps(rewards, indent=2), encoding="utf-8")

    ablation_dir = output_dir / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    with (ablation_dir / "traces.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            trace_id = record["trace_id"]
            payload = {
                "trace_id": trace_id,
                "spans": [
                    {
                        "span_id": f"{trace_id}-task",
                        "name": "conversation.send_message",
                        "input_text": json.dumps({"message": record.get("prompt", "")}),
                        "output_text": f"TASK:\n{record.get('prompt', '')}",
                        "start_time": "",
                        "end_time": "",
                    },
                    {
                        "span_id": f"{trace_id}-openhands-final",
                        "name": "agent.final_answer",
                        "input_text": "",
                        "output_text": record.get("final_response", ""),
                        "start_time": "",
                        "end_time": "",
                    },
                ],
                "metadata": {
                    "trace_id": trace_id,
                    "split": record.get("split"),
                    "source_row_idx": record.get("source_row_idx"),
                    "hf_index": record.get("hf_index"),
                    "data_source": record.get("data_source"),
                    "ability": record.get("ability"),
                    "reward_style": record.get("reward_style"),
                    "prediction": record.get("prediction"),
                    "ground_truth": record.get("ground_truth"),
                    "correct": record.get("correct"),
                },
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    logs_dir = output_dir / "trace2skill" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        outcome = "SUCCEED" if record.get("correct") else "FAILED"
        path = logs_dir / f"dapo_math_openhands_{record['trace_id']}_{outcome}.md"
        path.write_text(_render_trace2skill_log(record), encoding="utf-8")


def compute_metrics(
    records: list[dict[str, Any]],
    model: str,
    split: str,
    expected_total: int | None = None,
    missing_trace_ids: list[str] | None = None,
) -> dict[str, Any]:
    total = len(records)
    correct = sum(1 for record in records if record.get("correct"))
    errors = sum(1 for record in records if record.get("error"))
    missing = missing_trace_ids or []
    usages = [record.get("usage", {}) for record in records]
    total_usage = _sum_usages(usages)
    by_reward_style: dict[str, dict[str, Any]] = {}
    for record in records:
        key = str(record.get("reward_style") or "unknown")
        bucket = by_reward_style.setdefault(key, {"total": 0, "correct": 0})
        bucket["total"] += 1
        bucket["correct"] += int(bool(record.get("correct")))
    for bucket in by_reward_style.values():
        bucket["accuracy"] = bucket["correct"] / bucket["total"] if bucket["total"] else 0.0

    return {
        "model": model,
        "split": split,
        "total": total,
        "expected_total": expected_total if expected_total is not None else total,
        "missing_total": len(missing),
        "missing_trace_ids": missing,
        "correct": correct,
        "incorrect": total - correct,
        "errors": errors,
        "accuracy": correct / total if total else 0.0,
        "total_usage": total_usage,
        "avg_total_tokens": total_usage["total_tokens"] / total if total else 0.0,
        "avg_cost": total_usage["cost"] / total if total else 0.0,
        "by_reward_style": by_reward_style,
    }


def plot_results(output_dir: Path, records: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    ordered = sorted(
        records,
        key=lambda r: (r.get("source_row_idx") is None, r.get("source_row_idx") or 0),
    )
    xs = list(range(1, len(ordered) + 1))
    correctness = [1 if record.get("correct") else 0 for record in ordered]
    cumulative = [sum(correctness[:i]) / i for i in xs]
    token_counts = [
        int((record.get("usage") or {}).get("total_tokens", 0) or 0) for record in ordered
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        ["Correct", "Incorrect", "Errors"],
        [metrics["correct"], metrics["incorrect"], metrics["errors"]],
        color=["#2ca02c", "#d62728", "#ff7f0e"],
    )
    ax.set_title(f"DAPO-Math OpenHands accuracy: {metrics['accuracy']:.1%}")
    ax.set_ylabel("Examples")
    fig.tight_layout()
    fig.savefig(plots_dir / "accuracy_summary.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, cumulative, color="#1f77b4", linewidth=2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Evaluated examples")
    ax.set_ylabel("Cumulative accuracy")
    ax.set_title("Cumulative accuracy over DAPO-Math eval order")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "cumulative_accuracy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.scatter(xs, correctness, color=["#2ca02c" if v else "#d62728" for v in correctness], s=20)
    ax.set_yticks([0, 1], ["wrong", "correct"])
    ax.set_xlabel("Evaluated examples")
    ax.set_title("Per-example correctness")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "per_example_correctness.png", dpi=180)
    plt.close(fig)

    if any(token_counts):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(token_counts, bins=min(20, max(1, len(token_counts) // 2)), color="#9467bd")
        ax.set_xlabel("Total tokens")
        ax.set_ylabel("Examples")
        ax.set_title("Token usage distribution")
        fig.tight_layout()
        fig.savefig(plots_dir / "token_usage_histogram.png", dpi=180)
        plt.close(fig)


def load_existing_records(output_dir: Path, tasks: list[DapoTask]) -> list[dict[str, Any]]:
    """Load completed per-task records from an output directory."""
    records_dir = output_dir / "records"
    order = {task.trace_id: i for i, task in enumerate(tasks)}
    records: list[dict[str, Any]] = []
    if not records_dir.exists():
        return records
    for path in records_dir.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  [{path.name}] invalid JSON: {exc}", file=sys.stderr)
            continue
        trace_id = str(record.get("trace_id", path.stem))
        if trace_id in order:
            records.append(record)
    records.sort(key=lambda record: order.get(str(record.get("trace_id")), len(order)))
    return records


def write_errors_log(output_dir: Path, records: list[dict[str, Any]]) -> Path:
    """Log every crashed task's id + error, so failures can be found and rerun later.

    ``run_dapo_agent.py --skip-existing`` (see ``_existing_result``) already
    retries these automatically on a subsequent invocation — this file is purely for
    visibility (how many crashed, on which tasks, with what error) rather than being
    required for the retry itself.
    """
    errored = [
        {"trace_id": record.get("trace_id"), "error": record["error"]}
        for record in records
        if record.get("error")
    ]
    path = output_dir / "errors.json"
    path.write_text(json.dumps(errored, indent=2), encoding="utf-8")
    return path


def finalize_outputs(
    *,
    output_dir: Path,
    records: list[dict[str, Any]],
    tasks: list[DapoTask],
    model: str,
    split: str,
) -> dict[str, Any]:
    """Write derived artifacts, metrics, and plots from completed records."""
    seen = {str(record.get("trace_id")) for record in records}
    missing_trace_ids = [task.trace_id for task in tasks if task.trace_id not in seen]
    write_derived_outputs(output_dir, records)
    write_errors_log(output_dir, records)
    metrics = compute_metrics(
        records,
        model,
        split,
        expected_total=len(tasks),
        missing_trace_ids=missing_trace_ids,
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_results(output_dir, records, metrics)
    return metrics


def print_summary(output_dir: Path, metrics: dict[str, Any]) -> None:
    """Print a concise summary of written artifacts."""
    print("\nResults:")
    print(
        f"  Accuracy          : {metrics['correct']}/{metrics['total']} = {metrics['accuracy']:.1%}"
    )
    if metrics.get("missing_total"):
        print(
            f"  Missing records   : {metrics['missing_total']}/"
            f"{metrics.get('expected_total', metrics['total'])}"
        )
        print(f"  Missing trace IDs : {', '.join(metrics['missing_trace_ids'])}")
    print(f"  Errors            : {metrics['errors']}")
    print(f"  Prompt tokens     : {metrics['total_usage']['prompt_tokens']:,}")
    print(f"  Completion tokens : {metrics['total_usage']['completion_tokens']:,}")
    print(f"  Total tokens      : {metrics['total_usage']['total_tokens']:,}")
    print(f"  LLM calls         : {metrics['total_usage']['llm_calls']:,}")
    if metrics["total_usage"]["cost"]:
        print(f"  Cost              : ${metrics['total_usage']['cost']:.4f}")
    print("\nWrote:")
    print(f"  {output_dir / 'metrics.json'}")
    print(f"  {output_dir / 'predictions.jsonl'}")
    print(f"  {output_dir / 'binary_rewards.json'}")
    print(f"  {output_dir / 'plots'}")


def run(
    *,
    data_dir: Path,
    split: str,
    output_dir: Path,
    cfg: Any,
    limit: int | None,
    concurrency: int,
    max_iter: int,
    skip_existing: bool,
    use_terminal: bool,
    skill_file: Path | None,
    include_ids: set[str] | None = None,
) -> None:
    _disable_interactive_pagers()
    sweep_stale_pool_sessions()
    tasks = load_tasks(data_dir, split, limit, include_ids)
    if not tasks:
        raise RuntimeError(f"no tasks loaded from {data_dir}/splits/{split}.jsonl")

    skill_content = None
    if skill_file is not None:
        if not skill_file.exists():
            raise FileNotFoundError(f"skill file does not exist: {skill_file}")
        skill_content = skill_file.read_text(encoding="utf-8")
    system_prompt = compose_system_prompt(skill_content)

    print(f"Data dir          : {data_dir}")
    print(f"Split             : {split}")
    print(f"Tasks             : {len(tasks)}")
    print(f"Model             : {cfg.model}")
    print(f"Base URL          : {cfg.base_url}")
    print(f"Output dir        : {output_dir}")
    print(f"Skill             : {skill_file or '(no skill / baseline)'}")
    print(f"Use terminal      : {use_terminal}")
    print(f"Concurrency       : {concurrency}   max_iter/task: {max_iter}")

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=concurrency, mp_context=ctx) as pool:
        future_to_id = {
            pool.submit(
                _run_one_task,
                task=task,
                output_dir=output_dir,
                cfg=cfg,
                max_iter=max_iter,
                skip_existing=skip_existing,
                use_terminal=use_terminal,
                system_prompt=system_prompt,
            ): task.trace_id
            for task in tasks
        }
        for future in as_completed(future_to_id):
            trace_id = future_to_id[future]
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                print(f"  [{trace_id}] WORKER CRASHED: {exc}", file=sys.stderr)
                traceback.print_exc()

    order = {task.trace_id: i for i, task in enumerate(tasks)}
    results.sort(key=lambda record: order.get(record["trace_id"], len(order)))
    metrics = finalize_outputs(
        output_dir=output_dir,
        records=results,
        tasks=tasks,
        model=cfg.model,
        split=split,
    )
    print_summary(output_dir, metrics)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, metavar="DIR")
    parser.add_argument("--split", choices=("train", "eval"), default="eval")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, metavar="DIR")
    parser.add_argument(
        "--skill-file",
        type=Path,
        default=DEFAULT_SKILL_FILE,
        metavar="SKILL.md",
        help="Skill markdown to preload into the agent system prompt.",
    )
    parser.add_argument(
        "--no-skill",
        action="store_true",
        help="Run a no-skill baseline instead of preloading --skill-file.",
    )
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--llm-provider", default=os.environ.get("LLM_PROVIDER", "eval_proxy")
    )
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", ""))
    parser.add_argument("--api-key", default="")
    parser.add_argument(
        "--secrets-file",
        default=os.environ.get("SKILL_REFINER_SECRETS_FILE", DEFAULT_SECRETS_FILE),
    )
    parser.add_argument(
        "--include-ids-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="Restrict this run to the trace_ids in this file (JSON list or "
        "newline-delimited .txt; blank lines and #-comments ignored). --limit "
        "applies after filtering. Missing ids are warned about, not fatal.",
    )
    parser.add_argument("--limit", type=int, default=100, metavar="N")
    parser.add_argument("--concurrency", type=int, default=4, metavar="N")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER, metavar="N")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse completed records.")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Regenerate metrics/plots from existing records without launching rollouts.",
    )
    parser.add_argument(
        "--use-terminal",
        action="store_true",
        help="Expose the terminal tool. Disabled by default to avoid tmux fan-out failures.",
    )
    args = parser.parse_args()

    if args.concurrency < 1:
        print("ERROR: --concurrency must be >= 1", file=sys.stderr)
        sys.exit(1)

    include_ids: set[str] | None = None
    if args.include_ids_file is not None:
        if not args.include_ids_file.exists():
            print(f"ERROR: --include-ids-file not found: {args.include_ids_file}", file=sys.stderr)
            sys.exit(1)
        include_ids = load_include_ids(args.include_ids_file)
        if not include_ids:
            print(f"ERROR: --include-ids-file is empty: {args.include_ids_file}", file=sys.stderr)
            sys.exit(1)
        print(f"Include ids       : {len(include_ids)} from {args.include_ids_file}")

    if args.aggregate_only:
        tasks = load_tasks(args.data_dir, args.split, args.limit, include_ids)
        records = load_existing_records(args.output_dir, tasks)
        if not records:
            print(
                f"ERROR: no existing records found under {args.output_dir / 'records'}",
                file=sys.stderr,
            )
            sys.exit(1)
        metrics = finalize_outputs(
            output_dir=args.output_dir,
            records=records,
            tasks=tasks,
            model=args.model,
            split=args.split,
        )
        print_summary(args.output_dir, metrics)
        return

    skill_file = None if args.no_skill else args.skill_file

    cfg = _resolve_llm_config(
        model=args.model,
        provider=args.llm_provider,
        base_url=args.base_url,
        api_key=args.api_key,
        secrets_file=args.secrets_file,
    )

    run(
        data_dir=args.data_dir,
        split=args.split,
        output_dir=args.output_dir,
        cfg=cfg,
        limit=args.limit,
        concurrency=args.concurrency,
        max_iter=args.max_iter,
        skip_existing=args.skip_existing,
        use_terminal=args.use_terminal,
        skill_file=skill_file,
        include_ids=include_ids,
    )


if __name__ == "__main__":
    main()
