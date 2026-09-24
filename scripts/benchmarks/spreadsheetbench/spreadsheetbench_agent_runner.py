"""Run the SpreadsheetBench agent on a train/eval split using openhands-sdk.

This replaces Trace2Skill's own react_agent/bash-loop runner: the agent doing
the spreadsheet editing is built on openhands.sdk (Agent + Conversation + a
terminal tool), using the system prompt from
``Trace2Skill/spreadsheet_agent/system_prompt/cli_only_full_system_v1.txt``
(transformed for native tool-calling — see spreadsheetbench_common.build_system_prompt).

Everything about the benchmark other than the agent runtime stays tied to
Trace2Skill: dataset loading, spreadsheet-dir resolution, the task-prompt field
layout, and official scoring (evaluate_with_official.py, run as a subprocess).

Splits (fixed positional slices of the pre-ordered verified-400 dataset, exactly
as the Trace2Skill paper uses):
  train -> instances [0:200]   (generate initial traces to refine from)
  eval  -> instances [200:400] (held-out evaluation of a skill)

Per instance it writes:
  <output-dir>/<spreadsheet_path>/<name>_output.xlsx   (evaluate_with_official layout)
  <output-dir>/logs/<instance_id>.json                 (native openhands event dump)
and, after all rollouts, <output-dir>/eval_official_results.json.

Usage:
  # Train split, xlsx skill preloaded (produces traces to refine from):
  uv run python scripts/benchmarks/spreadsheetbench/spreadsheetbench_agent_runner.py \\
      --split train --skill-file baselines/trace2skill/spreadsheet_agent/skills/xlsx/SKILL.md \\
      --output-dir results/spreadsheetbench/train --concurrency 4

  # Held-out eval of a candidate skill:
  uv run python scripts/benchmarks/spreadsheetbench/spreadsheetbench_agent_runner.py \\
      --split eval --skill-file results/.../combined/proposed_skill.md \\
      --output-dir results/spreadsheetbench/eval_candidate

  # No-skill baseline (omit --skill-file):
  uv run python scripts/benchmarks/spreadsheetbench/spreadsheetbench_agent_runner.py \\
      --split eval --output-dir results/spreadsheetbench/eval_noskill
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import signal
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# REPO_ROOT here is the artifact root (three levels up from
# scripts/benchmarks/spreadsheetbench/). Trace2Skill ships inside this artifact at
# baselines/trace2skill/ (not as a sibling "Trace2Skill/" checkout) — this also has
# to be on sys.path below, since spreadsheetbench_support.py (imported further down)
# lives there.
REPO_ROOT = Path(__file__).resolve().parents[3]
TRACE2SKILL = REPO_ROOT / "baselines" / "trace2skill"
_SCRIPTS_DIR = Path(__file__).resolve().parents[2]  # scripts/ (holds _llm_config.py)
_THIS_DIR = Path(__file__).resolve().parent

for _p in (str(_THIS_DIR), str(_SCRIPTS_DIR), str(TRACE2SKILL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Offline guard: skill_refiner/__init__.py sets LITELLM_LOCAL_MODEL_COST_MAP before
# litellm's import-time HTTPS GET to raw.githubusercontent.com. It must run ahead of the
# openhands/litellm imports below, and a plain `import skill_refiner` cannot -- ruff's
# isort sorts `openhands` first, which is precisely how the guard got skipped here.
__import__("skill_refiner")  # noqa: F401 - imported for its import-time side effect

# Registers the built-in tools ("terminal", "file_editor", ...) so Tool(name=...)
# resolves. Import for the side effect.
import openhands.tools.preset.default  # noqa: E402,F401
from _llm_config import _register_extra_models, _resolve_llm_config  # noqa: E402
from openhands.sdk import (  # noqa: E402
    LLM,
    Agent,
    Conversation,
    LocalWorkspace,
    Tool,
)
from pydantic import SecretStr  # noqa: E402
from spreadsheetbench_common import (  # noqa: E402
    build_system_prompt,
    build_task_prompt,
    find_input_files,
    output_name_for_input,
    spreadsheet_content_preview,
)
from spreadsheetbench_support import find_spreadsheet_dir, load_dataset  # noqa: E402

DEFAULT_MODEL = "openai/MiniMaxAI/MiniMax-M2.7"
# Empty by design: no deployment URL is baked in here. _resolve_llm_config
# falls back to the LLM_BASE_URL env var or artifact.local.json's
# llm_base_urls.eval_proxy, and fails loudly if none is set -- see
# artifact.local.example.json / README "Local configuration".
DEFAULT_BASE_URL = ""
DEFAULT_DATA_PATH = (
    TRACE2SKILL / "data" / "spreadsheetbench_verified" / "spreadsheetbench_verified_400"
)
CLI_ONLY_PROMPT = (
    TRACE2SKILL / "spreadsheet_agent" / "system_prompt" / "cli_only_full_system_v1.txt"
)
EVALUATOR = TRACE2SKILL / "evaluate_with_official.py"
DEFAULT_SECRETS_FILE = ".llm.secrets.json"

SPLIT_RANGES = {"train": (0, 200), "eval": (200, 400)}
DEFAULT_MAX_ITER = 100

_SKILL_PRELOAD_HEADER = (
    "# Preloaded Skill\n\n"
    "The following skill has been loaded for this task. Follow its guidance. Its "
    "supporting files (e.g. `recalc.py` and any `references/`) have been copied "
    "into your working directory, so you can run them directly from there."
)


def compose_system_prompt(cli_only_raw: str, skill_content: str | None) -> str:
    """Base cli_only prompt (transformed) with the skill content appended when given.

    The cli_only prompt has no skill slot, so a preloaded skill is appended after
    the base framing rather than templated in — the most faithful analog to
    Trace2Skill's cli_skill_preloaded, and fully deterministic.
    """
    base = build_system_prompt(cli_only_raw)
    if skill_content and skill_content.strip():
        return f"{base.rstrip()}\n\n{_SKILL_PRELOAD_HEADER}\n\n{skill_content.strip()}\n"
    return base


# --------------------------------------------------------------------------- #
# Token accounting
# --------------------------------------------------------------------------- #

_USAGE_INT_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "llm_calls",
)


def _usage_from_metrics(metrics) -> dict:
    """Snapshot a conversation LLM's token usage + cost into a plain dict."""
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


def _sum_usages(usages: list[dict]) -> dict:
    """Aggregate per-instance usage dicts into a run total."""
    total = {k: sum(int(u.get(k, 0) or 0) for u in usages) for k in _USAGE_INT_KEYS}
    total["cost"] = sum(float(u.get("cost", 0.0) or 0.0) for u in usages)
    total["total_tokens"] = total["prompt_tokens"] + total["completion_tokens"]
    return total


def instance_outputs_complete(final_out_dir: Path, input_files: list[str]) -> bool:
    """True iff every expected output for this instance already exists.

    Used by --missing-only to resume a run without redoing finished instances.
    """
    if not input_files:
        return False
    return all(
        (final_out_dir / output_name_for_input(f)).exists() for f in input_files
    )


def _close_terminals(agent) -> None:
    """Close the agent's terminal tool(s) to release their tmux sessions/PTYs.

    The terminal tool opens a tmux pane pool per conversation; without an
    explicit close these leak (the tmux server is a daemon that survives the
    worker process), and a long run exhausts the system PTY pool
    (kern.tty.ptmx_max), after which every new pane fails with
    "fork failed: Device not configured". Calling executor.close() runs the
    pool teardown with kill_session=True, freeing this conversation's PTYs.
    """
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


def _raise_keyboard_interrupt_on_sigterm(signum, frame) -> None:
    """Make SIGTERM unwind like SIGINT does (Python already does this for SIGINT).

    Each instance runs in its own spawned worker process. When the harness/OS
    kills a worker for memory pressure it sends SIGTERM (exit code 143), whose
    default disposition terminates the
    process immediately with no ``finally`` blocks run — orphaning that
    instance's tmux pane pool (see ``_close_terminals``) instead of releasing
    it. Installing this handler makes SIGTERM raise KeyboardInterrupt the same
    way SIGINT already does by default, so the existing try/finally around
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


def _disable_interactive_pagers() -> None:
    """Force non-interactive pagers so agent bash can't hang on `less`/`help()`.

    An agent command like ``git log``, ``help(obj)``, or ``man`` otherwise opens a
    pager in the terminal that blocks forever waiting for a keypress. Setting these
    in the runner env propagates to the terminal tool's child shells.
    """
    os.environ.setdefault("PAGER", "cat")
    os.environ.setdefault("GIT_PAGER", "cat")
    os.environ.setdefault("MANPAGER", "cat")
    os.environ.setdefault("LESS", "-F -X -R")
    os.environ["GIT_TERMINAL_PROMPT"] = "0"


# --------------------------------------------------------------------------- #
# Per-instance rollout
# --------------------------------------------------------------------------- #


def _stage_working_dir(task_dir: Path, input_path: Path, skill_dir: Path | None) -> Path:
    """Create a fresh task dir with the input spreadsheet + skill support files."""
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True)
    task_dir_resolved = task_dir.resolve()
    work_input = task_dir / input_path.name
    shutil.copy(input_path, work_input)
    if skill_dir is not None and skill_dir.is_dir():
        for child in skill_dir.iterdir():
            dest = task_dir / child.name
            if child.is_dir():
                child_resolved = child.resolve()
                is_ancestor = child_resolved in task_dir_resolved.parents
                if child_resolved == task_dir_resolved or is_ancestor:
                    # skill_dir is an ancestor of task_dir (e.g. --skill-file points at a
                    # file inside this same run's output tree) — copying it in would nest
                    # the run's own (still-growing) output inside every task dir.
                    print(
                        f"  WARNING: skipping self-referential skill support dir {child}",
                        file=sys.stderr,
                    )
                    continue
                shutil.copytree(child, dest)
            else:
                shutil.copy(child, dest)
    return work_input


def _run_one_instance(
    *,
    instance: dict,
    data_path: Path,
    output_dir: Path,
    system_prompt: str,
    skill_dir: Path | None,
    cfg,
    max_iter: int,
    missing_only: bool = False,
) -> dict:
    """Run the agent on one instance (all its test cases) and dump the trace.

    Writes each test case's output spreadsheet into the evaluate_with_official
    layout and the first test case's event stream to logs/<instance_id>.json.
    Returns this instance's aggregated token usage.
    """
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt_on_sigterm)
    instance_id = str(instance["id"])
    spreadsheet_path = str(instance.get("spreadsheet_path", instance_id))
    ss_dir = find_spreadsheet_dir(str(data_path), instance)
    if ss_dir is None:
        print(f"  [{instance_id}] spreadsheet dir not found — skipping", file=sys.stderr)
        return _sum_usages([])
    input_files = find_input_files(list_dir_names(Path(ss_dir)))
    if not input_files:
        print(f"  [{instance_id}] no input files — skipping", file=sys.stderr)
        return _sum_usages([])
    if len(input_files) > 1:
        print(
            f"  [{instance_id}] {len(input_files)} test cases; logging first only",
            file=sys.stderr,
        )

    final_out_dir = output_dir / spreadsheet_path
    if missing_only and instance_outputs_complete(final_out_dir, input_files):
        print(f"  [{instance_id}] already complete — skipping (--missing-only)")
        return _sum_usages([])
    final_out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    tc_usages: list[dict] = []
    for tc_idx, input_file in enumerate(input_files):
        input_path = Path(ss_dir) / input_file
        output_name = output_name_for_input(input_file)
        input_base = Path(input_file).stem
        task_dir = (output_dir / ".work" / f"{instance_id}__{input_base}").resolve()
        work_input = _stage_working_dir(task_dir, input_path, skill_dir)
        work_output = task_dir / output_name

        prompt = build_task_prompt(
            working_dir=str(task_dir),
            input_file=str(work_input),
            output_file=str(work_output),
            instruction=instance["instruction"],
            spreadsheet_content=spreadsheet_content_preview(str(work_input)),
            instruction_type=instance.get("instruction_type", ""),
            answer_position=instance.get("answer_position", ""),
        )

        llm = LLM(
            model=cfg.model,
            base_url=cfg.base_url,
            api_key=SecretStr(cfg.api_key or "dummy"),
            usage_id=f"sb_{instance_id}",
            drop_params=True,
        )
        agent = Agent(llm=llm, tools=[Tool(name="terminal")], system_prompt=system_prompt)
        conversation = Conversation(
            agent=agent,
            workspace=LocalWorkspace(working_dir=str(task_dir)),
            max_iteration_per_run=max_iter,
            visualizer=None,
        )

        try:
            conversation.send_message(prompt)
            conversation.run()
            # Post-hoc reminder if the output file was not created (parity with
            # Trace2Skill's base agent), tool/model-agnostic insurance.
            if not work_output.exists():
                conversation.send_message(
                    f"[System Check] The output file was NOT created at: {work_output}\n"
                    "Please create the output file at the exact path above, then finish."
                )
                conversation.run()
            events = [e.model_dump(mode="json") for e in conversation.state.events]
        except Exception as exc:  # noqa: BLE001 - never let one instance kill the batch
            print(f"  [{instance_id}] ERROR: {exc}", file=sys.stderr)
            traceback.print_exc()
            events = []
        finally:
            # Always release this conversation's tmux panes/PTYs so a long run
            # doesn't exhaust the system PTY pool (see _close_terminals).
            _close_terminals(agent)

        usage = _usage_from_metrics(llm.metrics)
        tc_usages.append(usage)

        if work_output.exists():
            shutil.copy(work_output, final_out_dir / output_name)
            status = "ok"
        else:
            status = "no-output"

        if tc_idx == 0:
            (logs_dir / f"{instance_id}.json").write_text(
                json.dumps(events, ensure_ascii=False), encoding="utf-8"
            )
        print(
            f"  [{instance_id}] {input_file} -> {output_name} "
            f"({status}, {len(events)} events, {usage['total_tokens']:,} tok, "
            f"{usage['llm_calls']} calls)"
        )

    return _sum_usages(tc_usages)


def list_dir_names(path: Path) -> list[str]:
    return [p.name for p in path.iterdir()]


# --------------------------------------------------------------------------- #
# Batch run
# --------------------------------------------------------------------------- #


def run(
    *,
    split: str,
    data_path: Path,
    skill_file: Path | None,
    output_dir: Path,
    cfg,
    concurrency: int,
    limit: int | None,
    max_iter: int,
    skip_eval: bool,
    missing_only: bool = False,
    ids: set[str] | None = None,
) -> None:
    _disable_interactive_pagers()
    sweep_stale_pool_sessions()
    dataset = load_dataset(str(data_path))
    start, end = SPLIT_RANGES[split]
    instances = dataset[start:end]
    if ids is not None:
        instances = [x for x in instances if str(x["id"]) in ids]
        missing = ids - {str(x["id"]) for x in instances}
        if missing:
            print(f"WARNING: --ids not found in {split} split: {sorted(missing)}", file=sys.stderr)
    if limit is not None:
        instances = instances[:limit]
    n = len(instances)
    eval_end = start + n  # official evaluator must score exactly what we ran

    skill_content = skill_file.read_text(encoding="utf-8") if skill_file else None
    skill_dir = skill_file.parent if skill_file else None
    cli_only_raw = CLI_ONLY_PROMPT.read_text(encoding="utf-8")
    system_prompt = compose_system_prompt(cli_only_raw, skill_content)

    print(f"Split             : {split}  (dataset[{start}:{eval_end}], {n} instances)")
    print(f"Skill             : {skill_file or '(no skill / baseline)'}")
    print(f"Model             : {cfg.model}")
    print(f"Base URL          : {cfg.base_url}")
    print(f"Output dir        : {output_dir}")
    print(f"Concurrency       : {concurrency}   max_iter/instance: {max_iter}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process-based fan-out (NOT threads). The openhands-sdk LLM class guards
    # every completion with a process-shared ClassVar RLock (it toggles the
    # global litellm.modify_params flag), holding it across the whole network
    # call. Threads/asyncio therefore serialize to one in-flight LLM request
    # per process. Separate processes each get their own RLock + litellm global,
    # so the requests actually run concurrently against the endpoint.
    results: list[tuple[str, dict]] = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=concurrency, mp_context=ctx) as pool:
        future_to_id = {
            pool.submit(
                _run_one_instance,
                instance=instance,
                data_path=data_path,
                output_dir=output_dir,
                system_prompt=system_prompt,
                skill_dir=skill_dir,
                cfg=cfg,
                max_iter=max_iter,
                missing_only=missing_only,
            ): str(instance["id"])
            for instance in instances
        }
        for future in as_completed(future_to_id):
            iid = future_to_id[future]
            try:
                usage = future.result()
            except Exception as exc:  # noqa: BLE001 - one crashed worker != dead batch
                print(f"  [{iid}] WORKER CRASHED: {exc}", file=sys.stderr)
                traceback.print_exc()
                usage = _sum_usages([])
            results.append((iid, usage))

    # ------------------------------------------------------------- token usage
    per_instance = {iid: usage for iid, usage in results}
    total = _sum_usages([u for _, u in results])
    (output_dir / "token_usage.json").write_text(
        json.dumps(
            {"model": cfg.model, "split": split, "n_instances": n,
             "total": total, "per_instance": per_instance},
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nToken usage (this run):")
    print(f"  Instances         : {n}")
    print(f"  Prompt tokens     : {total['prompt_tokens']:,}")
    print(f"  Completion tokens : {total['completion_tokens']:,}")
    print(f"  Total tokens      : {total['total_tokens']:,}")
    if total["cache_read_tokens"] or total["cache_write_tokens"]:
        cache_read = total["cache_read_tokens"]
        cache_write = total["cache_write_tokens"]
        print(f"  Cache read/write  : {cache_read:,} / {cache_write:,}")
    if total["reasoning_tokens"]:
        print(f"  Reasoning tokens  : {total['reasoning_tokens']:,}")
    print(f"  LLM calls         : {total['llm_calls']:,}")
    if total["cost"]:
        print(f"  Cost              : ${total['cost']:.4f}")
    if n:
        print(f"  Avg tokens/inst   : {total['total_tokens'] // n:,}")
    print(f"  Wrote {output_dir / 'token_usage.json'}")

    if skip_eval:
        print("\nSkipping official evaluation (--skip-eval).")
        return
    if ids is not None:
        print("\nSkipping built-in official evaluation (--ids subset is non-contiguous; "
              "score the affected ids externally).")
        return

    print("\nRunning official SpreadsheetBench evaluation...")
    cmd = [
        "uv", "run", "python", str(EVALUATOR),
        "--data_path", str(data_path.resolve()),
        "--output_dir", str(output_dir.resolve()),
        "--start_idx", str(start),
        "--end_idx", str(eval_end),
    ]
    proc = subprocess.run(cmd, cwd=str(TRACE2SKILL))
    if proc.returncode != 0:
        print(f"WARNING: evaluator exited with code {proc.returncode}", file=sys.stderr)
    else:
        print(f"Wrote {output_dir / 'eval_official_results.json'}")


def main() -> None:
    _register_extra_models()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", choices=list(SPLIT_RANGES), required=True)
    parser.add_argument("--skill-file", type=Path, default=None, metavar="PATH",
                        help="SKILL.md to preload. Omit for a no-skill baseline.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, metavar="PATH")
    parser.add_argument("--output-dir", type=Path, required=True, metavar="DIR")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--llm-provider", default="eval_proxy", metavar="PROVIDER")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--secrets-file", default=DEFAULT_SECRETS_FILE, metavar="PATH")
    parser.add_argument("--concurrency", type=int, default=4, metavar="N")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Only run the first N instances of the split (smoke test).")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER, metavar="N")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip the official evaluation subprocess.")
    parser.add_argument("--missing-only", action="store_true",
                        help="Skip instances whose output already exists (resume a run).")
    parser.add_argument("--ids", default=None, metavar="ID,ID,...",
                        help="Only run this comma-separated set of instance ids (scoped ablation). "
                             "Built-in official eval is skipped; score the subset externally.")
    args = parser.parse_args()

    if args.skill_file is not None and not args.skill_file.exists():
        print(f"ERROR: skill file not found: {args.skill_file}", file=sys.stderr)
        sys.exit(1)
    if not EVALUATOR.exists():
        print(f"ERROR: evaluator not found: {EVALUATOR}", file=sys.stderr)
        sys.exit(1)

    cfg = _resolve_llm_config(
        model=args.model,
        provider=args.llm_provider,
        base_url=args.base_url,
        api_key=args.api_key,
        secrets_file=args.secrets_file,
    )

    run(
        split=args.split,
        data_path=args.data_path,
        skill_file=args.skill_file,
        output_dir=args.output_dir,
        cfg=cfg,
        concurrency=args.concurrency,
        limit=args.limit,
        max_iter=args.max_iter,
        skip_eval=args.skip_eval,
        missing_only=args.missing_only,
        ids={s.strip() for s in args.ids.split(",") if s.strip()} if args.ids else None,
    )


if __name__ == "__main__":
    main()
