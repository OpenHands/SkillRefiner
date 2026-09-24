"""GEPA baseline: reflectively tune the preloaded SKILL.md on SpreadsheetBench.

A third skill-producing baseline alongside the summariser→cluster→refine ablation
and the Trace2Skill skill_evolver. Instead of summarising traces, this uses
`gepa.optimize` to evolve the xlsx `SKILL.md` text against the train split:

  candidate = {"skill": <SKILL.md markdown>}   (seed = canonical xlsx SKILL.md)
  metric    = official SpreadsheetBench soft_score of the openhands agent rollout
              run with that candidate skill appended to the system prompt
  reflection= an LLM (default gpt-5.4-mini via eval-proxy) reads each rollout's
              trace + official pass/fail messages and rewrites the skill

Output: ``<output-dir>/proposed_skill.md`` (= result.best_candidate["skill"]),
directly consumable by ``eval_skill_on_spreadsheetbench.py --skill-file ...`` for
a held-out comparison, plus ``<output-dir>/gepa_result.json``.

The task/agent model reaches MiniMax through the same ``--llm-provider`` plumbing
as the rest of the harness; the reflection LM is wired as a litellm callable
against the proxy base_url + key.

Smoke test (5 instances, train==val overlap, tiny budget):

  uv run python baselines/gepa/gepa_tune_skill.py \\
      --output-dir /tmp/sb_gepa_smoke --limit 5 --max-metric-calls 10 \\
      --reflection-minibatch-size 2 --max-iter 30 \\
      --model openai/minimax-m2.7 --llm-provider eval_proxy \\
      --base-url https://your-llm-proxy.example.com \\
      --secrets-file .eval_proxy.secrets.json

Full paper-parity run: drop ``--limit`` (train split = dataset[0:200]), set
``--val-size`` for a held-out slice of the train split, and raise
``--max-metric-calls``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import signal
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE2SKILL = REPO_ROOT / "baselines" / "trace2skill"
_SCRIPTS_DIR = REPO_ROOT / "scripts"  # holds _llm_config.py
_SPREADSHEETBENCH_DIR = REPO_ROOT / "scripts" / "benchmarks" / "spreadsheetbench"

for _p in (str(_SPREADSHEETBENCH_DIR), str(_SCRIPTS_DIR), str(TRACE2SKILL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Offline guard: skill_refiner/__init__.py sets LITELLM_LOCAL_MODEL_COST_MAP before
# litellm's import-time HTTPS GET to raw.githubusercontent.com. It must run ahead of the
# openhands/litellm imports below, and a plain `import skill_refiner` cannot -- ruff's
# isort sorts `openhands` first, which is precisely how the guard got skipped here.
__import__("skill_refiner")  # noqa: F401 - imported for its import-time side effect

import gepa  # noqa: E402
from gepa.core.adapter import EvaluationBatch, GEPAAdapter  # noqa: E402

# Reuse the runner's rollout + the eval script's skill staging verbatim so the
# GEPA baseline exercises the exact same agent/scoring path as everything else.
from eval_skill_on_spreadsheetbench import (  # noqa: E402
    CANONICAL_XLSX_SKILL_DIR,
    stage_skill_dir,
)
from _llm_config import (  # noqa: E402
    _register_extra_models,
    _resolve_llm_config,
)
from spreadsheetbench_agent_runner import (  # noqa: E402
    CLI_ONLY_PROMPT,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_SECRETS_FILE,
    SPLIT_RANGES,
    _disable_interactive_pagers,
    _run_one_instance,
    _sum_usages,
    sweep_stale_pool_sessions,
)
from spreadsheetbench_common import spreadsheet_content_preview, strip_think  # noqa: E402
from spreadsheetbench_support import (  # noqa: E402
    find_output_dir,
    find_spreadsheet_dir,
    load_dataset,
)

# The official-first (local-fallback) comparison wrapper the evaluator uses.
from evaluate_with_official import compare_workbooks  # noqa: E402

SKILL_COMPONENT = "skill"
_MAX_TRACE_CHARS = 4000


# --------------------------------------------------------------------------- #
# Per-instance official scoring (mirrors evaluate_with_official's inner loop)
# --------------------------------------------------------------------------- #


def _gt_files(all_files: list[str]) -> list[str]:
    """Ground-truth files in an instance dir, in the evaluator's precedence."""
    gt = sorted(f for f in all_files if f.endswith("_answer.xlsx"))
    if gt:
        return gt
    gt = sorted(f for f in all_files if f.endswith("_golden.xlsx"))
    if gt:
        return gt
    return ["golden.xlsx"] if "golden.xlsx" in all_files else []


def _output_name_for_gt(gt_file: str) -> str:
    if gt_file.endswith("_answer.xlsx"):
        return gt_file.replace("_answer.xlsx", "_output.xlsx")
    if gt_file == "golden.xlsx":
        return "initial_output.xlsx"
    return gt_file.replace("_golden.xlsx", "_output.xlsx")


def _score_instance(
    instance: dict, data_path: Path, output_base: Path
) -> tuple[float, list[dict]]:
    """Official soft_score for one instance + per-test-case pass/fail messages.

    Reads the output spreadsheets the rollout wrote under ``output_base`` and
    compares them to ground truth with the official ``compare_workbooks``.
    Returns (soft_score in [0,1], [{output_file, passed, message}, ...]).
    """
    ss_dir = find_spreadsheet_dir(str(data_path), instance)
    if ss_dir is None:
        return 0.0, [{"output_file": None, "passed": False, "message": "spreadsheet dir not found"}]
    try:
        all_files = os.listdir(ss_dir)
    except OSError as exc:
        return 0.0, [{"output_file": None, "passed": False, "message": f"listdir: {exc}"}]

    gt_files = _gt_files(all_files)
    if not gt_files:
        return 0.0, [{"output_file": None, "passed": False, "message": "no ground truth files"}]

    out_dir = find_output_dir(str(output_base), instance)
    instruction_type = instance.get("instruction_type", "")
    answer_position = instance.get("answer_position", "")

    tcs: list[dict] = []
    passed = 0
    for gt_file in gt_files:
        output_file = _output_name_for_gt(gt_file)
        gt_path = os.path.join(ss_dir, gt_file)
        out_path = os.path.join(out_dir or str(output_base), output_file)
        try:
            ok, msg = compare_workbooks(gt_path, out_path, instruction_type, answer_position)
        except Exception as exc:  # noqa: BLE001 - a bad workbook must not kill scoring
            ok, msg = False, str(exc)
        passed += 1 if ok else 0
        tcs.append({"output_file": output_file, "passed": bool(ok), "message": msg})

    soft = passed / len(tcs) if tcs else 0.0
    return soft, tcs


# --------------------------------------------------------------------------- #
# Trajectory extraction for reflection
# --------------------------------------------------------------------------- #


def _summarize_events(events: list[dict], max_chars: int = _MAX_TRACE_CHARS) -> str:
    """Best-effort readable summary of an openhands event dump for reflection.

    Pulls whatever human-readable text each event carries (messages, executed
    commands, tool observations, errors), newest-last, truncated to the tail so
    the most recent actions/errors survive the char cap.
    """
    lines: list[str] = []
    for ev in events:
        kind = ev.get("kind") or ev.get("type") or ev.get("source") or "event"
        bits: list[str] = []
        for key in ("content", "message", "command", "code", "thought", "observation", "error", "text"):
            val = ev.get(key)
            if isinstance(val, str) and val.strip():
                bits.append(f"{key}={val.strip()}")
        # openhands nests message text under llm_message/tool_call payloads
        for nested_key in ("llm_message", "action", "tool_call", "args"):
            nested = ev.get(nested_key)
            if isinstance(nested, dict):
                for k, v in nested.items():
                    if isinstance(v, str) and v.strip():
                        bits.append(f"{nested_key}.{k}={v.strip()}")
        if bits:
            lines.append(f"[{kind}] " + " | ".join(bits))
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = "...(truncated)...\n" + text[-max_chars:]
    return text or "(no readable events captured)"


# --------------------------------------------------------------------------- #
# Isolated rollout worker (module-level so it is picklable for spawn)
# --------------------------------------------------------------------------- #


def _rollout_worker(kwargs: dict, usage_path: str) -> None:
    """Run one agent rollout in a fresh process; write its usage to ``usage_path``.

    Writing the result to disk (instead of returning it) lets the parent enforce a
    wall-clock timeout by terminating this process and still tell "timed out" apart
    from "finished". A timeout SIGINT surfaces as KeyboardInterrupt here; the inner
    ``_run_one_instance`` still runs its ``finally`` (closing tmux/PTYs) on the way
    out, and we record the interruption as an error.
    """
    try:
        usage = _run_one_instance(**kwargs) or {}
    except BaseException as exc:  # noqa: BLE001 - incl. KeyboardInterrupt from timeout
        usage = {"_error": f"{type(exc).__name__}: {exc}"}
    try:
        Path(usage_path).write_text(json.dumps(usage), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# GEPA adapter
# --------------------------------------------------------------------------- #


class SpreadsheetBenchSkillAdapter(GEPAAdapter):
    """Runs the openhands SpreadsheetBench agent with a candidate SKILL.md and
    scores it with the official evaluator; feeds traces + pass/fail to reflection.
    """

    def __init__(
        self,
        *,
        data_path: Path,
        cfg,
        run_root: Path,
        max_iter: int,
        isolate_rollouts: bool = True,
        concurrency: int = 1,
        rollout_timeout: float = 420.0,
        canonical_skill_dir: Path = CANONICAL_XLSX_SKILL_DIR,
    ) -> None:
        self.data_path = data_path
        self.cfg = cfg
        self.run_root = run_root
        self.max_iter = max_iter
        self.isolate_rollouts = isolate_rollouts
        self.concurrency = max(1, concurrency)
        self.rollout_timeout = rollout_timeout if rollout_timeout and rollout_timeout > 0 else None
        self.canonical_skill_dir = canonical_skill_dir
        self.cli_only_raw = CLI_ONLY_PROMPT.read_text(encoding="utf-8")
        self._eval_counter = 0
        self.rollout_usages: list[dict] = []  # one per agent rollout (for token accounting)
        self.run_root.mkdir(parents=True, exist_ok=True)

    # -- rollout dispatch -------------------------------------------------- #

    @staticmethod
    def _reap(proc) -> None:
        """Release the parent-side resources of a finished child (FDs/semaphores).

        Without ``Process.close()`` these accumulate across hundreds of rollouts and
        add to memory/FD pressure on the parent — a contributor to being SIGKILL'd
        on a busy machine.
        """
        from contextlib import suppress

        with suppress(Exception):
            proc.close()

    @staticmethod
    def _terminate(proc) -> None:
        """Stop a stalled rollout: SIGINT (lets its finally close tmux) then hard kill."""
        from contextlib import suppress

        for sig, wait in ((signal.SIGINT, 10), (signal.SIGTERM, 5), (signal.SIGKILL, 2)):
            if not proc.is_alive():
                return
            with suppress(Exception):
                os.kill(proc.pid, sig)
            proc.join(wait)

    def _rollout_batch(self, batch, output_dir, system_prompt, skill_dir) -> list[dict]:
        """Run every rollout in ``batch``, returning per-instance usage (by index).

        GEPA hands the adapter the whole minibatch/valset at once and does no
        parallelism itself ("parallelism, if any, is the adapter's"), so this fans
        the rollouts out over up to ``concurrency`` fresh spawned subprocesses and
        enforces a per-rollout ``rollout_timeout`` — a stalled agent (hung tool/LLM
        call) is terminated and scored as an error instead of wedging the run. Each
        rollout is its own short-lived process, so tmux/PTY/memory never accumulate.
        A failed/timed-out rollout yields ``{"_error": ...}``.
        """
        n = len(batch)
        usages: list[dict] = [{} for _ in range(n)]

        if not self.isolate_rollouts:  # dev-only: in-process, sequential, no timeout
            for i, instance in enumerate(batch):
                try:
                    usages[i] = _run_one_instance(
                        instance=instance, data_path=self.data_path, output_dir=output_dir,
                        system_prompt=system_prompt, skill_dir=skill_dir, cfg=self.cfg,
                        max_iter=self.max_iter, missing_only=False,
                    ) or {}
                except Exception as exc:  # noqa: BLE001
                    usages[i] = {"_error": str(exc)}
            return usages

        ctx = mp.get_context("spawn")
        usage_dir = Path(output_dir) / "_usage"
        usage_dir.mkdir(parents=True, exist_ok=True)
        pending = list(enumerate(batch))
        active: dict = {}  # proc -> (i, usage_path, deadline)

        def _launch() -> None:
            while pending and len(active) < self.concurrency:
                i, instance = pending.pop(0)
                up = usage_dir / f"{i}.json"
                if up.exists():
                    up.unlink()
                kwargs = dict(
                    instance=instance, data_path=self.data_path, output_dir=output_dir,
                    system_prompt=system_prompt, skill_dir=skill_dir, cfg=self.cfg,
                    max_iter=self.max_iter, missing_only=False,
                )
                proc = ctx.Process(target=_rollout_worker, args=(kwargs, str(up)))
                proc.start()
                deadline = time.monotonic() + self.rollout_timeout if self.rollout_timeout else None
                active[proc] = (i, up, deadline)

        _launch()
        while active:
            for proc in list(active):
                i, up, deadline = active[proc]
                proc.join(timeout=2)
                if not proc.is_alive():
                    try:
                        usages[i] = json.loads(up.read_text(encoding="utf-8")) if up.exists() else {"_error": "no result written"}
                    except Exception as exc:  # noqa: BLE001
                        usages[i] = {"_error": f"unreadable usage: {exc}"}
                    del active[proc]
                    self._reap(proc)
                elif deadline is not None and time.monotonic() > deadline:
                    self._terminate(proc)
                    usages[i] = {"_error": f"rollout timeout (> {self.rollout_timeout:.0f}s)"}
                    del active[proc]
                    self._reap(proc)
            _launch()
        return usages

    # -- GEPAAdapter API --------------------------------------------------- #

    def evaluate(
        self,
        batch: list[dict],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        skill_text = candidate[SKILL_COMPONENT]
        idx = self._eval_counter
        self._eval_counter += 1

        # Stage a skill dir (canonical xlsx dir + candidate SKILL.md) once per
        # candidate; the rollout copies its files (recalc.py, ...) into each task.
        staged_md = stage_skill_dir(
            skill="xlsx",
            canonical_skill_dir=self.canonical_skill_dir,
            candidate_md=skill_text,
            dest_root=self.run_root / f"eval{idx}" / ".staged-skill",
        )
        skill_dir = staged_md.parent
        # Composed prompt = base cli_only + candidate skill appended (parity with
        # the runner's compose_system_prompt, done here via the runner import).
        from spreadsheetbench_agent_runner import compose_system_prompt

        system_prompt = compose_system_prompt(self.cli_only_raw, skill_text)
        rollout_out = self.run_root / f"eval{idx}" / "rollouts"

        outputs: list[dict] = []
        scores: list[float] = []
        trajectories: list[dict] = []

        # Run every rollout (optionally concurrent), then score/read traces.
        usages = self._rollout_batch(batch, rollout_out, system_prompt, skill_dir)

        for instance, usage in zip(batch, usages):
            iid = str(instance["id"])
            if usage.get("_error"):
                soft, tcs = 0.0, [{"output_file": None, "passed": False, "message": f"rollout error: {usage['_error']}"}]
                events: list[dict] = []
            else:
                self.rollout_usages.append(usage or {})
                soft, tcs = _score_instance(instance, self.data_path, rollout_out)
                log_path = rollout_out / "logs" / f"{iid}.json"
                try:
                    events = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
                except Exception:  # noqa: BLE001
                    events = []

            produced = any(tc.get("passed") is not None and tc.get("output_file") for tc in tcs)
            outputs.append({"id": iid, "soft_score": soft, "test_cases": tcs})
            scores.append(float(soft))
            if capture_traces:
                trajectories.append(
                    {
                        "id": iid,
                        "instruction": instance.get("instruction", ""),
                        "instruction_type": instance.get("instruction_type", ""),
                        "answer_position": instance.get("answer_position", ""),
                        "soft_score": soft,
                        "test_cases": tcs,
                        "output_produced": produced,
                        "trace": _summarize_events(events) if events else "(no trace captured)",
                    }
                )

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict]]:
        records: list[dict] = []
        for traj in eval_batch.trajectories or []:
            fail_msgs = [
                f"{tc.get('output_file')}: {tc.get('message')}"
                for tc in traj.get("test_cases", [])
                if not tc.get("passed") and tc.get("message")
            ]
            verdict = "CORRECT" if traj["soft_score"] >= 1.0 else f"score={traj['soft_score']:.2f}"
            feedback_parts = [
                f"Official SpreadsheetBench result: {verdict}.",
                "The agent DID create an output spreadsheet."
                if traj.get("output_produced")
                else "The agent did NOT produce a valid output spreadsheet.",
            ]
            if fail_msgs:
                feedback_parts.append("Failing cell/range comparisons:\n" + "\n".join(fail_msgs))
            if traj["soft_score"] >= 1.0:
                feedback_parts.append(
                    "This instance passed — reinforce whatever guidance made it succeed."
                )
            else:
                feedback_parts.append(
                    "Improve the skill so an agent following it would fix these failures "
                    "(e.g. clearer steps, common openpyxl/formula pitfalls, verification/recalc)."
                )
            records.append(
                {
                    "Inputs": {
                        "instruction": traj["instruction"],
                        "instruction_type": traj["instruction_type"],
                        "answer_position": traj["answer_position"],
                    },
                    "Generated Outputs": traj["trace"],
                    "Feedback": "\n".join(feedback_parts),
                }
            )
        # Reflect on the skill component regardless of which subset GEPA asked for
        # (there is only one component here).
        return {SKILL_COMPONENT: records}


# --------------------------------------------------------------------------- #
# Reflection LM (litellm against the eval-proxy)
# --------------------------------------------------------------------------- #


def make_reflection_lm(*, model: str, base_url: str, api_key: str, usage_sink: list[dict]):
    """A `reflection_lm` callable: (prompt str|messages) -> str, via litellm.

    Appends a per-call usage dict (prompt/completion/total tokens + cost) to
    ``usage_sink`` so the driver can report the reflection model's token/cost
    footprint (gpt-5.4-mini via eval-proxy is priced, unlike MiniMax).

    GEPA stores this callable's return value verbatim as the new candidate, so a
    reasoning model's raw completion (chain-of-thought preamble, stray code fence)
    would otherwise leak straight into ``proposed_skill.md`` and every subsequent
    rollout's system prompt. ``strip_think`` keeps only the intended skill markdown.
    """
    import litellm

    def _call(prompt) -> str:
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        resp = litellm.completion(
            model=model,
            messages=messages,
            base_url=base_url or None,
            api_key=api_key or None,
            drop_params=True,
        )
        u = getattr(resp, "usage", None)
        cost = 0.0
        try:
            cost = float(resp._hidden_params.get("response_cost") or 0.0)
        except Exception:  # noqa: BLE001
            cost = 0.0
        usage_sink.append(
            {
                "prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(u, "total_tokens", 0) or 0),
                "cost": cost,
            }
        )
        return strip_think(resp.choices[0].message.content or "")

    return _call


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main() -> None:
    _register_extra_models()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, required=True, metavar="DIR")
    parser.add_argument("--split", choices=list(SPLIT_RANGES), default="train",
                        help="Split to tune on (default: train = dataset[0:200]).")
    parser.add_argument("--data-path", type=Path,
                        default=TRACE2SKILL / "data" / "spreadsheetbench_verified" / "spreadsheetbench_verified_400")
    parser.add_argument("--seed-skill", type=Path, default=CANONICAL_XLSX_SKILL_DIR / "SKILL.md",
                        metavar="PATH", help="Seed SKILL.md GEPA evolves (default: canonical xlsx).")
    # Task/agent model plumbing (same flags as the runner).
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--llm-provider", default="eval_proxy")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--secrets-file", default=DEFAULT_SECRETS_FILE, metavar="PATH")
    parser.add_argument("--max-iter", type=int, default=100, metavar="N")
    # Reflection LM.
    parser.add_argument("--reflection-model", default="openai/gpt-5.4-mini",
                        help="litellm model id for reflection (default: openai/gpt-5.4-mini).")
    # Data slicing / GEPA budget.
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Use only the first N instances of the split (smoke test).")
    parser.add_argument("--val-size", type=int, default=None, metavar="N",
                        help="Hold out the LAST N sliced instances as the valset; the rest are "
                        "the trainset. Omit to use ALL sliced instances as both (overlap).")
    parser.add_argument("--max-metric-calls", type=int, default=60, metavar="N")
    parser.add_argument("--reflection-minibatch-size", type=int, default=3, metavar="N")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-isolate-rollouts", dest="isolate_rollouts", action="store_false",
                        help="Run each rollout in-process instead of a fresh subprocess. "
                        "Default (isolated) prevents tmux/PTY/memory accumulation over a long run.")
    parser.set_defaults(isolate_rollouts=True)
    parser.add_argument("--concurrency", type=int, default=4, metavar="N",
                        help="Rollouts to run in parallel within each valset/minibatch eval "
                        "(fresh subprocess each). GEPA does no parallelism itself. Default 4.")
    parser.add_argument("--rollout-timeout", type=float, default=420.0, metavar="SEC",
                        help="Kill and score-as-error any single rollout exceeding this wall-clock "
                        "budget (default 420s). Prevents one stalled agent from hanging the run. "
                        "Set 0 to disable.")
    parser.add_argument("--no-eval-cache", dest="cache_evaluation", action="store_false",
                        help="Disable GEPA's (candidate, example) eval cache. Default caches, so "
                        "an instance's score for a given skill is computed once, not re-run every "
                        "time it lands in a minibatch (the main source of redundant rollouts).")
    parser.set_defaults(cache_evaluation=True)
    parser.add_argument("--fresh", action="store_true",
                        help="Wipe any existing GEPA state under <output-dir>/gepa_state before "
                        "starting. Default resumes from it (GEPA reads run_dir and continues).")
    args = parser.parse_args()

    if not args.seed_skill.exists():
        print(f"ERROR: seed skill not found: {args.seed_skill}", file=sys.stderr)
        sys.exit(1)

    _disable_interactive_pagers()
    sweep_stale_pool_sessions()
    cfg = _resolve_llm_config(
        model=args.model,
        provider=args.llm_provider,
        base_url=args.base_url,
        api_key=args.api_key,
        secrets_file=args.secrets_file,
    )

    dataset = load_dataset(str(args.data_path))
    start, end = SPLIT_RANGES[args.split]
    instances = dataset[start:end]
    if args.limit is not None:
        instances = instances[: args.limit]
    if not instances:
        print("ERROR: no instances after slicing.", file=sys.stderr)
        sys.exit(1)

    if args.val_size and args.val_size < len(instances):
        trainset = instances[: len(instances) - args.val_size]
        valset = instances[len(instances) - args.val_size :]
        overlap = False
    else:
        trainset = list(instances)
        valset = list(instances)  # overlap (smoke-test default)
        overlap = True

    seed_skill_text = args.seed_skill.read_text(encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gepa_state_dir = args.output_dir / "gepa_state"
    if args.fresh and gepa_state_dir.exists():
        shutil.rmtree(gepa_state_dir)
        print(f"Wiped existing GEPA state: {gepa_state_dir}")
    resuming = gepa_state_dir.exists()

    adapter = SpreadsheetBenchSkillAdapter(
        data_path=args.data_path,
        cfg=cfg,
        run_root=args.output_dir / "gepa_rollouts",
        max_iter=args.max_iter,
        isolate_rollouts=args.isolate_rollouts,
        concurrency=args.concurrency,
        rollout_timeout=args.rollout_timeout,
    )
    reflection_usages: list[dict] = []
    reflection_lm = make_reflection_lm(
        model=args.reflection_model, base_url=cfg.base_url, api_key=cfg.api_key,
        usage_sink=reflection_usages,
    )

    print(f"GEPA tune SKILL.md on SpreadsheetBench")
    print(f"  Split / slice     : {args.split}  train={len(trainset)}  val={len(valset)}"
          f"{'  (overlap)' if overlap else ''}")
    print(f"  Task model        : {cfg.model}  @ {cfg.base_url}")
    print(f"  Reflection model  : {args.reflection_model}")
    print(f"  Budget            : max_metric_calls={args.max_metric_calls}  "
          f"reflection_minibatch={args.reflection_minibatch_size}  max_iter={args.max_iter}")
    print(f"  Rollout isolation : {'subprocess-per-rollout' if args.isolate_rollouts else 'in-process'}"
          f"  concurrency={args.concurrency}  timeout={args.rollout_timeout:.0f}s")
    print(f"  Eval cache        : {'on' if args.cache_evaluation else 'off'}")
    print(f"  State / resume    : {gepa_state_dir}  ({'RESUMING' if resuming else 'fresh'})")
    print(f"  Output dir        : {args.output_dir}")

    result = gepa.optimize(
        seed_candidate={SKILL_COMPONENT: seed_skill_text},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.reflection_minibatch_size,
        cache_evaluation=args.cache_evaluation,
        display_progress_bar=True,
        seed=args.seed,
        run_dir=str(gepa_state_dir),
    )

    best_skill = result.best_candidate[SKILL_COMPONENT]
    proposed = args.output_dir / "proposed_skill.md"
    proposed.write_text(best_skill, encoding="utf-8")

    # Per-candidate aggregate val scores (mean over the valset); best_idx picks
    # the argmax. `val_aggregate_scores` is the correct GEPAResult attribute.
    val_scores = getattr(result, "val_aggregate_scores", None)
    best_idx = getattr(result, "best_idx", None)
    best_score = seed_score = None
    if isinstance(val_scores, (list, tuple)) and val_scores:
        seed_score = val_scores[0]
        if isinstance(best_idx, int) and 0 <= best_idx < len(val_scores):
            best_score = val_scores[best_idx]

    # ---------------------------------------------------------- token / cost
    rollout_total = _sum_usages(adapter.rollout_usages)
    refl_total = {
        "prompt_tokens": sum(int(u.get("prompt_tokens", 0)) for u in reflection_usages),
        "completion_tokens": sum(int(u.get("completion_tokens", 0)) for u in reflection_usages),
        "total_tokens": sum(int(u.get("total_tokens", 0)) for u in reflection_usages),
        "cost": sum(float(u.get("cost", 0.0)) for u in reflection_usages),
        "calls": len(reflection_usages),
    }
    token_usage = {
        "task_model": cfg.model,
        "reflection_model": args.reflection_model,
        "n_agent_rollouts": len(adapter.rollout_usages),
        "agent_rollouts_total": rollout_total,  # cost=$0 when task model is unpriced (e.g. MiniMax)
        "reflection_total": refl_total,
        "combined_cost": float(rollout_total.get("cost", 0.0)) + refl_total["cost"],
        "combined_total_tokens": int(rollout_total.get("total_tokens", 0)) + refl_total["total_tokens"],
    }
    (args.output_dir / "token_usage.json").write_text(json.dumps(token_usage, indent=2), encoding="utf-8")

    summary = {
        "task_model": cfg.model,
        "reflection_model": args.reflection_model,
        "split": args.split,
        "train_size": len(trainset),
        "val_size": len(valset),
        "max_metric_calls": args.max_metric_calls,
        "total_metric_calls": getattr(result, "total_metric_calls", None),
        "num_candidates": getattr(result, "num_candidates", None),
        "best_idx": best_idx,
        "seed_val_score": seed_score,
        "best_val_score": best_score,
        "val_aggregate_scores": list(val_scores) if val_scores is not None else None,
        "seed_skill_len": len(seed_skill_text),
        "best_skill_len": len(best_skill),
        "changed_from_seed": best_skill.strip() != seed_skill_text.strip(),
        "token_usage": token_usage,
    }
    (args.output_dir / "gepa_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== GEPA done ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {proposed}")
    print(f"Wrote {args.output_dir / 'gepa_result.json'}")
    print(f"Wrote {args.output_dir / 'token_usage.json'}")
    print(
        "\nEvaluate held-out with:\n"
        f"  uv run python {_SPREADSHEETBENCH_DIR / 'eval_skill_on_spreadsheetbench.py'} \\\n"
        f"      --skill-file {proposed} --output-dir <eval-out> [model/proxy args]"
    )


if __name__ == "__main__":
    main()
