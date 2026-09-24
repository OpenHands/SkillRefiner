"""GEPA baseline: reflectively tune the DAPO-Math SKILL.md against the train split.

The GEPA counterpart to SkillRefiner and the Trace2Skill evolver, ported from
baselines/gepa/gepa_tune_skill.py to DAPO-Math:

  candidate  = {"skill": <SKILL.md markdown>}   (seed = --seed-skill)
  metric     = 1.0 if the openhands DAPO agent (run with that candidate skill)
               produces a MATH_v2-correct answer, else 0.0
  reflection = an LLM reads each rollout's trace + correct/incorrect + the
               ground-truth answer and rewrites the skill

Reuses run_dapo_agent._run_one_task for the exact same agent/grader path as
every other DAPO run, and mirrors the SpreadsheetBench adapter's subprocess-
per-rollout isolation + per-rollout wall-clock timeout (MiniMax on the eval proxy
intermittently hangs a single call for hours; the timeout scores it as an error
instead of wedging the whole GEPA run).

Output: ``<output-dir>/proposed_skill.md`` (= result.best_candidate["skill"]),
directly consumable by run_dapo_agent.py --skill-file, plus gepa_result.json.

Smoke test (2 instances, tiny budget):

  uv run python baselines/gepa/gepa_tune_dapo_skill.py \\
      --output-dir /tmp/dapo_gepa_smoke --limit 2 --max-metric-calls 6 \\
      --reflection-minibatch-size 2 --val-size 0 \\
      --model gpt-5.4-mini --llm-provider eval_proxy \\
      --secrets-file .eval_proxy.secrets.json
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
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = REPO_ROOT / "scripts"  # holds _llm_config.py
_DAPO_DIR = REPO_ROOT / "scripts" / "benchmarks" / "dapo"
for _p in (str(_DAPO_DIR), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Offline guard: skill_refiner/__init__.py sets LITELLM_LOCAL_MODEL_COST_MAP before
# litellm's import-time HTTPS GET to raw.githubusercontent.com. It must run ahead of the
# openhands/litellm imports below, and a plain `import skill_refiner` cannot -- ruff's
# isort sorts `openhands` first, which is precisely how the guard got skipped here.
__import__("skill_refiner")  # noqa: F401 - imported for its import-time side effect

import gepa  # noqa: E402
from gepa.core.adapter import EvaluationBatch, GEPAAdapter  # noqa: E402
from _llm_config import _register_extra_models, _resolve_llm_config  # noqa: E402

from run_dapo_agent import (  # noqa: E402
    DapoTask,
    _run_one_task,
    _sum_usages,
    compose_system_prompt,
    load_tasks,
)

SKILL_COMPONENT = "skill"
_MAX_TRACE_CHARS = 4000

DEFAULT_DATA_DIR = Path("results/dapo_math_17k")
# The paper's DAPO-Math initial skill is LLM-generated (see datasets/README.md).
DEFAULT_SEED_SKILL = Path("datasets/dapo/seed_skills/parametric_seed.md")


# --------------------------------------------------------------------------- #
# Isolated rollout worker (module-level so it is picklable for spawn)
# --------------------------------------------------------------------------- #


def _rollout_worker(kwargs: dict, done_path: str) -> None:
    """Run one DAPO agent rollout in a fresh process; touch ``done_path`` when done.

    ``_run_one_task`` already persists its record + events to disk under
    ``output_dir``; the parent reads that record back by trace_id. Writing a tiny
    done-marker lets the parent distinguish "finished" from "killed by timeout"
    without depending on the (larger) record write having flushed.
    """
    from contextlib import suppress

    marker = "ok"
    try:
        _run_one_task(**kwargs)
    except BaseException as exc:  # noqa: BLE001 - incl. KeyboardInterrupt from timeout
        marker = f"error: {type(exc).__name__}: {exc}"
    with suppress(Exception):
        Path(done_path).write_text(marker, encoding="utf-8")


def _trace_from_events(events: list[dict], max_chars: int = _MAX_TRACE_CHARS) -> str:
    """Readable tail of an openhands event dump for reflection (newest-last)."""
    lines: list[str] = []
    for ev in events:
        kind = ev.get("kind") or ev.get("type") or ev.get("source") or "event"
        bits: list[str] = []
        text_keys = (
            "content", "message", "command", "code",
            "thought", "observation", "error", "text",
        )
        for key in text_keys:
            val = ev.get(key)
            if isinstance(val, str) and val.strip():
                bits.append(f"{key}={val.strip()}")
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
# GEPA adapter
# --------------------------------------------------------------------------- #


class DapoSkillAdapter(GEPAAdapter):
    """Runs the openhands DAPO agent with a candidate SKILL.md, scores each task
    0/1 with the MATH_v2 grader, and feeds traces + correctness to reflection.
    """

    def __init__(
        self,
        *,
        cfg,
        run_root: Path,
        max_iter: int,
        concurrency: int = 4,
        rollout_timeout: float = 420.0,
    ) -> None:
        self.cfg = cfg
        self.run_root = run_root
        self.max_iter = max_iter
        self.concurrency = max(1, concurrency)
        self.rollout_timeout = rollout_timeout if rollout_timeout and rollout_timeout > 0 else None
        self._eval_counter = 0
        self.rollout_usages: list[dict] = []
        self.run_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _reap(proc) -> None:
        from contextlib import suppress

        with suppress(Exception):
            proc.close()

    @staticmethod
    def _terminate(proc) -> None:
        from contextlib import suppress

        for sig, wait in ((signal.SIGINT, 8), (signal.SIGTERM, 4), (signal.SIGKILL, 2)):
            if not proc.is_alive():
                return
            with suppress(Exception):
                os.kill(proc.pid, sig)
            proc.join(wait)

    def _rollout_batch(
        self, batch: list[DapoTask], output_dir: Path, system_prompt: str
    ) -> list[dict]:
        """Run every task in ``batch`` (up to ``concurrency`` at once); return records.

        Each rollout is a fresh spawned process with a per-rollout wall-clock
        timeout. A finished rollout's record is read back from
        ``output_dir/records/<trace_id>.json``; a timed-out/failed one yields a
        synthetic error record (scored 0.0).
        """
        ctx = mp.get_context("spawn")
        done_dir = output_dir / "_done"
        done_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict] = [None] * len(batch)  # type: ignore[list-item]
        pending = list(enumerate(batch))
        active: dict = {}  # proc -> (i, task, done_path, deadline)

        def _record_path(task: DapoTask) -> Path:
            return output_dir / "records" / f"{task.trace_id}.json"

        def _launch() -> None:
            while pending and len(active) < self.concurrency:
                i, task = pending.pop(0)
                dp = done_dir / f"{task.trace_id}.done"
                if dp.exists():
                    dp.unlink()
                kwargs = dict(
                    task=task, output_dir=output_dir, cfg=self.cfg, max_iter=self.max_iter,
                    skip_existing=False, use_terminal=False, system_prompt=system_prompt,
                )
                proc = ctx.Process(target=_rollout_worker, args=(kwargs, str(dp)))
                proc.start()
                deadline = time.monotonic() + self.rollout_timeout if self.rollout_timeout else None
                active[proc] = (i, task, dp, deadline)

        def _finish(i: int, task: DapoTask, err: str | None) -> None:
            rp = _record_path(task)
            if err is None and rp.exists():
                try:
                    records[i] = json.loads(rp.read_text(encoding="utf-8"))
                    return
                except Exception as exc:  # noqa: BLE001
                    err = f"unreadable record: {exc}"
            records[i] = {
                "trace_id": task.trace_id, "prompt": task.prompt,
                "ground_truth": task.ground_truth, "final_response": "",
                "prediction": "", "correct": False,
                "error": err or "no record written", "events_path": "",
            }

        _launch()
        while active:
            for proc in list(active):
                i, task, dp, deadline = active[proc]
                proc.join(timeout=2)
                if not proc.is_alive():
                    err = None
                    if dp.exists():
                        marker = dp.read_text(encoding="utf-8").strip()
                        err = None if marker == "ok" else marker
                    _finish(i, task, err)
                    del active[proc]
                    self._reap(proc)
                elif deadline is not None and time.monotonic() > deadline:
                    self._terminate(proc)
                    _finish(i, task, f"rollout timeout (> {self.rollout_timeout:.0f}s)")
                    del active[proc]
                    self._reap(proc)
            _launch()
        return records

    # -- GEPAAdapter API --------------------------------------------------- #

    def evaluate(
        self, batch: list[DapoTask], candidate: dict[str, str], capture_traces: bool = False
    ) -> EvaluationBatch:
        skill_text = candidate[SKILL_COMPONENT]
        idx = self._eval_counter
        self._eval_counter += 1
        system_prompt = compose_system_prompt(skill_text)
        rollout_out = self.run_root / f"eval{idx}"

        records = self._rollout_batch(batch, rollout_out, system_prompt)

        outputs: list[dict] = []
        scores: list[float] = []
        trajectories: list[dict] = []
        for task, record in zip(batch, records, strict=False):
            correct = bool(record.get("correct"))
            if not record.get("error"):
                self.rollout_usages.append(record.get("usage") or {})
            outputs.append({"id": task.trace_id, "correct": correct,
                            "prediction": record.get("prediction", "")})
            scores.append(1.0 if correct else 0.0)
            if capture_traces:
                events: list[dict] = []
                ep = record.get("events_path")
                if ep and Path(ep).exists():
                    try:
                        events = json.loads(Path(ep).read_text(encoding="utf-8"))
                    except Exception:  # noqa: BLE001
                        events = []
                trace = _trace_from_events(events) if events else (
                    record.get("final_response") or f"(rollout error: {record.get('error')})"
                )
                trajectories.append({
                    "id": task.trace_id,
                    "problem": task.prompt,
                    "ground_truth": task.ground_truth,
                    "prediction": record.get("prediction", ""),
                    "correct": correct,
                    "error": record.get("error", ""),
                    "trace": trace,
                })

        return EvaluationBatch(
            outputs=outputs, scores=scores,
            trajectories=trajectories if capture_traces else None,
        )

    def make_reflective_dataset(
        self, candidate: dict[str, str], eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict]]:
        records: list[dict] = []
        for traj in eval_batch.trajectories or []:
            if traj["correct"]:
                feedback = (
                    "CORRECT. The agent produced the right final answer "
                    f"({traj['prediction']!r}). Reinforce whatever guidance made this succeed; "
                    "do not add friction that would slow down easy problems."
                )
            else:
                err = traj.get("error")
                if err:
                    feedback = (
                        f"WRONG (rollout error: {err}). The agent failed to produce a usable "
                        "answer. Improve the skill so an agent reliably finishes and emits a "
                        "single boxed final answer."
                    )
                else:
                    feedback = (
                        f"WRONG. The agent answered {traj['prediction']!r} but the correct answer "
                        f"is {traj['ground_truth']!r}. Improve the skill so an agent following it "
                        "would reach and correctly format this answer — e.g. clearer derivation "
                        "discipline, verification against the question asked, and stripping the "
                        "answer to its bare value."
                    )
            records.append({
                "Inputs": {"problem": traj["problem"]},
                "Generated Outputs": traj["trace"],
                "Feedback": feedback,
            })
        return {SKILL_COMPONENT: records}


# --------------------------------------------------------------------------- #
# Reflection LM (litellm against the eval proxy)
# --------------------------------------------------------------------------- #


def _strip_think(text: str) -> str:
    """Strip reflection-model chain-of-thought and any single outer code-fence wrapper.

    Reasoning models (e.g. MiniMax) emit ``<think>…</think>`` before the actual skill,
    and often wrap the final skill in a ```` ```yaml `` fence. GEPA stores the raw
    reflection output verbatim as the candidate, so without this the whole think-block
    (and fence markers) leak into ``proposed_skill.md`` and thus into the agent's system
    prompt. Keep only the intended skill markdown. Robust to a missing opening ``<think>``
    (some models emit only the closing tag).
    """
    if not text:
        return text
    if "</think>" in text:  # drop everything up to and including the last </think>
        text = text.rsplit("</think>", 1)[-1]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("```"):  # unwrap a single outer fence
        lines = lines[1:]
        fence_idx = [i for i, ln in enumerate(lines) if ln.lstrip().startswith("```")]
        if len(fence_idx) % 2 == 1:  # unmatched trailing fence == the wrapper's close
            j = fence_idx[-1]
            lines = lines[:j] + lines[j + 1:]
        text = "\n".join(lines).strip()
    return text


def make_reflection_lm(*, model: str, base_url: str, api_key: str, usage_sink: list[dict]):
    import litellm

    def _call(prompt) -> str:
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        resp = litellm.completion(
            model=model, messages=messages, base_url=base_url or None,
            api_key=api_key or None, drop_params=True,
        )
        u = getattr(resp, "usage", None)
        try:
            cost = float(resp._hidden_params.get("response_cost") or 0.0)
        except Exception:  # noqa: BLE001
            cost = 0.0
        usage_sink.append({
            "prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(u, "total_tokens", 0) or 0),
            "cost": cost,
        })
        return _strip_think(resp.choices[0].message.content or "")

    return _call


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--output-dir", type=Path, required=True, metavar="DIR")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, metavar="DIR")
    p.add_argument("--split", default="train")
    p.add_argument("--seed-skill", type=Path, default=DEFAULT_SEED_SKILL, metavar="PATH")
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument("--llm-provider", default="eval_proxy")
    p.add_argument("--base-url", default="")
    p.add_argument("--api-key", default="")
    p.add_argument("--secrets-file", default=".eval_proxy.secrets.json", metavar="PATH")
    p.add_argument("--max-iter", type=int, default=12, metavar="N")
    p.add_argument("--reflection-model", default="openai/gpt-5.4-mini",
                   help="litellm model id for reflection (priced; default gpt-5.4-mini).")
    p.add_argument("--limit", type=int, default=None, metavar="N",
                   help="Use only the first N train instances.")
    p.add_argument("--val-size", type=int, default=20, metavar="N",
                   help="Hold out the LAST N sliced instances as GEPA's internal valset for "
                   "candidate selection (capped so cost doesn't scale with --limit). 0 => "
                   "valset==trainset (overlap). Final skill quality is judged on the held-out "
                   "eval split separately, not this valset.")
    p.add_argument("--max-metric-calls", type=int, default=80, metavar="N")
    p.add_argument("--reflection-minibatch-size", type=int, default=3, metavar="N")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=4, metavar="N")
    p.add_argument("--rollout-timeout", type=float, default=420.0, metavar="SEC",
                   help="Kill + score-as-error any rollout over this wall-clock (0 disables).")
    p.add_argument("--no-eval-cache", dest="cache_evaluation", action="store_false")
    p.set_defaults(cache_evaluation=True)
    p.add_argument("--fresh", action="store_true",
                   help="Wipe <output-dir>/gepa_state before starting (default resumes).")
    return p.parse_args()


def main() -> None:
    _register_extra_models()
    args = parse_args()

    if not args.seed_skill.exists():
        print(f"ERROR: seed skill not found: {args.seed_skill}", file=sys.stderr)
        sys.exit(1)

    cfg = _resolve_llm_config(
        model=args.model, provider=args.llm_provider, base_url=args.base_url,
        api_key=args.api_key, secrets_file=args.secrets_file,
    )

    instances = load_tasks(args.data_dir, args.split, limit=args.limit)
    if not instances:
        print("ERROR: no tasks after slicing.", file=sys.stderr)
        sys.exit(1)

    if args.val_size and 0 < args.val_size < len(instances):
        trainset = instances[: len(instances) - args.val_size]
        valset = instances[len(instances) - args.val_size :]
        overlap = False
    else:
        trainset = list(instances)
        valset = list(instances)
        overlap = True

    seed_skill_text = args.seed_skill.read_text(encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gepa_state_dir = args.output_dir / "gepa_state"
    if args.fresh and gepa_state_dir.exists():
        shutil.rmtree(gepa_state_dir)
        print(f"Wiped existing GEPA state: {gepa_state_dir}")
    resuming = gepa_state_dir.exists()

    adapter = DapoSkillAdapter(
        cfg=cfg, run_root=args.output_dir / "gepa_rollouts", max_iter=args.max_iter,
        concurrency=args.concurrency, rollout_timeout=args.rollout_timeout,
    )
    reflection_usages: list[dict] = []
    reflection_lm = make_reflection_lm(
        model=args.reflection_model, base_url=cfg.base_url, api_key=cfg.api_key,
        usage_sink=reflection_usages,
    )

    print("GEPA tune SKILL.md on DAPO-Math")
    print(f"  Split / slice     : {args.split}  train={len(trainset)}  val={len(valset)}"
          f"{'  (overlap)' if overlap else ''}")
    print(f"  Task model        : {cfg.model}  @ {cfg.base_url}")
    print(f"  Reflection model  : {args.reflection_model}")
    print(f"  Budget            : max_metric_calls={args.max_metric_calls}  "
          f"reflection_minibatch={args.reflection_minibatch_size}")
    print(f"  Rollout           : concurrency={args.concurrency}  "
          f"timeout={args.rollout_timeout:.0f}s")
    print(f"  State / resume    : {gepa_state_dir}  ({'RESUMING' if resuming else 'fresh'})")

    result = gepa.optimize(
        seed_candidate={SKILL_COMPONENT: seed_skill_text},
        trainset=trainset, valset=valset, adapter=adapter, reflection_lm=reflection_lm,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.reflection_minibatch_size,
        cache_evaluation=args.cache_evaluation, display_progress_bar=True,
        seed=args.seed, run_dir=str(gepa_state_dir),
    )

    best_skill = _strip_think(result.best_candidate[SKILL_COMPONENT])
    proposed = args.output_dir / "proposed_skill.md"
    proposed.write_text(best_skill, encoding="utf-8")

    val_scores = getattr(result, "val_aggregate_scores", None)
    best_idx = getattr(result, "best_idx", None)
    best_score = seed_score = None
    if isinstance(val_scores, (list, tuple)) and val_scores:
        seed_score = val_scores[0]
        if isinstance(best_idx, int) and 0 <= best_idx < len(val_scores):
            best_score = val_scores[best_idx]

    rollout_total = _sum_usages(adapter.rollout_usages)
    refl_total = {
        "prompt_tokens": sum(int(u.get("prompt_tokens", 0)) for u in reflection_usages),
        "completion_tokens": sum(int(u.get("completion_tokens", 0)) for u in reflection_usages),
        "total_tokens": sum(int(u.get("total_tokens", 0)) for u in reflection_usages),
        "cost": sum(float(u.get("cost", 0.0)) for u in reflection_usages),
        "calls": len(reflection_usages),
    }
    summary = {
        "task_model": cfg.model, "reflection_model": args.reflection_model,
        "split": args.split, "train_size": len(trainset), "val_size": len(valset),
        "max_metric_calls": args.max_metric_calls,
        "total_metric_calls": getattr(result, "total_metric_calls", None),
        "num_candidates": getattr(result, "num_candidates", None),
        "best_idx": best_idx, "seed_val_score": seed_score, "best_val_score": best_score,
        "val_aggregate_scores": list(val_scores) if val_scores is not None else None,
        "changed_from_seed": best_skill.strip() != seed_skill_text.strip(),
        "n_agent_rollouts": len(adapter.rollout_usages),
        "agent_rollouts_total": rollout_total,
        "reflection_total": refl_total,
        "combined_cost": float(rollout_total.get("cost", 0.0)) + refl_total["cost"],
    }
    (args.output_dir / "gepa_result.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== GEPA done ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {proposed}")


if __name__ == "__main__":
    main()
