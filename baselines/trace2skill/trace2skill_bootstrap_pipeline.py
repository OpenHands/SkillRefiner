"""Generate a SpreadsheetBench xlsx skill via Trace2Skill's skill_evolver: seed it from
the model's own parametric knowledge (one LLM call, no traces), then actually RUN that
seed skill on the SpreadsheetBench train split, and evolve it against its OWN rollout
traces (self-referential, matching how skill_evolver is used everywhere else in this
repo — e.g. dapo_trace2skill_bootstrap_pipeline.py evolves its seed using that seed's
own traces, never a different skill's traces).

Usage:
  uv run python baselines/trace2skill/trace2skill_bootstrap_pipeline.py \\
      --output-dir results/spreadsheetbench/gpt-5.4-mini/trace2skill_bootstrap \\
      --model gpt-5.4-mini --llm-provider eval_proxy \\
      --base-url https://your-llm-proxy.example.com \\
      --secrets-file .eval_proxy.secrets.json \\
      --rollout-concurrency 2 --concurrency 8

  # Smoke test (15 instances):
  uv run python .../trace2skill_bootstrap_pipeline.py --limit 15 \\
      --output-dir /tmp/trace2skill_bootstrap_smoke ...
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = REPO_ROOT / "scripts"  # holds _llm_config.py
TRACE2SKILL_DIR = REPO_ROOT / "baselines" / "trace2skill"
ERROR_ANALYSIS_SCRIPT = TRACE2SKILL_DIR / "analysis" / "run_error_analysis_llm.py"
SUCCESS_ANALYSIS_SCRIPT = TRACE2SKILL_DIR / "analysis" / "run_success_analysis_llm.py"

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Offline guard: skill_refiner/__init__.py sets LITELLM_LOCAL_MODEL_COST_MAP before
# litellm's import-time HTTPS GET to raw.githubusercontent.com. It must run ahead of the
# openhands/litellm imports below, and a plain `import skill_refiner` cannot -- ruff's
# isort sorts `openhands` first, which is precisely how the guard got skipped here.
__import__("skill_refiner")  # noqa: F401 - imported for its import-time side effect

from _llm_config import _resolve_llm_config  # noqa: E402
from skill_refiner.pipeline import _load_traces  # noqa: E402

_SPREADSHEETBENCH_DIR = REPO_ROOT / "scripts" / "benchmarks" / "spreadsheetbench"
CANONICAL_XLSX_SKILL_DIR = TRACE2SKILL_DIR / "spreadsheet_agent" / "skills" / "xlsx"
SPREADSHEETBENCH_RUNNER = _SPREADSHEETBENCH_DIR / "spreadsheetbench_agent_runner.py"
DEFAULT_TRAIN_SPLIT_SIZE = 200  # spreadsheetbench_agent_runner.py's train = [0:200]

if str(_SPREADSHEETBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_SPREADSHEETBENCH_DIR))

from convert_traces_to_ablation_format import convert  # noqa: E402
from eval_skill_on_spreadsheetbench import stage_skill_dir  # noqa: E402
from spreadsheetbench_common import strip_think  # noqa: E402

from skill_refiner.trace.serialization import serialize_trace  # noqa: E402

if TYPE_CHECKING:
    from skill_refiner.trace.store import Trace  # noqa: E402

TRANSCRIPT_PREFIX = "parametric_seed_agent"
DEFAULT_MODEL = "gpt-5.4-mini"


# --------------------------------------------------------------------------- #
# Stage 2: trace -> Trace2Skill transcript conversion
# --------------------------------------------------------------------------- #


def render_trace2skill_transcript(trace_id: str, trace: Trace, passed: bool) -> str:
    outcome = "SUCCEED" if passed else "FAILED"
    return (
        f"# Agent Trajectory: {trace_id} ({outcome})\n\n## Trajectory\n{serialize_trace(trace)}\n"
    )


def convert_traces_to_transcripts(
    traces_jsonl: Path,
    binary_rewards: Path,
    output_dir: Path,
    limit: int | None = None,
) -> list[Path]:
    rewards: dict[str, bool] = json.loads(binary_rewards.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for trace_id, trace in _load_traces(traces_jsonl, limit=limit):
        if trace_id not in rewards:
            continue
        outcome = "SUCCEED" if rewards[trace_id] else "FAILED"
        path = output_dir / f"{TRANSCRIPT_PREFIX}_{trace_id}_{outcome}.md"
        path.write_text(
            render_trace2skill_transcript(trace_id, trace, rewards[trace_id]),
            encoding="utf-8",
        )
        written.append(path)
    return written


# --------------------------------------------------------------------------- #
# Stage 1: parametric-knowledge seed skill
# --------------------------------------------------------------------------- #

_PARAMETRIC_SEED_PROMPT = (
    "Write a complete SKILL.md for an Agent skill named 'xlsx' that helps an agent "
    "create, edit, and analyze Excel spreadsheets (.xlsx, .xlsm, .csv, .tsv) using "
    "openpyxl. Base this entirely on your own general knowledge of spreadsheet editing "
    "best practices — do not reference any specific traces or examples, none are "
    "provided. Include: (1) YAML frontmatter with `name: xlsx` and a `description` "
    "field describing when to use this skill, (2) a body covering how to inspect a "
    "workbook before editing, how to make targeted edits, how to handle formulas and "
    "recalculation, and how to verify the result before finishing. Output ONLY the "
    "SKILL.md content, starting with the YAML frontmatter delimiter `---`."
)


def generate_parametric_seed(client, model: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": _PARAMETRIC_SEED_PROMPT}],
    )
    return strip_think(response.choices[0].message.content or "")


def stage_parametric_seed(seed_content: str, output_dir: Path) -> Path:
    return stage_skill_dir(
        skill="xlsx",
        canonical_skill_dir=CANONICAL_XLSX_SKILL_DIR,
        candidate_md=seed_content,
        dest_root=output_dir,
    )


PRISTINE_SEED_NAME = "parametric_seed.md"


def snapshot_parametric_seed(seed_content: str, output_dir: Path) -> Path:
    """Write an immutable copy of the parametric seed to <output-dir>/parametric_seed.md.

    ``stage_parametric_seed`` writes into ``seed_skill/xlsx/``, and Trace2Skill's
    ``skill_evolver`` later rewrites *that same directory in place* (it snapshots to
    ``xlsx_backup_<ts>/`` first, then edits). So ``seed_skill/xlsx/SKILL.md`` holds the
    seed before evolve and the evolved skill after it — reading that path later gives
    you the wrong artifact. Anything that needs the un-refined seed (a control arm, or
    another engine that should start from the same place) must read this snapshot.

    Deliberately write-once: a resumed run re-reads the on-disk SKILL.md, which by then
    is the evolved text, and must not be allowed to overwrite the real seed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / PRISTINE_SEED_NAME
    if not path.exists():
        path.write_text(seed_content, encoding="utf-8")
    return path


def recover_pristine_seed(output_dir: Path) -> Path | None:
    """Best-effort path to the un-refined seed for a run directory.

    Prefers the ``parametric_seed.md`` snapshot. Falls back to the oldest
    ``seed_skill/xlsx_backup_<ts>/SKILL.md`` that ``skill_evolver`` leaves behind, which
    lets runs made before the snapshot existed still be analysed correctly. Returns
    ``None`` when neither is present rather than silently handing back the evolved
    ``seed_skill/xlsx/SKILL.md``.
    """
    snapshot = output_dir / PRISTINE_SEED_NAME
    if snapshot.exists():
        return snapshot
    backups = sorted((output_dir / "seed_skill").glob("xlsx_backup_*"))
    for backup in backups:
        candidate = backup / "SKILL.md"
        if candidate.exists():
            return candidate
    return None


# --------------------------------------------------------------------------- #
# Stage 1.5: run the parametric seed itself on the train split (real rollout)
# --------------------------------------------------------------------------- #


def rollout_is_complete(rollout_dir: Path, expected_count: int) -> bool:
    """True if a seed-skill train rollout already has a full set of logs + eval
    results. Mirrors run_no_skill_bootstrap.py's baseline_is_complete: a directory with
    fewer logs than expected, or no eval_official_results.json at all, is treated as
    incomplete/absent rather than erroring — lets a killed rollout resume cleanly via
    --missing-only instead of restarting the whole (expensive, real-money) run.
    """
    logs_dir = rollout_dir / "logs"
    eval_path = rollout_dir / "eval_official_results.json"
    if not logs_dir.is_dir() or not eval_path.exists():
        return False
    n_logs = len(list(logs_dir.glob("*.json")))
    if n_logs < expected_count:
        return False
    try:
        results = json.loads(eval_path.read_text(encoding="utf-8")).get("results", [])
    except (OSError, json.JSONDecodeError):
        return False
    return len(results) >= expected_count


def run_seed_rollout(
    output_dir: Path,
    *,
    skill_file: Path,
    model: str,
    llm_provider: str,
    base_url: str,
    secrets_file: str,
    concurrency: int,
    max_iter: int,
    limit: int | None,
) -> None:
    """Run spreadsheetbench_agent_runner.py on the train split WITH the parametric
    seed loaded as --skill-file, so the resulting traces reflect the seed's own
    behavior (self-referential, matching skill_evolver's normal usage elsewhere in this
    repo — see module docstring)."""
    cmd = [
        "uv",
        "run",
        "python",
        str(SPREADSHEETBENCH_RUNNER),
        "--split",
        "train",
        "--skill-file",
        str(skill_file),
        "--output-dir",
        str(output_dir),
        "--model",
        model,
        "--llm-provider",
        llm_provider,
        "--base-url",
        base_url,
        "--secrets-file",
        secrets_file,
        "--concurrency",
        str(concurrency),
        "--max-iter",
        str(max_iter),
        "--missing-only",
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    print(f"STEP: parametric-seed train rollout\nCMD:  {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(
            f"WARNING: seed rollout exited {result.returncode} — "
            "scoring/converting whatever completed.",
            file=sys.stderr,
        )


# --------------------------------------------------------------------------- #
# Stages 3-4: analysis + evolution subprocess argv builders and runner
# --------------------------------------------------------------------------- #


def build_error_analysis_argv(
    *,
    logs_dir: Path,
    output_dir: Path,
    model: str,
    base_url: str,
    api_key: str,
    max_workers: int | None = None,
) -> list[str]:
    argv = [
        "uv",
        "run",
        "python",
        str(ERROR_ANALYSIS_SCRIPT),
        "--logs_dir",
        str(logs_dir),
        "--output_dir",
        str(output_dir),
        "--model",
        model,
        "--base_url",
        base_url,
        "--api_key",
        api_key,
    ]
    if max_workers is not None:
        argv += ["--max_workers", str(max_workers)]
    return argv


def build_success_analysis_argv(
    *,
    logs_dir: Path,
    output_dir: Path,
    model: str,
    base_url: str,
    api_key: str,
    max_workers: int | None = None,
) -> list[str]:
    argv = [
        "uv",
        "run",
        "python",
        str(SUCCESS_ANALYSIS_SCRIPT),
        "--logs_dir",
        str(logs_dir),
        "--output_dir",
        str(output_dir),
        "--model",
        model,
        "--base_url",
        base_url,
        "--api_key",
        api_key,
    ]
    if max_workers is not None:
        argv += ["--max_workers", str(max_workers)]
    return argv


def build_evolution_argv(
    *,
    error_dir: Path,
    success_dir: Path,
    skill_dir: Path,
    model: str,
    base_url: str,
    api_key: str,
    max_workers: int | None = None,
) -> list[str]:
    argv = [
        "uv",
        "run",
        "python",
        "-m",
        "skill_evolver.run_parallel_combined_skill_evolution",
        "--error-json",
        str(error_dir),
        "--success-json",
        str(success_dir),
        "--skill-dir",
        str(skill_dir),
        "--model",
        model,
        "--base-url",
        base_url,
        "--api-key",
        api_key,
    ]
    if max_workers is not None:
        argv += ["--max-workers", str(max_workers)]
    return argv


def _redact_argv_for_display(argv: list[str]) -> list[str]:
    redacted = list(argv)
    for i, token in enumerate(redacted):
        if token in ("--api_key", "--api-key") and i + 1 < len(redacted):
            redacted[i + 1] = "***REDACTED***"
    return redacted


def run_stage(name: str, argv: list[str], log_path: Path, cwd: Path = REPO_ROOT) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"=== {name} ===")
    print(f"$ {' '.join(_redact_argv_for_display(argv))}")
    print(f"(logging to {log_path})")
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(argv, cwd=cwd, stdout=log_file, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"stage '{name}' failed (exit {result.returncode}); see {log_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, required=True, metavar="DIR")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--analysis-model",
        default=None,
        help="Model string for the parametric-seed call and Trace2Skill's "
        "analysis/evolution stages (raw OpenAI client), if different from --model. "
        "litellm-based rollouts sometimes need a provider-prefixed model string "
        "(e.g. 'openai/MiniMaxAI/MiniMax-M2.7') while the raw-client stages need the "
        "bare name (e.g. 'MiniMaxAI/MiniMax-M2.7'). Defaults to --model.",
    )
    parser.add_argument("--llm-provider", default="eval_proxy", metavar="PROVIDER")
    parser.add_argument("--base-url", default="", metavar="URL")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--secrets-file", default=".llm.secrets.json", metavar="PATH")
    parser.add_argument("--limit", type=int, default=None, metavar="N")
    parser.add_argument(
        "--rollout-concurrency",
        type=int,
        default=2,
        metavar="N",
        help="Concurrent agent instances for the real seed-skill train rollout "
        "(stage 1.5). Kept conservative by default — this is a full agent rollout, "
        "not a lightweight LLM call.",
    )
    parser.add_argument("--max-iter", type=int, default=100, metavar="N")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        metavar="N",
        help="Threaded through as --max_workers/--max-workers to Trace2Skill's "
        "analysis and evolution scripts. Omit to use each script's own default.",
    )
    parser.add_argument(
        "--regenerate-seed",
        action="store_true",
        help="Force a fresh parametric-seed sample even if <output-dir>/seed_skill/"
        "xlsx/SKILL.md already exists. Default reuses the on-disk seed, so a re-run "
        "(e.g. after a crash, or a manually edited/cleaned seed) doesn't silently "
        "replace it with a different nondeterministic sample.",
    )
    return parser.parse_args(argv)


def main() -> None:
    from openai import OpenAI

    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _resolve_llm_config(
        model=args.model,
        provider=args.llm_provider,
        base_url=args.base_url,
        api_key=args.api_key,
        secrets_file=args.secrets_file,
    )
    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    analysis_model = args.analysis_model or cfg.model

    # Resume must read the *snapshot*, never seed_skill/xlsx/SKILL.md: evolve rewrites
    # that path in place, so on a re-run it holds the evolved skill, and reusing it
    # would promote a refined artifact into the seed slot.
    pristine = recover_pristine_seed(args.output_dir)
    seed_md_path = args.output_dir / "seed_skill" / "xlsx" / "SKILL.md"
    if pristine is not None and not args.regenerate_seed:
        print(f"STEP: parametric seed generation — reusing existing {pristine}")
        seed_content = pristine.read_text(encoding="utf-8")
    elif seed_md_path.exists() and not args.regenerate_seed:
        print(f"STEP: parametric seed generation — reusing existing {seed_md_path}")
        seed_content = seed_md_path.read_text(encoding="utf-8")
    else:
        print("STEP: parametric seed generation")
        seed_content = generate_parametric_seed(client, analysis_model)
    seed_skill_path = stage_parametric_seed(seed_content, args.output_dir / "seed_skill")
    pristine_path = snapshot_parametric_seed(seed_content, args.output_dir)
    print(f"Wrote parametric seed: {seed_skill_path}")
    print(f"Pristine seed snapshot: {pristine_path}")

    expected_count = args.limit or DEFAULT_TRAIN_SPLIT_SIZE
    rollout_dir = args.output_dir / "seed_rollout"
    if rollout_is_complete(rollout_dir, expected_count):
        print(f"Seed rollout already complete: {rollout_dir}")
    else:
        run_seed_rollout(
            rollout_dir,
            skill_file=seed_skill_path,
            model=cfg.model,
            llm_provider=args.llm_provider,
            base_url=cfg.base_url,
            secrets_file=args.secrets_file,
            concurrency=args.rollout_concurrency,
            max_iter=args.max_iter,
            limit=args.limit,
        )

    traces_jsonl, binary_rewards = convert(rollout_dir, args.output_dir)
    print(f"Converted rollout traces: {traces_jsonl}\nBinary rewards: {binary_rewards}")

    print("STEP: trace conversion")
    logs_dir = args.output_dir / "trace2skill_logs"
    written = convert_traces_to_transcripts(
        traces_jsonl, binary_rewards, logs_dir, limit=args.limit
    )
    print(f"Wrote {len(written)} transcripts -> {logs_dir}")

    error_dir = args.output_dir / "analysis" / "error"
    success_dir = args.output_dir / "analysis" / "success"
    logs_root = args.output_dir / "logs"

    run_stage(
        "error_analysis",
        build_error_analysis_argv(
            logs_dir=logs_dir,
            output_dir=error_dir,
            model=analysis_model,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            max_workers=args.concurrency,
        ),
        logs_root / "error_analysis.log",
    )
    run_stage(
        "success_analysis",
        build_success_analysis_argv(
            logs_dir=logs_dir,
            output_dir=success_dir,
            model=analysis_model,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            max_workers=args.concurrency,
        ),
        logs_root / "success_analysis.log",
    )
    run_stage(
        "evolve",
        build_evolution_argv(
            error_dir=error_dir,
            success_dir=success_dir,
            skill_dir=seed_skill_path.parent,
            model=analysis_model,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            max_workers=args.concurrency,
        ),
        logs_root / "evolve.log",
        cwd=TRACE2SKILL_DIR,
    )

    evolved_path = args.output_dir / "evolved_skill.md"
    evolved_path.write_text(seed_skill_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {evolved_path}")


if __name__ == "__main__":
    main()
