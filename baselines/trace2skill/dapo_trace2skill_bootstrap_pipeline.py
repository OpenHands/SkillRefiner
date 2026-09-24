"""Generate a DAPO-Math skill via Trace2Skill's skill_evolver, seeded from the model's
own parametric knowledge and evolved against that seed's own rollout traces.

``--logs-dir`` must point at trace2skill logs produced by running the *same* seed
skill this run generates/reuses (e.g. via ``run_dapo_agent.py --skill-file
<seed> --split train``) — evolving against unrelated traces (a no-skill or
different-skill run) defeats the point of a self-referential bootstrap. Pass
``--parametric-seed-file`` to reuse a seed already generated for that rollout (and
to share one seed with the SkillRefiner/GEPA DAPO bootstrap arms) instead of
generating a fresh one.

Usage:
  # 1) Collect the seed's own rollout on the train split (or reuse a shared one):
  uv run python scripts/parametric_seed/dapo.py \\
      --seed-path results/dapo_math_17k/gpt/bootstrap_shared/parametric_seed.md \\
      --model gpt-5.4-mini --llm-provider eval_proxy \\
      --secrets-file .eval_proxy.secrets.json
  uv run python scripts/benchmarks/dapo/run_dapo_agent.py \\
      --skill-file results/dapo_math_17k/gpt/bootstrap_shared/parametric_seed.md \\
      --split train --limit 400 \\
      --output-dir results/dapo_math_17k/gpt/bootstrap_shared/seed_rollout \\
      --model gpt-5.4-mini --llm-provider eval_proxy \\
      --secrets-file .eval_proxy.secrets.json

  # 2) Evolve against that rollout's own logs:
  uv run python baselines/trace2skill/dapo_trace2skill_bootstrap_pipeline.py \\
      --parametric-seed-file results/dapo_math_17k/gpt/bootstrap_shared/parametric_seed.md \\
      --logs-dir results/dapo_math_17k/gpt/bootstrap_shared/seed_rollout/trace2skill/logs \\
      --output-dir results/dapo_math_17k/gpt/trace2skill_bootstrap \\
      --model gpt-5.4-mini --llm-provider eval_proxy \\
      --base-url https://your-llm-proxy.example.com \\
      --secrets-file .eval_proxy.secrets.json
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = REPO_ROOT / "scripts"  # holds _llm_config.py
_PARAMETRIC_SEED_DIR = REPO_ROOT / "scripts" / "parametric_seed"
TRACE2SKILL_DIR = REPO_ROOT / "baselines" / "trace2skill"
ERROR_ANALYSIS_SCRIPT = TRACE2SKILL_DIR / "analysis" / "run_error_analysis_llm.py"
SUCCESS_ANALYSIS_SCRIPT = TRACE2SKILL_DIR / "analysis" / "run_success_analysis_llm.py"
MATH_ERROR_PROMPT = TRACE2SKILL_DIR / "analysis" / "error_analysis_system_llm_math.txt"
MATH_SUCCESS_PROMPT = TRACE2SKILL_DIR / "analysis" / "success_analysis_system_llm_math.txt"

for _p in (str(_SCRIPTS_DIR), str(_PARAMETRIC_SEED_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Offline guard: skill_refiner/__init__.py sets LITELLM_LOCAL_MODEL_COST_MAP before
# litellm's import-time HTTPS GET to raw.githubusercontent.com. It must run ahead of the
# openhands/litellm imports below, and a plain `import skill_refiner` cannot -- ruff's
# isort sorts `openhands` first, which is precisely how the guard got skipped here.
__import__("skill_refiner")  # noqa: F401 - imported for its import-time side effect

from _llm_config import _resolve_llm_config  # noqa: E402
from dapo import (  # noqa: E402,F401
    bare_model,
    generate_parametric_seed,
    resolve_or_generate_seed,
)

DEFAULT_MODEL = "gpt-5.4-mini"


# --------------------------------------------------------------------------- #
# Stage 1: parametric-knowledge seed skill
# --------------------------------------------------------------------------- #


def write_parametric_seed(seed_content: str, output_dir: Path) -> Path:
    skill_dir = output_dir / "seed_skill"
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(seed_content, encoding="utf-8")
    return skill_md


# --------------------------------------------------------------------------- #
# Stages 2-3: analysis + evolution subprocess argv builders and runner
# --------------------------------------------------------------------------- #


def build_error_analysis_argv(
    *, logs_dir: Path, output_dir: Path, model: str, base_url: str, api_key: str
) -> list[str]:
    return [
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
        "--system_prompt_path",
        str(MATH_ERROR_PROMPT),
    ]


def build_success_analysis_argv(
    *, logs_dir: Path, output_dir: Path, model: str, base_url: str, api_key: str
) -> list[str]:
    return [
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
        "--system_prompt_path",
        str(MATH_SUCCESS_PROMPT),
    ]


def build_evolution_argv(
    *, error_dir: Path, success_dir: Path, skill_dir: Path, model: str, base_url: str, api_key: str
) -> list[str]:
    return [
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
    parser.add_argument("--logs-dir", type=Path, required=True, metavar="DIR")
    parser.add_argument("--output-dir", type=Path, required=True, metavar="DIR")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--llm-provider", default="eval_proxy", metavar="PROVIDER")
    parser.add_argument("--base-url", default="", metavar="URL")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--secrets-file", default=".llm.secrets.json", metavar="PATH")
    parser.add_argument(
        "--parametric-seed-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="Reuse an already-generated seed SKILL.md (e.g. the file passed to "
        "run_dapo_agent.py --skill-file to collect --logs-dir), so this run "
        "evolves the SAME seed it was rolled out against, and so the SkillRefiner/GEPA "
        "DAPO bootstrap arms can share one seed. Default: generate a fresh one into "
        "<output-dir>/seed_skill/SKILL.md.",
    )
    parser.add_argument(
        "--regenerate-seed",
        action="store_true",
        help="Force a fresh parametric seed even if --parametric-seed-file exists.",
    )
    parser.add_argument(
        "--analysis-model",
        default=None,
        help=(
            "Model string for the seed-generation call and Trace2Skill's "
            "analysis/evolution stages (raw OpenAI client), if different from "
            "--model. litellm-based rollout stages (run outside this script) "
            "sometimes need a provider-prefixed model string (e.g. "
            "'openai/minimax-m2.7') while these raw-client stages need the bare "
            "name (e.g. 'minimax-m2.7'). Defaults to --model with a leading "
            "'openai/' stripped."
        ),
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
    analysis_model = args.analysis_model or bare_model(cfg.model)
    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    print("STEP: parametric seed generation")
    seed_path = args.parametric_seed_file or (args.output_dir / "seed_skill" / "SKILL.md")
    existed = seed_path.exists() and not args.regenerate_seed
    seed_content = resolve_or_generate_seed(
        seed_path, client, analysis_model, regenerate=args.regenerate_seed
    )
    seed_skill_path = write_parametric_seed(seed_content, args.output_dir)
    print(f"{'Reused' if existed else 'Generated'} seed: {seed_path}")
    print(f"Staged seed for evolution: {seed_skill_path}")

    error_dir = args.output_dir / "analysis" / "error"
    success_dir = args.output_dir / "analysis" / "success"
    logs_root = args.output_dir / "logs"

    run_stage(
        "error_analysis",
        build_error_analysis_argv(
            logs_dir=args.logs_dir,
            output_dir=error_dir,
            model=analysis_model,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
        ),
        logs_root / "error_analysis.log",
    )
    run_stage(
        "success_analysis",
        build_success_analysis_argv(
            logs_dir=args.logs_dir,
            output_dir=success_dir,
            model=analysis_model,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
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
        ),
        logs_root / "evolve.log",
        cwd=TRACE2SKILL_DIR,
    )

    evolved_path = args.output_dir / "evolved_skill.md"
    evolved_path.write_text(seed_skill_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {evolved_path}")


if __name__ == "__main__":
    main()
