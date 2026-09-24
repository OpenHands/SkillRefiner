"""GEPA bootstrap: evolve a DAPO-Math skill from a parametric seed, no human-written
skill text, no ground truth.

This wrapper generates (or reuses) a parametric-knowledge-only SKILL.md — one LLM call,
no traces, see ``scripts/parametric_seed/dapo.py`` — and seeds
``gepa_tune_dapo_skill.py`` with it. That is the paper's DAPO-Math setup (an
LLM-generated initial skill), so the run is directly comparable to the SkillRefiner and
Trace2Skill DAPO arms: same benchmark, same model, same held-out split, none starting
from a human-written skill.

Two stages: (1) generate/reuse the parametric seed; (2) run
``gepa_tune_dapo_skill.py --seed-skill <seed>`` as a subprocess, which owns rollouts,
scoring, the eval cache, ``gepa_state`` resume, and token accounting — every flag it
defines is forwarded verbatim. GEPA drives its own rollouts internally during
optimization, so unlike the SkillRefiner/Trace2Skill arms there is no separate
seed-rollout-collection step here.

Usage:
  uv run python baselines/gepa/gepa_bootstrap_dapo.py \\
      --output-dir results/dapo_math_17k/gpt/gepa_bootstrap \\
      --model gpt-5.4-mini --llm-provider eval_proxy \\
      --base-url https://your-llm-proxy.example.com \\
      --secrets-file .eval_proxy.secrets.json \\
      --val-size 40 --max-metric-calls 200 --concurrency 6

  # Share one seed with the SkillRefiner/Trace2Skill bootstrap arms:
  ... --parametric-seed-file \\
      results/dapo_math_17k/gpt/bootstrap_shared/parametric_seed.md

Output layout:
  <output-dir>/parametric_seed_skill.md   raw seed text (reuse/cache file, unless
                                           --parametric-seed-file points elsewhere)
  <output-dir>/gepa/                      gepa_tune_dapo_skill.py's outputs (+gepa_state)
  <output-dir>/logs/gepa.log              GEPA stdout+stderr
  <output-dir>/bootstrap_skill.md         final artifact for held-out eval
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = REPO_ROOT / "scripts"  # holds _llm_config.py
_DAPO_DIR = REPO_ROOT / "scripts" / "benchmarks" / "dapo"
_PARAMETRIC_SEED_DIR = REPO_ROOT / "scripts" / "parametric_seed"
GEPA_TUNE_SCRIPT = _THIS_DIR / "gepa_tune_dapo_skill.py"
RUN_DAPO_AGENT = _DAPO_DIR / "run_dapo_agent.py"

for _p in (str(_SCRIPTS_DIR), str(_PARAMETRIC_SEED_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dapo import bare_model, resolve_or_generate_seed  # noqa: E402

DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_SECRETS_FILE = ".eval_proxy.secrets.json"
SEED_FILENAME = "parametric_seed_skill.md"

# Flags this wrapper sets itself; passing them through to gepa_tune_dapo_skill.py would
# either silently override the parametric seed or scatter GEPA's outputs outside
# <output-dir>.
_CONFLICTING_PASSTHROUGH = ("--seed-skill", "--output-dir")


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def resolve_seed_path(explicit: Path | None, output_dir: Path) -> Path:
    """Where the raw seed text lives: an explicit (possibly shared) file, or in-run cache."""
    return explicit if explicit is not None else output_dir / SEED_FILENAME


def reject_conflicting_passthrough(extras: list[str]) -> None:
    """Raise if the passthrough set contains a flag this wrapper owns."""
    for token in extras:
        name = token.split("=", 1)[0]
        if name in _CONFLICTING_PASSTHROUGH:
            raise ValueError(
                f"{name} is set by this wrapper and cannot be passed through to "
                f"{GEPA_TUNE_SCRIPT.name}. The seed comes from --parametric-seed-file "
                "and GEPA's outputs go under <output-dir>/gepa."
            )


def build_gepa_argv(
    *,
    seed_skill: Path,
    output_dir: Path,
    model: str,
    llm_provider: str,
    base_url: str,
    api_key: str,
    secrets_file: str,
    passthrough: list[str],
) -> list[str]:
    """argv for the GEPA tuning subprocess, with unrecognized flags forwarded verbatim.

    ``api_key`` is forwarded only when explicitly provided; otherwise
    ``gepa_tune_dapo_skill.py`` resolves it from the environment/secrets file itself,
    keeping the key out of the process table.
    """
    argv = [
        "uv", "run", "python", str(GEPA_TUNE_SCRIPT),
        "--seed-skill", str(seed_skill),
        "--output-dir", str(output_dir),
        "--model", model,
        "--llm-provider", llm_provider,
        "--base-url", base_url,
        "--secrets-file", secrets_file,
    ]
    if api_key:
        argv += ["--api-key", api_key]
    return argv + list(passthrough)


def _redact_argv_for_display(argv: list[str]) -> list[str]:
    redacted = list(argv)
    for i, token in enumerate(redacted):
        if token in ("--api_key", "--api-key") and i + 1 < len(redacted):
            redacted[i + 1] = "***REDACTED***"
    return redacted


def run_stage(name: str, argv: list[str], log_path: Path, cwd: Path = REPO_ROOT) -> None:
    """Run ``argv``, logging stdout+stderr to ``log_path``; raise on non-zero exit."""
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


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Wrapper-owned flags plus every other flag, forwarded untouched to gepa_tune_dapo_skill."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Any flag not listed here is forwarded verbatim to gepa_tune_dapo_skill.py "
        "(--max-metric-calls, --val-size, --limit, --concurrency, --reflection-model, "
        "--data-dir, --split, --fresh, ...).",
    )
    parser.add_argument("--output-dir", type=Path, required=True, metavar="DIR")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="litellm task/agent model id, forwarded to gepa_tune_dapo_skill.py.")
    parser.add_argument("--llm-provider", default="eval_proxy", metavar="PROVIDER")
    parser.add_argument("--base-url", default="", metavar="URL")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--secrets-file", default=DEFAULT_SECRETS_FILE, metavar="PATH")
    parser.add_argument("--parametric-seed-file", type=Path, default=None, metavar="PATH",
                        help="Seed SKILL.md text to reuse (e.g. the SkillRefiner/Trace2Skill "
                        "bootstrap arm's, for an engine-only comparison). Default: "
                        f"<output-dir>/{SEED_FILENAME}, generated on first run and reused "
                        "after.")
    parser.add_argument("--regenerate-seed", action="store_true",
                        help="Force a fresh parametric seed even if the seed file exists. "
                        "Note: GEPA resumes from <output-dir>/gepa/gepa_state by default, so "
                        "combine with --fresh to avoid resuming another seed's state.")
    parser.add_argument("--seed-model", default="", metavar="NAME",
                        help="Bare model string for the one seed-generation call "
                        "(default: --model with a leading 'openai/' stripped).")
    return parser.parse_known_args(argv)


def main() -> None:
    # Imported here, not at module scope: it pulls in litellm/openhands-sdk (~2s + a
    # banner), which the pure-function tests should not pay for.
    from openai import OpenAI

    from _llm_config import _resolve_llm_config

    args, passthrough = parse_args()
    try:
        reject_conflicting_passthrough(passthrough)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _resolve_llm_config(
        model=args.model,
        provider=args.llm_provider,
        base_url=args.base_url,
        api_key=args.api_key,
        secrets_file=args.secrets_file,
    )
    seed_model = args.seed_model or bare_model(cfg.model)
    seed_path = resolve_seed_path(args.parametric_seed_file, output_dir)

    print("GEPA bootstrap from a parametric seed (DAPO-Math)")
    print(f"  Task model      : {cfg.model}  @ {cfg.base_url}")
    print(f"  Seed model      : {seed_model}  (raw OpenAI-format client)")
    print(f"  Seed file       : {seed_path}")
    print(f"  Output dir      : {output_dir}")
    print(f"  GEPA passthrough: {' '.join(passthrough) if passthrough else '(none)'}")

    print("\nSTEP 1/2: parametric seed")
    existed = seed_path.exists() and not args.regenerate_seed
    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    seed_content = resolve_or_generate_seed(
        seed_path, client, seed_model, regenerate=args.regenerate_seed
    )
    print(f"{'Reused' if existed else 'Generated'} seed ({len(seed_content)} chars): {seed_path}")

    print("\nSTEP 2/2: GEPA tuning")
    gepa_dir = output_dir / "gepa"
    run_stage(
        "gepa_tune",
        build_gepa_argv(
            seed_skill=seed_path,
            output_dir=gepa_dir,
            model=args.model,
            llm_provider=args.llm_provider,
            base_url=args.base_url,
            api_key=args.api_key,
            secrets_file=args.secrets_file,
            passthrough=passthrough,
        ),
        output_dir / "logs" / "gepa.log",
    )

    print("\nSTEP 2/2 (collect artifact)")
    proposed = gepa_dir / "proposed_skill.md"
    if not proposed.exists():
        raise RuntimeError(
            f"GEPA exited 0 but {proposed} is missing; see {output_dir / 'logs' / 'gepa.log'}"
        )
    bootstrap_skill = output_dir / "bootstrap_skill.md"
    shutil.copyfile(proposed, bootstrap_skill)
    print(f"Wrote {bootstrap_skill}")

    print(
        "\nEvaluate held-out with:\n"
        f"  uv run python {RUN_DAPO_AGENT} \\\n"
        f"      --split eval --skill-file {bootstrap_skill} --output-dir <eval-out> "
        "[model/proxy args]\n"
        f"\nGEPA summary: {gepa_dir / 'gepa_result.json'}  "
        f"(check `changed_from_seed`)\n"
    )


if __name__ == "__main__":
    main()
