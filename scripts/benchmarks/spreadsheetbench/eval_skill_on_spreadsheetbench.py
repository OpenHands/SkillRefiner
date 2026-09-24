"""Evaluate a (candidate or baseline) skill on the SpreadsheetBench held-out split.

Stages a skill directory from the canonical xlsx skill (so aux files like
recalc.py / LICENSE.txt are preserved) with an optional candidate SKILL.md
overwriting the content, then delegates to spreadsheetbench_agent_runner.py
with ``--split eval`` pointed at the staged SKILL.md.

Run it twice to compare:
  - baseline (omit --skill-file): the untouched xlsx SKILL.md
  - candidate (--skill-file .../combined/proposed_skill.md): the refined skill
then diff with compare_results.py.

Usage:
  # Candidate:
  uv run python scripts/benchmarks/spreadsheetbench/eval_skill_on_spreadsheetbench.py \\
      --skill-file results/spreadsheetbench/refine_out/combined/proposed_skill.md \\
      --output-dir results/spreadsheetbench/eval_refined

  # Baseline (untouched xlsx skill):
  uv run python scripts/benchmarks/spreadsheetbench/eval_skill_on_spreadsheetbench.py \\
      --output-dir results/spreadsheetbench/eval_baseline
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# REPO_ROOT is the artifact root (three levels up from
# scripts/benchmarks/spreadsheetbench/). Trace2Skill ships inside this artifact at
# baselines/trace2skill/ (not as a sibling "Trace2Skill/" checkout).
REPO_ROOT = Path(__file__).resolve().parents[3]
TRACE2SKILL = REPO_ROOT / "baselines" / "trace2skill"
RUNNER = Path(__file__).resolve().parent / "spreadsheetbench_agent_runner.py"
CANONICAL_XLSX_SKILL_DIR = TRACE2SKILL / "spreadsheet_agent" / "skills" / "xlsx"

DEFAULT_MODEL = "openai/MiniMaxAI/MiniMax-M2.7"
# Empty by design: forwarded as-is to spreadsheetbench_agent_runner.py's own
# --base-url (see that file for the resolution chain / fail-loud behavior).
DEFAULT_BASE_URL = ""


def stage_skill_dir(
    *,
    skill: str,
    canonical_skill_dir: Path,
    candidate_md: str | None,
    dest_root: Path,
) -> Path:
    """Build ``<dest_root>/<skill>/`` from the canonical skill dir + optional candidate.

    Copies every file from the canonical dir (aux scripts, references, LICENSE)
    so nothing is dropped, then overwrites SKILL.md with ``candidate_md`` when
    given. Fully idempotent — the destination is rebuilt each call. Returns the
    staged SKILL.md path.
    """
    skill_dest = dest_root / skill
    if skill_dest.exists():
        shutil.rmtree(skill_dest)
    shutil.copytree(canonical_skill_dir, skill_dest)
    staged_md = skill_dest / "SKILL.md"
    if candidate_md is not None:
        staged_md.write_text(candidate_md, encoding="utf-8")
    return staged_md


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--skill-file", type=Path, default=None, metavar="PATH",
        help="Candidate SKILL.md/proposed_skill.md. Omit for the baseline xlsx skill.",
    )
    parser.add_argument("--skill", default="xlsx", metavar="NAME")
    parser.add_argument(
        "--canonical-skill-dir", type=Path, default=CANONICAL_XLSX_SKILL_DIR,
        metavar="DIR", help="Canonical skill dir supplying aux files (default: xlsx).",
    )
    parser.add_argument("--output-dir", type=Path, required=True, metavar="DIR")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--llm-provider", default="eval_proxy")
    parser.add_argument("--secrets-file", default=".llm.secrets.json", metavar="PATH",
                        help="Forwarded to the runner for API-key resolution (e.g. the "
                        "eval-proxy secrets file).")
    parser.add_argument("--concurrency", type=int, default=4, metavar="N")
    parser.add_argument("--limit", type=int, default=None, metavar="N")
    parser.add_argument("--max-iter", type=int, default=100, metavar="N")
    parser.add_argument("--missing-only", action="store_true",
                        help="Skip instances whose output already exists (resume).")
    parser.add_argument("--ids", default=None, metavar="ID,ID,...",
                        help="Only run this comma-separated set of instance ids (scoped ablation). "
                             "Built-in official eval is skipped; score the subset externally.")
    args = parser.parse_args()

    if args.skill_file is not None and not args.skill_file.exists():
        print(f"ERROR: skill file not found: {args.skill_file}", file=sys.stderr)
        sys.exit(1)
    if not args.canonical_skill_dir.is_dir():
        print(f"ERROR: canonical skill dir not found: {args.canonical_skill_dir}", file=sys.stderr)
        sys.exit(1)

    candidate_md = args.skill_file.read_text(encoding="utf-8") if args.skill_file else None
    staged_md = stage_skill_dir(
        skill=args.skill,
        canonical_skill_dir=args.canonical_skill_dir,
        candidate_md=candidate_md,
        dest_root=args.output_dir / ".staged-skill",
    )
    label = "candidate" if candidate_md is not None else "baseline (canonical xlsx)"
    print(f"Staged {label} skill -> {staged_md}")

    cmd = [
        "uv", "run", "python", str(RUNNER),
        "--split", "eval",
        "--skill-file", str(staged_md),
        "--output-dir", str(args.output_dir),
        "--model", args.model,
        "--base-url", args.base_url,
        "--llm-provider", args.llm_provider,
        "--secrets-file", args.secrets_file,
        "--concurrency", str(args.concurrency),
        "--max-iter", str(args.max_iter),
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.missing_only:
        cmd += ["--missing-only"]
    if args.ids:
        cmd += ["--ids", args.ids]
    print(f"Delegating to runner: {' '.join(cmd)}")
    raise SystemExit(subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode)


if __name__ == "__main__":
    main()
