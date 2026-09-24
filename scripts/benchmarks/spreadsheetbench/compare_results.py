"""Compare two SpreadsheetBench eval_official_results.json summaries side by side.

Prints baseline vs. candidate for the headline metrics (instance accuracy, soft
and hard scores) plus per-instruction-type breakdown, with deltas.

Usage:
  uv run python scripts/benchmarks/spreadsheetbench/compare_results.py \\
      --baseline results/spreadsheetbench/eval_baseline/eval_official_results.json \\
      --candidate results/spreadsheetbench/eval_refined/eval_official_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_HEADLINE_METRICS = ("instance_accuracy", "avg_soft_score", "avg_hard_score")


def diff_summaries(baseline: dict, candidate: dict) -> dict:
    """Return {metric: {baseline, candidate, delta}} for the headline metrics."""
    out: dict[str, dict[str, float]] = {}
    for metric in _HEADLINE_METRICS:
        b = float(baseline.get(metric, 0.0) or 0.0)
        c = float(candidate.get(metric, 0.0) or 0.0)
        out[metric] = {"baseline": b, "candidate": c, "delta": c - b}
    return out


def _fmt_pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def format_comparison(baseline: dict, candidate: dict) -> str:
    """Render a human-readable comparison table string."""
    diff = diff_summaries(baseline, candidate)
    lines = [
        "=" * 66,
        f"{'metric':<24}{'baseline':>12}{'candidate':>12}{'delta':>14}",
        "-" * 66,
    ]
    for metric, vals in diff.items():
        delta = vals["delta"]
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"{metric:<24}{_fmt_pct(vals['baseline']):>12}{_fmt_pct(vals['candidate']):>12}"
            f"{sign + _fmt_pct(delta):>14}"
        )
    # Per-instruction-type breakdown (if both provide it).
    b_types = baseline.get("by_instruction_type", {})
    c_types = candidate.get("by_instruction_type", {})
    if b_types or c_types:
        lines.append("-" * 66)
        lines.append("by instruction type (hard score):")
        for itype in sorted(set(b_types) | set(c_types)):
            b = float((b_types.get(itype) or {}).get("avg_hard_score", 0.0) or 0.0)
            c = float((c_types.get(itype) or {}).get("avg_hard_score", 0.0) or 0.0)
            d = c - b
            sign = "+" if d >= 0 else ""
            lines.append(
                f"  {itype or '(unknown)':<22}{_fmt_pct(b):>12}{_fmt_pct(c):>12}"
                f"{sign + _fmt_pct(d):>14}"
            )
    lines.append("=" * 66)
    return "\n".join(lines)


def _load_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    # eval_official_results.json nests metrics under "summary".
    return data.get("summary", data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--baseline", type=Path, required=True, metavar="PATH")
    parser.add_argument("--candidate", type=Path, required=True, metavar="PATH")
    args = parser.parse_args()
    print(format_comparison(_load_summary(args.baseline), _load_summary(args.candidate)))


if __name__ == "__main__":
    main()
