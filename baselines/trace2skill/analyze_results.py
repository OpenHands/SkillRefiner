#!/usr/bin/env python3
"""
Analyze evaluation results and match with agent log files.

Usage:
    python analyze_results.py --eval_results outputs/spreadsheetbench/eval_official_results.json --log_dir logs/

    # Show only failed instances
    python analyze_results.py --eval_results outputs/spreadsheetbench/eval_official_results.json --log_dir logs/ --failed_only

    # Show only passed instances
    python analyze_results.py --eval_results outputs/spreadsheetbench/eval_official_results.json --log_dir logs/ --passed_only

    # Export to CSV
    python analyze_results.py --eval_results outputs/spreadsheetbench/eval_official_results.json --log_dir logs/ --csv results_analysis.csv
"""

import argparse
import json
import os
import re
from dataclasses import dataclass


@dataclass
class InstanceAnalysis:
    instance_id: str
    passed: bool
    error_message: str
    log_file: str | None
    instruction: str = ""
    soft_score: float = 0.0
    hard_score: float = 0.0


def find_log_file(log_dir: str, instance_id: str) -> str | None:
    """Find log file for an instance ID."""
    if not log_dir or not os.path.isdir(log_dir):
        return None

    # Patterns to match: cli_*_agent_<id>.log, cli_*_agent_<id>.md, etc.
    # The instance_id might contain special chars like "-"
    patterns = [
        f"*_{instance_id}.log",
        f"*_{instance_id}.md",
        f"*_{instance_id}_*.log",
        f"*_{instance_id}_*.md",
    ]

    for filename in os.listdir(log_dir):
        # Check if instance_id is in the filename
        # Handle cases like: cli_only_agent_13-1.md, cli_skill_preloaded_agent_17-35.log
        if re.search(rf'_agent_{re.escape(instance_id)}\.', filename):
            return os.path.join(log_dir, filename)
        if re.search(rf'_agent_{re.escape(instance_id)}_', filename):
            return os.path.join(log_dir, filename)
        # Also check for exact match at end (before extension)
        name_without_ext = os.path.splitext(filename)[0]
        if name_without_ext.endswith(f"_{instance_id}"):
            return os.path.join(log_dir, filename)

    return None


def load_eval_results(eval_file: str) -> list[dict]:
    """Load evaluation results from JSON file."""
    with open(eval_file, "r") as f:
        data = json.load(f)

    # Handle both formats: with "results" key or direct list
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unexpected evaluation results format in {eval_file}")


def analyze_results(eval_file: str, log_dir: str | None) -> list[InstanceAnalysis]:
    """Analyze evaluation results and match with log files."""
    results = load_eval_results(eval_file)
    analyses = []

    for result in results:
        instance_id = str(result.get("id", ""))
        passed = result.get("success", False)
        soft_score = result.get("soft_score", 0.0)
        hard_score = result.get("hard_score", 0.0)

        # Get error message from test cases or top-level error
        error_message = result.get("error", "")
        if not error_message:
            test_cases = result.get("test_cases", [])
            for tc in test_cases:
                if not tc.get("passed", True):
                    error_message = tc.get("message", "")
                    break

        # Find matching log file
        log_file = find_log_file(log_dir, instance_id) if log_dir else None

        analyses.append(InstanceAnalysis(
            instance_id=instance_id,
            passed=passed,
            error_message=error_message,
            log_file=log_file,
            instruction=result.get("instruction", "")[:100],  # Truncate
            soft_score=soft_score,
            hard_score=hard_score,
        ))

    return analyses


def rename_log_files(analyses: list[InstanceAnalysis]):
    """Rename log files with _SUCCEED or _FAILED suffix based on pass/fail status."""
    for a in analyses:
        if not a.log_file or not os.path.isfile(a.log_file):
            continue

        suffix = "_SUCCEED" if a.passed else "_FAILED"
        base, ext = os.path.splitext(a.log_file)
        new_path = f"{base}{suffix}{ext}"

        if a.log_file != new_path:
            os.rename(a.log_file, new_path)
            a.log_file = new_path


def print_summary(analyses: list[InstanceAnalysis]):
    """Print summary statistics."""
    total = len(analyses)
    passed = sum(1 for a in analyses if a.passed)
    failed = total - passed
    with_logs = sum(1 for a in analyses if a.log_file)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total instances:    {total}")
    print(f"Passed:             {passed} ({passed/total*100:.1f}%)")
    print(f"Failed:             {failed} ({failed/total*100:.1f}%)")
    print(f"With log files:     {with_logs}")
    print(f"Missing log files:  {total - with_logs}")
    print("=" * 70)


def print_results(analyses: list[InstanceAnalysis], show_passed: bool, show_failed: bool):
    """Print detailed results."""
    for a in analyses:
        if a.passed and not show_passed:
            continue
        if not a.passed and not show_failed:
            continue

        status = "PASS" if a.passed else "FAIL"
        log_info = a.log_file if a.log_file else "(no log file)"

        print(f"\n[{status}] {a.instance_id}")
        print(f"  Log: {log_info}")
        if a.soft_score or a.hard_score:
            print(f"  Scores: soft={a.soft_score:.2f}, hard={a.hard_score:.0f}")
        if a.error_message:
            print(f"  Error: {a.error_message[:200]}")


def export_csv(analyses: list[InstanceAnalysis], csv_file: str):
    """Export results to CSV."""
    import csv

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "instance_id", "passed", "soft_score", "hard_score",
            "error_message", "log_file", "instruction"
        ])
        for a in analyses:
            writer.writerow([
                a.instance_id,
                a.passed,
                a.soft_score,
                a.hard_score,
                a.error_message,
                a.log_file or "",
                a.instruction,
            ])

    print(f"Exported to: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation results with log files")
    parser.add_argument(
        "--eval_results",
        type=str,
        required=True,
        help="Path to evaluation results JSON file",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory containing agent log files",
    )
    parser.add_argument(
        "--failed_only",
        action="store_true",
        help="Show only failed instances",
    )
    parser.add_argument(
        "--passed_only",
        action="store_true",
        help="Show only passed instances",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Export results to CSV file",
    )
    parser.add_argument(
        "--no_details",
        action="store_true",
        help="Show only summary, not detailed results",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of seed runs to process. When > 1, treats --eval_results as a base "
             "directory containing seed_*/ subdirs, each with eval_official_results.json "
             "(default: 1).",
    )
    parser.add_argument(
        "--base_log_dir",
        type=str,
        default=None,
        help="Base log directory containing seed_*/ subdirs (used with --repeat > 1). "
             "When set, each seed's logs are read from base_log_dir/seed_*/.",
    )
    args = parser.parse_args()

    if args.repeat > 1:
        _run_repeat_analyze(args)
        return

    # Analyze results
    analyses = analyze_results(args.eval_results, args.log_dir)

    # Rename log files based on pass/fail status
    rename_log_files(analyses)

    # Sort by instance_id
    analyses.sort(key=lambda a: a.instance_id)

    # Print summary
    print_summary(analyses)

    # Determine what to show
    if args.failed_only:
        show_passed, show_failed = False, True
    elif args.passed_only:
        show_passed, show_failed = True, False
    else:
        show_passed, show_failed = True, True

    # Print detailed results
    if not args.no_details:
        print_results(analyses, show_passed, show_failed)

    # Export to CSV if requested
    if args.csv:
        export_csv(analyses, args.csv)


def _run_repeat_analyze(args) -> None:
    """Analyze all seed_* subdirectory results when --repeat > 1."""
    base_eval_dir = args.eval_results
    if not os.path.isdir(base_eval_dir):
        print(f"ERROR: --eval_results must be a directory when --repeat > 1: {base_eval_dir}")
        import sys; sys.exit(1)

    seed_dirs = sorted(
        d for d in os.scandir(base_eval_dir)
        if d.is_dir() and d.name.startswith("seed_")
    )
    if not seed_dirs:
        print(f"No seed_* subdirectories found in {base_eval_dir}")
        import sys; sys.exit(1)

    all_analyses = []
    for seed_dir in seed_dirs:
        seed_name = seed_dir.name
        eval_file = os.path.join(seed_dir.path, "eval_official_results.json")
        if not os.path.isfile(eval_file):
            print(f"Warning: {eval_file} not found, skipping {seed_name}")
            continue

        seed_log_dir = None
        if args.base_log_dir:
            seed_log_dir = os.path.join(args.base_log_dir, seed_name)
            if not os.path.isdir(seed_log_dir):
                seed_log_dir = None

        print(f"\n{'='*60}")
        print(f"Seed: {seed_name}")
        print(f"{'='*60}")
        analyses = analyze_results(eval_file, seed_log_dir)
        rename_log_files(analyses)
        analyses.sort(key=lambda a: a.instance_id)
        print_summary(analyses)

        if not args.no_details:
            if args.failed_only:
                show_passed, show_failed = False, True
            elif args.passed_only:
                show_passed, show_failed = True, False
            else:
                show_passed, show_failed = True, True
            print_results(analyses, show_passed, show_failed)

        all_analyses.extend(analyses)

    if args.csv and all_analyses:
        export_csv(all_analyses, args.csv)


if __name__ == "__main__":
    main()
