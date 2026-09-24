"""Convert categorized suggestions into skill-learning trace artifacts.

The refine pipeline (``skill_refiner.pipeline.refine``) loads JSONL records with
``trace_id`` and Laminar-like ``spans``. Suggestion
category files are already distilled, so this module groups them by original
trace and emits synthetic trace records that preserve the learning signal while
staying compatible with the existing loader.
"""

from __future__ import annotations

import json
import random
from collections import Counter, OrderedDict
from collections.abc import Iterable, Sequence  # noqa: TC003
from pathlib import Path  # noqa: TC003
from typing import Any

DEFAULT_LEARNING_TRACES_FILENAME = "suggestion_learning_traces.jsonl"
DEFAULT_LEARNING_RECORDS_FILENAME = "suggestion_learning_records.json"
DEFAULT_TRAIN_TRACES_FILENAME = "suggestion_learning_train_traces.jsonl"
DEFAULT_TEST_TRACES_FILENAME = "suggestion_learning_test_traces.jsonl"
DEFAULT_TRAIN_RECORDS_FILENAME = "suggestion_learning_train_records.json"
DEFAULT_TEST_RECORDS_FILENAME = "suggestion_learning_test_records.json"
DEFAULT_SPLIT_SUMMARY_FILENAME = "suggestion_learning_split_summary.json"
DEFAULT_TEST_SAMPLE_SIZES = (50, 10)
DEFAULT_TEST_SIZE = 0.2
DEFAULT_SPLIT_SEED = 0


def load_suggestions_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load suggestion JSONL rows from ``path``."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def write_learning_artifacts(
    suggestions: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    traces_filename: str = DEFAULT_LEARNING_TRACES_FILENAME,
    records_filename: str = DEFAULT_LEARNING_RECORDS_FILENAME,
    split_test_size: float | None = DEFAULT_TEST_SIZE,
    split_seed: int = DEFAULT_SPLIT_SEED,
    test_sample_sizes: Sequence[int] = DEFAULT_TEST_SAMPLE_SIZES,
) -> tuple[Path, Path]:
    """Write ``refine()``-compatible artifacts for ``suggestions``.

    Produces:
    - ``suggestion_learning_traces.jsonl``: JSONL accepted by
      ``RefineConfig.raw_jsonl``.
    - ``suggestion_learning_records.json``: structured sidecar with the compact
      categorized record shape (``task_type`` + ``suggestions``).
    - train/test trace and record files with category-balanced trace-level splits.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records = build_learning_records(suggestions)
    write_learning_records(records, output_dir / traces_filename, output_dir / records_filename)

    if split_test_size is not None:
        write_learning_split_artifacts(
            records,
            output_dir,
            test_size=split_test_size,
            seed=split_seed,
            test_sample_sizes=test_sample_sizes,
        )

    return output_dir / traces_filename, output_dir / records_filename


def build_learning_records(suggestions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group flat suggestion rows by trace into compact learning records."""
    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for raw in suggestions:
        trace_id = str(raw.get("trace_id") or "").strip()
        if not trace_id:
            continue
        group = grouped.setdefault(
            trace_id,
            {
                "trace_id": trace_id,
                "task_type": "code_review",
                "repo_name": str(raw.get("repo_name") or ""),
                "pr_number": str(raw.get("pr_number") or ""),
                "suggestions": [],
            },
        )
        if not group.get("repo_name") and raw.get("repo_name"):
            group["repo_name"] = str(raw.get("repo_name"))
        if not group.get("pr_number") and raw.get("pr_number"):
            group["pr_number"] = str(raw.get("pr_number"))
        group["suggestions"].append(_suggestion_to_learning_item(raw))
    return list(grouped.values())


def write_learning_records(
    records: Sequence[dict[str, Any]],
    traces_path: Path,
    records_path: Path,
) -> tuple[Path, Path]:
    """Write grouped learning records and their Laminar-like trace JSONL."""
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with traces_path.open("w", encoding="utf-8") as fh:
        for record in records:
            trace_record = learning_record_to_trace_jsonl_record(record)
            fh.write(json.dumps(trace_record, ensure_ascii=False) + "\n")
    records_path.write_text(
        json.dumps(list(records), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return traces_path, records_path


def write_learning_split_artifacts(
    records: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SPLIT_SEED,
    train_traces_filename: str = DEFAULT_TRAIN_TRACES_FILENAME,
    test_traces_filename: str = DEFAULT_TEST_TRACES_FILENAME,
    train_records_filename: str = DEFAULT_TRAIN_RECORDS_FILENAME,
    test_records_filename: str = DEFAULT_TEST_RECORDS_FILENAME,
    summary_filename: str = DEFAULT_SPLIT_SUMMARY_FILENAME,
    test_sample_sizes: Sequence[int] = DEFAULT_TEST_SAMPLE_SIZES,
) -> dict[str, Any]:
    """Write deterministic category-balanced trace-level train/test artifacts."""
    train_records, test_records = split_learning_records(
        records,
        test_size=test_size,
        seed=seed,
    )
    write_learning_records(
        train_records,
        output_dir / train_traces_filename,
        output_dir / train_records_filename,
    )
    write_learning_records(
        test_records,
        output_dir / test_traces_filename,
        output_dir / test_records_filename,
    )

    test_sample_summaries = []
    for sample_size in test_sample_sizes:
        sample_records = sample_learning_records(
            test_records,
            sample_size=sample_size,
            seed=seed + sample_size,
        )
        traces_path = output_dir / _test_sample_traces_filename(sample_size)
        records_path = output_dir / _test_sample_records_filename(sample_size)
        write_learning_records(sample_records, traces_path, records_path)
        test_sample_summaries.append(
            build_sample_summary(
                test_records,
                sample_records,
                sample_size=sample_size,
                seed=seed + sample_size,
                traces_path=traces_path,
                records_path=records_path,
            )
        )

    summary = build_split_summary(
        train_records,
        test_records,
        test_size=test_size,
        seed=seed,
        test_samples=test_sample_summaries,
    )
    (output_dir / summary_filename).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def split_learning_records(
    records: Sequence[dict[str, Any]],
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records at trace granularity while approximating category proportions."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if len(records) < 2:
        return list(records), []

    indexed_records = list(enumerate(records))
    record_counts = {index: _record_category_counts(record) for index, record in indexed_records}
    total_counts = _records_category_counts(records)
    target_test_records = max(1, min(len(records) - 1, round(len(records) * test_size)))

    rng = random.Random(seed)
    tie_breakers = {index: rng.random() for index, _ in indexed_records}
    selected = _initial_test_record_indices(
        records,
        record_counts,
        total_counts,
        target_test_records,
        test_size,
        tie_breakers,
    )
    selected = _improve_test_record_indices(
        selected,
        record_counts,
        total_counts,
        target_test_records,
        test_size,
    )

    train_records = [record for index, record in indexed_records if index not in selected]
    test_records = [record for index, record in indexed_records if index in selected]
    return train_records, test_records


def sample_learning_records(
    records: Sequence[dict[str, Any]],
    *,
    sample_size: int,
    seed: int = DEFAULT_SPLIT_SEED,
) -> list[dict[str, Any]]:
    """Select an exact-size representative trace-level subset from ``records``."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > len(records):
        raise ValueError(f"sample_size={sample_size} exceeds available records={len(records)}")
    if sample_size == len(records):
        return list(records)

    indexed_records = list(enumerate(records))
    record_counts = {index: _record_category_counts(record) for index, record in indexed_records}
    total_counts = _records_category_counts(records)
    target_fraction = sample_size / len(records)

    rng = random.Random(seed)
    tie_breakers = {index: rng.random() for index, _ in indexed_records}
    required_categories = _required_sample_categories(total_counts, sample_size)
    selected = _initial_sample_record_indices(
        records,
        record_counts,
        total_counts,
        sample_size,
        target_fraction,
        tie_breakers,
        required_categories,
    )
    selected = _improve_sample_record_indices(
        selected,
        record_counts,
        total_counts,
        sample_size,
        target_fraction,
        required_categories,
    )
    return [record for index, record in indexed_records if index in selected]


def build_split_summary(
    train_records: Sequence[dict[str, Any]],
    test_records: Sequence[dict[str, Any]],
    *,
    test_size: float,
    seed: int,
    test_samples: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Summarize train/test split sizes and per-category distributions."""
    train_counts = _records_category_counts(train_records)
    test_counts = _records_category_counts(test_records)
    total_counts = train_counts + test_counts
    categories = sorted(total_counts)
    return {
        "test_size": test_size,
        "seed": seed,
        "train_records": len(train_records),
        "test_records": len(test_records),
        "train_suggestions": sum(train_counts.values()),
        "test_suggestions": sum(test_counts.values()),
        "categories": [
            {
                "category": category,
                "total": total_counts[category],
                "train": train_counts[category],
                "test": test_counts[category],
                "test_fraction": _safe_fraction(test_counts[category], total_counts[category]),
            }
            for category in categories
        ],
        "test_samples": list(test_samples),
    }


def build_sample_summary(
    parent_records: Sequence[dict[str, Any]],
    sample_records: Sequence[dict[str, Any]],
    *,
    sample_size: int,
    seed: int,
    traces_path: Path,
    records_path: Path,
) -> dict[str, Any]:
    """Summarize a representative subset selected from the test split."""
    parent_counts = _records_category_counts(parent_records)
    sample_counts = _records_category_counts(sample_records)
    categories = sorted(parent_counts)
    return {
        "sample_size": sample_size,
        "seed": seed,
        "records": len(sample_records),
        "parent_records": len(parent_records),
        "suggestions": sum(sample_counts.values()),
        "parent_suggestions": sum(parent_counts.values()),
        "traces_path": str(traces_path),
        "records_path": str(records_path),
        "categories": [
            {
                "category": category,
                "parent": parent_counts[category],
                "sample": sample_counts[category],
                "sample_fraction": _safe_fraction(sample_counts[category], parent_counts[category]),
            }
            for category in categories
        ],
    }


def learning_record_to_trace_jsonl_record(record: dict[str, Any]) -> dict[str, Any]:
    """Render one grouped learning record as a Laminar-like trace JSONL row."""
    trace_id = str(record["trace_id"])
    prompt_text = _prompt_text(record)
    review_text = _synthetic_review_text(record)
    sources = sorted({str(s.get("source") or "unknown") for s in record.get("suggestions", [])})
    return {
        "trace_id": trace_id,
        "trace_type": "suggestion_learning",
        "status": "success",
        "metadata": {
            "repo_name": record.get("repo_name", ""),
            "pr_number": record.get("pr_number", ""),
            "task_type": record.get("task_type", "code_review"),
            "sources": sources,
            "suggestion_count": len(record.get("suggestions", [])),
            "synthetic": True,
        },
        "spans": [
            _span(
                trace_id,
                "synthetic-prompt",
                "conversation.send_message",
                input_text=prompt_text,
                output_text=prompt_text,
            ),
            _span(
                trace_id,
                "synthetic-suggestions",
                "suggestion_learning.synthetic_review",
                input_text="",
                output_text=review_text,
            ),
        ],
    }


def merge_suggestion_files(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Load and concatenate suggestions from multiple JSONL files."""
    merged: list[dict[str, Any]] = []
    for path in paths:
        merged.extend(load_suggestions_jsonl(path))
    return merged


def _records_category_counts(records: Sequence[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(_record_category_counts(record))
    return counts


def _record_category_counts(record: dict[str, Any]) -> Counter[str]:
    return Counter(
        str(suggestion.get("category") or "unknown") for suggestion in record.get("suggestions", [])
    )


def _test_sample_traces_filename(sample_size: int) -> str:
    return f"suggestion_learning_test_sample_{sample_size}_traces.jsonl"


def _test_sample_records_filename(sample_size: int) -> str:
    return f"suggestion_learning_test_sample_{sample_size}_records.json"


def _required_sample_categories(total_counts: Counter[str], sample_size: int) -> set[str]:
    categories = sorted(total_counts)
    if sample_size >= len(categories):
        return set(categories)
    return {
        category
        for category, _ in sorted(total_counts.items(), key=lambda item: (-item[1], item[0]))[
            :sample_size
        ]
    }


def _initial_sample_record_indices(
    records: Sequence[dict[str, Any]],
    record_counts: dict[int, Counter[str]],
    total_counts: Counter[str],
    target_sample_records: int,
    sample_fraction: float,
    tie_breakers: dict[int, float],
    required_categories: set[str],
) -> set[int]:
    selected: set[int] = set()
    current_counts: Counter[str] = Counter()
    remaining = set(range(len(records)))

    for category in sorted(required_categories, key=lambda name: (total_counts[name], name)):
        if len(selected) >= target_sample_records or category in current_counts:
            continue
        candidates = [index for index in remaining if record_counts[index][category] > 0]
        if not candidates:
            continue
        best_index = min(
            candidates,
            key=lambda index: (
                _sample_balance_objective(
                    current_counts + record_counts[index],
                    total_counts,
                    sample_fraction,
                    required_categories,
                ),
                -record_counts[index][category],
                tie_breakers[index],
                str(records[index].get("trace_id") or ""),
            ),
        )
        selected.add(best_index)
        remaining.remove(best_index)
        current_counts.update(record_counts[best_index])

    while len(selected) < target_sample_records and remaining:
        best_index = min(
            remaining,
            key=lambda index: (
                _sample_balance_objective(
                    current_counts + record_counts[index],
                    total_counts,
                    sample_fraction,
                    required_categories,
                ),
                tie_breakers[index],
                str(records[index].get("trace_id") or ""),
            ),
        )
        selected.add(best_index)
        remaining.remove(best_index)
        current_counts.update(record_counts[best_index])
    return selected


def _improve_sample_record_indices(
    selected: set[int],
    record_counts: dict[int, Counter[str]],
    total_counts: Counter[str],
    target_sample_records: int,
    sample_fraction: float,
    required_categories: set[str],
) -> set[int]:
    current_counts: Counter[str] = Counter()
    for index in selected:
        current_counts.update(record_counts[index])

    remaining = set(record_counts) - selected
    max_swaps = max(25, target_sample_records)
    for _ in range(max_swaps):
        best_swap: tuple[int, int, Counter[str]] | None = None
        best_objective = _sample_balance_objective(
            current_counts,
            total_counts,
            sample_fraction,
            required_categories,
        )
        for removed_index in selected:
            counts_after_removal = current_counts - record_counts[removed_index]
            for added_index in remaining:
                candidate_counts = counts_after_removal + record_counts[added_index]
                candidate_objective = _sample_balance_objective(
                    candidate_counts,
                    total_counts,
                    sample_fraction,
                    required_categories,
                )
                if candidate_objective < best_objective:
                    best_swap = (removed_index, added_index, candidate_counts)
                    best_objective = candidate_objective
        if best_swap is None:
            break
        removed_index, added_index, current_counts = best_swap
        selected.remove(removed_index)
        selected.add(added_index)
        remaining.remove(added_index)
        remaining.add(removed_index)
    return selected


def _sample_balance_objective(
    counts: Counter[str],
    total_counts: Counter[str],
    sample_fraction: float,
    required_categories: set[str],
) -> float:
    missing_required = sum(1 for category in required_categories if counts[category] == 0)
    return missing_required * 1_000_000 + _split_balance_objective(
        counts,
        total_counts,
        sample_fraction,
    )


def _initial_test_record_indices(
    records: Sequence[dict[str, Any]],
    record_counts: dict[int, Counter[str]],
    total_counts: Counter[str],
    target_test_records: int,
    test_size: float,
    tie_breakers: dict[int, float],
) -> set[int]:
    selected: set[int] = set()
    current_counts: Counter[str] = Counter()
    remaining = set(range(len(records)))
    while len(selected) < target_test_records and remaining:
        best_index = min(
            remaining,
            key=lambda index: (
                _split_balance_objective(
                    current_counts + record_counts[index],
                    total_counts,
                    test_size,
                ),
                tie_breakers[index],
                str(records[index].get("trace_id") or ""),
            ),
        )
        selected.add(best_index)
        remaining.remove(best_index)
        current_counts.update(record_counts[best_index])
    return selected


def _improve_test_record_indices(
    selected: set[int],
    record_counts: dict[int, Counter[str]],
    total_counts: Counter[str],
    target_test_records: int,
    test_size: float,
) -> set[int]:
    current_counts: Counter[str] = Counter()
    for index in selected:
        current_counts.update(record_counts[index])

    remaining = set(record_counts) - selected
    max_swaps = max(25, target_test_records)
    for _ in range(max_swaps):
        best_swap: tuple[int, int, Counter[str]] | None = None
        best_objective = _split_balance_objective(current_counts, total_counts, test_size)
        for removed_index in selected:
            counts_after_removal = current_counts - record_counts[removed_index]
            for added_index in remaining:
                candidate_counts = counts_after_removal + record_counts[added_index]
                objective = _split_balance_objective(candidate_counts, total_counts, test_size)
                if objective + 1e-12 < best_objective:
                    best_objective = objective
                    best_swap = (removed_index, added_index, candidate_counts)
        if best_swap is None:
            break
        removed_index, added_index, current_counts = best_swap
        selected.remove(removed_index)
        remaining.add(removed_index)
        remaining.remove(added_index)
        selected.add(added_index)
    return selected


def _split_balance_objective(
    counts: Counter[str],
    total_counts: Counter[str],
    test_size: float,
) -> float:
    if not total_counts:
        return 0.0
    category_error = sum(
        abs(_safe_fraction(counts[category], total_counts[category]) - test_size)
        for category in total_counts
    )
    total_suggestions = sum(total_counts.values())
    suggestion_error = abs(_safe_fraction(sum(counts.values()), total_suggestions) - test_size)
    return category_error + suggestion_error


def _safe_fraction(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def _suggestion_to_learning_item(raw: dict[str, Any]) -> dict[str, Any]:
    source = str(raw.get("source") or "unknown")
    section = str(raw.get("section") or "")
    subtype = source if not section else f"{source}:{section}"
    return {
        "text": str(raw.get("body") or ""),
        "file": str(raw.get("path") or ""),
        "line": _optional_int(raw.get("line")),
        "category": str(raw.get("category_id") or "unknown"),
        "subtype": subtype,
        "source": source,
        "section": section,
        "confidence": raw.get("confidence"),
        "reason": str(raw.get("reason") or ""),
    }


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except ValueError:
        return None


def _prompt_text(record: dict[str, Any]) -> str:
    repo = record.get("repo_name") or "unknown repository"
    pr = record.get("pr_number") or "unknown PR"
    return (
        f"/codereview Review pull request {repo}#{pr}. "
        "This synthetic learning trace is derived from categorized AI and human "
        "review suggestions."
    )


def _synthetic_review_text(record: dict[str, Any]) -> str:
    suggestions = list(record.get("suggestions") or [])
    ai = [s for s in suggestions if s.get("source") == "ai"]
    human = [s for s in suggestions if s.get("source") == "human"]
    other = [s for s in suggestions if s.get("source") not in {"ai", "human"}]

    lines = [
        "SYNTHETIC CODE REVIEW LEARNING TRACE",
        "This record is derived from categorized review suggestions, not raw tool execution.",
        f"Repository: {record.get('repo_name') or 'unknown'}",
        f"Pull request: {record.get('pr_number') or 'unknown'}",
        f"Task type: {record.get('task_type') or 'code_review'}",
        "",
        "Interpretation:",
        "- AI suggestions are issues the agent surfaced in its review.",
        "- Human suggestions are reviewer feedback to learn from.",
    ]
    lines.extend(_format_source_section("AI suggestions observed", ai))
    lines.extend(_format_source_section("Human reviewer suggestions", human))
    lines.extend(_format_source_section("Other suggestions", other))
    return "\n".join(lines).strip() + "\n"


def _format_source_section(title: str, suggestions: Sequence[dict[str, Any]]) -> list[str]:
    lines = ["", f"{title} ({len(suggestions)}):"]
    if not suggestions:
        lines.append("- None recorded.")
        return lines
    for index, suggestion in enumerate(suggestions, 1):
        location = _location(suggestion)
        category = suggestion.get("category") or "unknown"
        section = suggestion.get("section") or suggestion.get("subtype") or ""
        lines.append(f"{index}. [{category}] {section} {location}".rstrip())
        lines.append(f"   Suggestion: {suggestion.get('text') or ''}")
        reason = str(suggestion.get("reason") or "").strip()
        if reason:
            lines.append(f"   Category rationale: {reason}")
    return lines


def _location(suggestion: dict[str, Any]) -> str:
    file = str(suggestion.get("file") or "").strip()
    line = suggestion.get("line")
    if file and line is not None:
        return f"({file}:{line})"
    if file:
        return f"({file})"
    return ""


def _span(
    trace_id: str,
    span_id: str,
    name: str,
    *,
    input_text: str,
    output_text: str,
) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "name": name,
        "span_type": "synthetic",
        "input_text": input_text,
        "output_text": output_text,
        "start_time": "",
        "end_time": "",
        "status": "success",
        "attributes": {},
        "tags": [],
    }
