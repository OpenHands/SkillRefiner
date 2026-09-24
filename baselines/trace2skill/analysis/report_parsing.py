from __future__ import annotations

import re
from pathlib import Path


ERROR_REPORT_BASENAME = "analysis_report.md"
SUCCESS_REPORT_BASENAME = "success_analysis.md"
FAILED_PARSE_DIRNAME = "failed_to_parse"

_ERROR_REPORT_FILE_PATTERN = re.compile(r"^error_analysis_(.+)\.md$")
_SUCCESS_REPORT_FILE_PATTERN = re.compile(r"^success_analysis_(.+)\.md$")

_ERROR_ITEM_TYPE_MAP = {
    "Failure Cause Item": "failure_cause",
    "Failure Memory Item": "failure_memory",
}

_ERROR_ITEM_PATTERN = re.compile(
    r"^#\s+(Failure Cause Item|Failure Memory Item)\s+(\d+)\s*\n"
    r"(.*?)(?=\n#\s+(?:Failure Cause Item|Failure Memory Item)\s+\d+|\Z)",
    re.MULTILINE | re.DOTALL,
)
_ERROR_SECTION_PATTERN = re.compile(
    r"^##\s+{name}\s*\n(.*?)(?=\n##\s+|\Z)",
    re.MULTILINE | re.DOTALL,
)

_SUCCESS_ITEM_HEADING_PATTERN = re.compile(
    r"^(#+)\s+Success Memory Item\s+(\d+)\s*$",
    re.MULTILINE,
)
_SUCCESS_SECTION_HEADING_PATTERN = re.compile(
    r"^(#+)\s+(Title|Description|Content)\s*$",
    re.MULTILINE,
)


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = re.sub(r"^```\w*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    return stripped


def strip_think_prefix(text: str) -> str:
    if "</think>" not in text:
        return text
    return text.rsplit("</think>", 1)[-1]


def _extract_error_section(body: str, section_name: str) -> str:
    pattern = _ERROR_SECTION_PATTERN.pattern.format(name=re.escape(section_name))
    match = re.search(pattern, body, re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_error_items(text: str) -> list[dict]:
    text = strip_code_fences(strip_think_prefix(text))
    items = []
    for match in _ERROR_ITEM_PATTERN.finditer(text):
        item_type_raw = match.group(1)
        body = match.group(3).strip()
        record = {
            "type": _ERROR_ITEM_TYPE_MAP[item_type_raw],
            "number": int(match.group(2)),
            "title": _extract_error_section(body, "Title"),
            "description": _extract_error_section(body, "Description"),
            "content": _extract_error_section(body, "Content"),
        }
        if record["type"] == "failure_cause":
            record["relation_to_skill"] = _extract_error_section(body, "Relation to Skill")
        elif record["type"] == "failure_memory":
            record["skill_reflection"] = _extract_error_section(body, "Skill Reflection")
        items.append(record)
    return items


def _clean_success_section_content(text: str) -> str:
    cleaned = re.sub(r"\n(?:---+|\*\*\*+|___+)\s*\Z", "", text.strip())
    return cleaned.strip()


def parse_success_items(text: str) -> list[dict]:
    text = strip_code_fences(strip_think_prefix(text))
    items = []
    item_matches = list(_SUCCESS_ITEM_HEADING_PATTERN.finditer(text))

    for idx, match in enumerate(item_matches):
        body_start = match.end()
        body_end = item_matches[idx + 1].start() if idx + 1 < len(item_matches) else len(text)
        body = text[body_start:body_end]
        sections: dict[str, str] = {}
        section_matches = list(_SUCCESS_SECTION_HEADING_PATTERN.finditer(body))

        for section_idx, section_match in enumerate(section_matches):
            name = section_match.group(2)
            content_start = section_match.end()
            content_end = (
                section_matches[section_idx + 1].start()
                if section_idx + 1 < len(section_matches)
                else len(body)
            )
            sections[name] = _clean_success_section_content(body[content_start:content_end])

        items.append(
            {
                "type": "success_memory",
                "number": int(match.group(2)),
                "title": sections.get("Title", ""),
                "description": sections.get("Description", ""),
                "content": sections.get("Content", ""),
            }
        )

    return items


def save_failed_parse_debug_artifacts(report_path: Path, instance_id: str) -> Path:
    debug_dir = report_path.parent / FAILED_PARSE_DIRNAME / instance_id
    if report_path.parent.name == instance_id:
        debug_dir = report_path.parent.parent / FAILED_PARSE_DIRNAME / instance_id

    debug_dir.mkdir(parents=True, exist_ok=True)
    target = debug_dir / report_path.name
    target.write_text(report_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    return debug_dir


def _collect_instance_paths(
    output_dir: str,
    *,
    report_basename: str,
    flat_pattern: re.Pattern[str],
) -> list[Path]:
    base = Path(output_dir)
    if not base.exists():
        return []

    collected: list[Path] = []
    for path in sorted(base.iterdir()):
        if path.is_dir() and (path / report_basename).exists():
            collected.append(path)
            continue
        if not path.is_file():
            continue
        if path.name.endswith("_prompt.md"):
            continue
        if flat_pattern.match(path.name):
            collected.append(path)
    return collected


def _resolve_report_path(instance_path: Path, report_basename: str) -> Path:
    return instance_path / report_basename if instance_path.is_dir() else instance_path


def _extract_flat_instance_id(instance_path: Path, flat_pattern: re.Pattern[str]) -> str:
    if instance_path.is_dir():
        return instance_path.name
    match = flat_pattern.match(instance_path.name)
    if not match:
        raise ValueError(f"Unrecognized report filename: {instance_path.name}")
    return match.group(1)


def collect_error_records(output_dir: str) -> tuple[list[dict], int, int]:
    instance_paths = _collect_instance_paths(
        output_dir,
        report_basename=ERROR_REPORT_BASENAME,
        flat_pattern=_ERROR_REPORT_FILE_PATTERN,
    )
    total = 0
    passed = 0
    results: list[dict] = []

    for instance_path in instance_paths:
        report_path = _resolve_report_path(instance_path, ERROR_REPORT_BASENAME)
        total += 1
        if instance_path.is_dir():
            if not (instance_path / "evaluate_passed.flag").exists():
                continue
            passed += 1
        else:
            passed += 1

        record = {
            "instance_id": _extract_flat_instance_id(instance_path, _ERROR_REPORT_FILE_PATTERN),
            "source_file": report_path.name,
            "items": parse_error_items(report_path.read_text(encoding="utf-8", errors="replace")),
        }
        if record["items"]:
            results.append(record)
        else:
            save_failed_parse_debug_artifacts(report_path, record["instance_id"])
    return results, passed, total


def collect_success_records(output_dir: str) -> list[dict]:
    instance_paths = _collect_instance_paths(
        output_dir,
        report_basename=SUCCESS_REPORT_BASENAME,
        flat_pattern=_SUCCESS_REPORT_FILE_PATTERN,
    )
    results: list[dict] = []

    for instance_path in instance_paths:
        report_path = _resolve_report_path(instance_path, SUCCESS_REPORT_BASENAME)
        record = {
            "instance_id": _extract_flat_instance_id(instance_path, _SUCCESS_REPORT_FILE_PATTERN),
            "source_file": report_path.name,
            "items": parse_success_items(report_path.read_text(encoding="utf-8", errors="replace")),
        }
        if record["items"]:
            results.append(record)
        else:
            save_failed_parse_debug_artifacts(report_path, record["instance_id"])
    return results
