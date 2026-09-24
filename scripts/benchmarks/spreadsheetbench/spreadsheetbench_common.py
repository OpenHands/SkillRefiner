"""Pure, LLM-free helpers shared across the SpreadsheetBench ablation harness.

Kept deliberately dependency-light (stdlib only) so both the agent runner and
the trace converter can import it, and so every function here is trivially
unit-testable without a network or an LLM.

Contents:
  - build_system_prompt: transform Trace2Skill's cli_only_full_system_v1.txt
    into a system prompt appropriate for openhands-sdk's native tool-calling
    (drop the text-parsed ReAct "Action:/Observation:" protocol; keep the
    task-context framing and workflow guidance).
  - find_input_files / output_name_for_input: SpreadsheetBench spreadsheet-dir
    file-naming conventions, mirroring Trace2Skill's SpreadsheetBenchRunner so
    outputs land where evaluate_with_official.py expects them.
"""

from __future__ import annotations

import os
import re

# --------------------------------------------------------------------------- #
# Reasoning-model output cleanup
# --------------------------------------------------------------------------- #


def strip_think(text: str) -> str:
    """Strip a reasoning model's chain-of-thought and any single outer code-fence.

    Reasoning models (e.g. MiniMax) emit ``<think>...</think>`` before the actual
    content, and often wrap the real payload in a ```` ```yaml `` / ```` ```markdown ``
    fence. Callers that persist a raw completion verbatim as a skill file (parametric
    seed generation, GEPA's reflection candidate) must run it through this first, or
    the think-block (and fence markers) leak into the skill and end up injected into
    every downstream agent's system prompt. Robust to a missing opening ``<think>``
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


# --------------------------------------------------------------------------- #
# System-prompt transformation
# --------------------------------------------------------------------------- #

# "## <header>" sections describing Trace2Skill's own text-parsed ReAct envelope
# (JSON "Action:" blocks, the "ACTION: TASK_COMPLETE" sentinel). None of this
# applies under openhands-sdk, which dispatches real tool calls and ends a turn
# when the model stops calling tools. Dropped wholesale.
_DROP_SECTION_HEADERS = frozenset(
    {
        "CRITICAL: Action Format Requirements",
        "Completing the Task",
        "Action Examples",
    }
)

# Individual lines inside kept sections that describe the Action/Observation
# loop mechanics. Matched by line prefix (after stripping) so exact trailing
# wording can drift without breaking the removal.
_REACT_LOOP_LINE_PREFIXES = (
    "The tool call you write is an action",
    "This Action/Observation can repeat",
)

# The workflow's final step tells the agent to emit the ReAct completion
# sentinel. Rewrite it for native tool-calling instead of dropping it, so the
# numbered workflow stays coherent.
_WORKFLOW_COMPLETION_SENTINEL = "ACTION: TASK_COMPLETE"
_WORKFLOW_COMPLETION_REPLACEMENT = (
    "5. **Complete**: Once the output file is created and verified correct, "
    "you are done — no explicit completion signal is needed."
)


def build_system_prompt(raw: str) -> str:
    """Transform the raw cli_only system prompt for openhands-sdk tool-calling.

    Keeps the task framing (role, field descriptions, restrictions, workflow)
    and strips the text-parsed ReAct protocol sections/lines. See module
    docstring for rationale.
    """
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in raw.splitlines():
        if line.startswith("## "):
            blocks.append(current)
            current = [line]
        else:
            current.append(line)
    blocks.append(current)

    kept: list[list[str]] = []
    for block in blocks:
        header = block[0][3:].strip() if block and block[0].startswith("## ") else ""
        if header in _DROP_SECTION_HEADERS:
            continue
        cleaned: list[str] = []
        for line in block:
            stripped = line.strip()
            if any(stripped.startswith(p) for p in _REACT_LOOP_LINE_PREFIXES):
                continue
            if _WORKFLOW_COMPLETION_SENTINEL in line:
                cleaned.append(_WORKFLOW_COMPLETION_REPLACEMENT)
                continue
            cleaned.append(line)
        kept.append(cleaned)

    text = "\n".join("\n".join(block) for block in kept)
    # Removing sentences/sections leaves runs of blank lines; collapse them.
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip() + "\n"


# --------------------------------------------------------------------------- #
# SpreadsheetBench file-naming conventions
# --------------------------------------------------------------------------- #


def find_input_files(files: list[str]) -> list[str]:
    """Return the input spreadsheet file(s) in a SpreadsheetBench instance dir.

    Mirrors Trace2Skill's SpreadsheetBenchRunner._find_input_files precedence:
    standard ``*_input.xlsx`` first, then verified ``*_init.xlsx``, then the
    simple ``initial.xlsx`` / ``input.xlsx`` names, then any ``.xlsx`` that
    isn't a ground-truth / output artifact. An instance may have more than one
    test case (multiple input files), returned sorted.
    """
    input_files = sorted(f for f in files if f.endswith("_input.xlsx"))
    if input_files:
        return input_files

    init_files = sorted(f for f in files if f.endswith("_init.xlsx"))
    if init_files:
        return init_files

    for simple_name in ("initial.xlsx", "input.xlsx"):
        if simple_name in files:
            return [simple_name]

    return sorted(
        f
        for f in files
        if f.endswith(".xlsx")
        and "answer" not in f.lower()
        and "output" not in f.lower()
        and "golden" not in f.lower()
    )


def output_name_for_input(input_file: str) -> str:
    """Map an input filename to the output filename evaluate_with_official expects.

    ``*_input.xlsx`` / ``*_init.xlsx`` swap the suffix for ``_output.xlsx``;
    everything else appends ``_output`` to the stem (so ``initial.xlsx`` becomes
    ``initial_output.xlsx``, which is exactly what the official evaluator derives
    from a bare ``golden.xlsx`` ground truth).
    """
    if "_input.xlsx" in input_file:
        return input_file.replace("_input.xlsx", "_output.xlsx")
    if "_init.xlsx" in input_file:
        return input_file.replace("_init.xlsx", "_output.xlsx")
    base = os.path.splitext(input_file)[0]
    return f"{base}_output.xlsx"


# --------------------------------------------------------------------------- #
# Task prompt + spreadsheet preview (parity with Trace2Skill's runner)
# --------------------------------------------------------------------------- #


def spreadsheet_content_preview(file_path: str, max_rows: int = 5) -> str:
    """Render the first rows of a spreadsheet as SpreadsheetBench expects.

    Mirrors Trace2Skill's ``get_spreadsheet_content``: one tuple-like line per
    row, truncated after ``max_rows`` with a "(N more rows)" marker. Any read
    error is returned as a bracketed message rather than raised, matching the
    reference's graceful degradation.
    """
    try:
        import openpyxl

        wb = openpyxl.load_workbook(file_path, data_only=True)
        ws = wb.active
        lines: list[str] = []
        for i, row in enumerate(ws.iter_rows(values_only=True), 1):
            if i > max_rows:
                lines.append(f"... ({ws.max_row - max_rows} more rows)")
                break
            row_values = [str(cell) if cell is not None else "" for cell in row]
            lines.append(str(tuple(row_values)))
        wb.close()
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001 - parity with reference behavior
        return f"[Could not read spreadsheet: {exc}]"


def build_task_prompt(
    *,
    working_dir: str,
    input_file: str,
    output_file: str,
    instruction: str,
    spreadsheet_content: str,
    instruction_type: str,
    answer_position: str,
) -> str:
    """Build the per-instance task message, matching Trace2Skill's official format.

    Uses the same field layout as ``BaseSpreadsheetAgent.build_task_prompt`` so
    the agent sees the exact context the SpreadsheetBench system prompt describes.
    """
    return f"""Below is the spreadsheet manipulation question you need to solve:

### working_directory
{working_dir}

### instruction
{instruction}

### spreadsheet_path
{input_file}

### spreadsheet_content
{spreadsheet_content}

### instruction_type
{instruction_type}

### answer_position
{answer_position}

### output_path
{output_file}

---
**REMINDER**: You can ONLY access files within `{working_dir}`. Save output to the exact \
path: `{output_file}`
---

Execute Python code to solve the question and save the modified spreadsheet to the exact \
output_path shown above."""
