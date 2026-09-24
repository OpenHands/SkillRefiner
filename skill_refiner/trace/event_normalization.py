"""Register SDK tool definitions so trace events can be deserialized.

Importing a tool class from ``openhands.tools`` registers it in the
``ToolDefinition`` discriminated union.  This module ensures every tool
that might appear in a stored trace is registered before we attempt
``Event.model_validate()``.

Legacy tool renames (e.g. ``DelegateTool`` → ``TaskTool``) are handled
by ``normalize_event`` which patches the raw dict before validation.
"""

from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task import TaskTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool

__all__ = [
    "FileEditorTool",
    "TaskTool",
    "TaskTrackerTool",
    "TerminalTool",
    "normalize_event",
]

_TOOL_KIND_ALIASES: dict[str, str] = {
    "DelegateTool": "TaskTool",
}


def normalize_event(raw: dict) -> dict:
    """Return a copy of *raw* with legacy tool kinds patched for the current SDK.

    Only the ``tools`` list on system-prompt events is affected — individual
    action/observation events do not carry a ``kind`` for the tool definition.
    """
    tools = raw.get("tools")
    if not isinstance(tools, list):
        return raw

    needs_patch = any(isinstance(t, dict) and t.get("kind") in _TOOL_KIND_ALIASES for t in tools)
    if not needs_patch:
        return raw

    patched_tools = []
    for t in tools:
        if isinstance(t, dict) and t.get("kind") in _TOOL_KIND_ALIASES:
            t = {**t, "kind": _TOOL_KIND_ALIASES[t["kind"]]}
        patched_tools.append(t)
    return {**raw, "tools": patched_tools}
