"""Trace and TraceEvent types.

Traces are lists of JSON-serializable event dicts. At this layer we treat
them as opaque blobs — no schema is enforced. If richer typed events are needed
downstream, the openhands software-agent-sdk Event model can be used to parse
individual events.
"""

from typing import Any, NewType

# A single event within a trace: an arbitrary JSON-serializable dict.
# NewType gives pyright a distinct type to check against while staying zero-cost
# at runtime. When we later adopt the SDK's typed Event models, we swap the
# underlying type here and pyright catches every callsite that needs updating.
TraceEvent = NewType("TraceEvent", dict[str, Any])

# A trace: an ordered list of events from a single agent run.
Trace = NewType("Trace", list[TraceEvent])
