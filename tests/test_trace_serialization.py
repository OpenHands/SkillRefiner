from skill_refiner.trace import Trace, deserialize_trace, serialize_trace
from skill_refiner.trace.store import TraceEvent


def test_serialize_trace_returns_text():
    event = {"kind": "raw_span", "output": '[{"role": "assistant", "content": "hi"}]'}
    trace: Trace = Trace([event])
    out = serialize_trace(trace)
    assert isinstance(out, str)


def test_serialize_empty_trace_returns_empty_string():
    trace: Trace = Trace([])
    assert serialize_trace(trace) == ""


def test_deserialize_trace_round_trips_a_message_event():
    from openhands.sdk.event import LLMConvertibleEvent, MessageEvent
    from openhands.sdk.llm import Message, TextContent

    message = Message(role="assistant", content=[TextContent(text="hello")])
    event = MessageEvent(source="agent", llm_message=message)
    raw = event.model_dump(mode="json")
    trace: Trace = Trace([TraceEvent(raw)])

    events = deserialize_trace(trace)

    assert len(events) == 1
    assert isinstance(events[0], LLMConvertibleEvent)
    assert isinstance(events[0], MessageEvent)


def test_deserialize_trace_drops_events_that_do_not_validate():
    trace: Trace = Trace([TraceEvent({"kind": "raw_span", "output": "not a real sdk event"})])
    assert deserialize_trace(trace) == []


def test_deserialize_empty_trace_returns_empty_list():
    assert deserialize_trace(Trace([])) == []
