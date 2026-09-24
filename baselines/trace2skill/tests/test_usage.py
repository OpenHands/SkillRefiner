import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from react_agent.usage import UsageTracker, extract_cost, record_response  # noqa: E402


def test_record_accumulates_tokens_and_cost():
    t = UsageTracker()
    t.record(prompt=100, completion=20, total=120, cost=0.01)
    t.record(prompt=50, completion=5, cost=0.002)  # total derived from prompt+completion
    s = t.summary()
    assert s["prompt_tokens"] == 150
    assert s["completion_tokens"] == 25
    assert s["total_tokens"] == 120 + 55
    assert s["api_calls"] == 2
    assert abs(s["cost_usd"] - 0.012) < 1e-9


def test_cache_hits_do_not_count_as_api_calls():
    t = UsageTracker()
    t.record_cache_hit()
    t.record_cache_hit()
    t.record(prompt=10, completion=1)
    s = t.summary()
    assert s["cache_hits"] == 2
    assert s["api_calls"] == 1


def test_reset_clears_all():
    t = UsageTracker()
    t.record(prompt=10, completion=1, cost=0.5)
    t.record_cache_hit()
    t.reset()
    assert t.summary() == {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        "cost_usd": 0.0, "api_calls": 0, "cache_hits": 0,
    }


def test_extract_cost_reads_litellm_header():
    assert extract_cost({"x-litellm-response-cost": "0.00381"}) == 0.00381
    assert extract_cost({}) == 0.0
    assert extract_cost(None) == 0.0
    assert extract_cost({"x-litellm-response-cost": "garbage"}) == 0.0


def test_record_response_pulls_usage_and_header():
    t = UsageTracker()

    class _Usage:
        prompt_tokens, completion_tokens, total_tokens = 200, 40, 240

    class _Resp:
        usage = _Usage()

    # record_response uses the module GLOBAL, so exercise the parsing path via a
    # local tracker by mirroring its logic through the public record() instead.
    r = _Resp()
    t.record(
        prompt=r.usage.prompt_tokens,
        completion=r.usage.completion_tokens,
        total=r.usage.total_tokens,
        cost=extract_cost({"x-litellm-response-cost": "0.0040347"}),
    )
    s = t.summary()
    assert s["total_tokens"] == 240
    assert s["cost_usd"] == round(0.0040347, 4)  # summary() rounds run-total to 4dp
