import json

import pytest
from openhands.sdk.skills import Skill

from skill_refiner.propose import PolarityClusterRefiner
from skill_refiner.propose.core import RefinementContext
from skill_refiner.propose.polarity import _verify_prompt


def test_verify_prompt_mentions_the_cluster_and_asks_for_support():
    class P:
        theme = "recalculate before reading values"
        soften = ["always force a recalculation"]
        reinforce = []
        suggested_edit = "force recalculation before reading any cell value"

    out = _verify_prompt(7, P(), "summaries here")
    assert "recalculate before reading values" in out
    assert "summaries here" in out


# ---------------------------------------------------------------------------
# The gate's DROP path. Everything above only inspects the prompt string.
#
# Spec §7 requires "gate admits supported, rejects unsupported", and the drop
# path had zero coverage: tests/test_pipeline_end_to_end.py's stub LLM always
# answers {"supported": True}, so polarity.py's `return None` (the drop), its
# `rejected` audit record, and both fail-open branches never executed anywhere
# in the suite. These exercise the real _is_supported against a stub LLM that
# returns each verdict in turn.
# ---------------------------------------------------------------------------

MODEL = "openai/gpt-5.4-mini"
SKILL = Skill(name="xlsx", content="# xlsx\n\nbase skill body\n")

_CLUSTER_PROPOSAL_JSON = {
    "theme": "the agent left a formula in the answer cell",
    "soften": ["trusting a formula without recalculating"],
    "reinforce": ["should be ignored on the negative path"],
    "suggested_edit": "Force a recalculation before reading any graded cell.",
}


class _Text:
    def __init__(self, text):
        self.text = text


class _Msg:
    def __init__(self, text):
        self.content = [_Text(text)]


class _Resp:
    def __init__(self, text):
        self.message = _Msg(text)


class StubLLM:
    """Answers the per-cluster proposal prompt, then the verify prompt.

    `verify_reply` is whatever the verifier should return: a JSON string, or an
    exception instance to raise (the infrastructure-failure case).
    """

    def __init__(self, verify_reply):
        self.verify_reply = verify_reply
        self.verify_calls = 0

    async def acompletion(self, *, messages, **_kwargs):
        prompt = messages[0].content[0].text
        if "auditing a proposed skill edit" in prompt:
            self.verify_calls += 1
            if isinstance(self.verify_reply, BaseException):
                raise self.verify_reply
            return _Resp(self.verify_reply)
        return _Resp(json.dumps(_CLUSTER_PROPOSAL_JSON))


class _Summary:
    """Duck-typed TraceSummary: what _summary_block and the refiner read."""

    def __init__(self, trace_id):
        self.trace_id = trace_id
        self.summary = f"the run {trace_id} left a stale cached value in the answer cell"
        self.observations = [f"observation for {trace_id}"]


def _items(n=3):
    return [_Summary(f"t{i}") for i in range(n)]


async def _propose(verify_reply, *, verify=True, polarity="negative"):
    refiner = PolarityClusterRefiner(polarity, verify=verify)
    llm = StubLLM(verify_reply)
    proposal = await refiner.propose(7, _items(), SKILL, RefinementContext(llm=llm), MODEL)
    return refiner, llm, proposal


async def test_gate_admits_a_supported_proposal():
    refiner, llm, proposal = await _propose(
        json.dumps({"supported": True, "reason": "the traces show the cached value"})
    )
    assert llm.verify_calls == 1
    assert proposal is not None
    assert proposal.theme == _CLUSTER_PROPOSAL_JSON["theme"]
    assert proposal.polarity == "negative"
    assert refiner.rejected == []


async def test_gate_drops_an_unsupported_proposal():
    """polarity.py's `return None`: the lesson never reaches the skill."""
    refiner, llm, proposal = await _propose(
        json.dumps({"supported": False, "reason": "the traces never show a formula"})
    )
    assert llm.verify_calls == 1
    assert proposal is None
    assert len(refiner.rejected) == 1


async def test_a_dropped_proposal_is_recorded_in_full_for_audit():
    """Dropping the RECORD too makes the gate unauditable after the run.

    The log line keeps only a 120-char reason and no theme, so the rejected
    entry is the only place a dropped proposal can be quoted or counted from.
    """
    reason = "the traces never show a formula being left behind"
    refiner, _, proposal = await _propose(
        json.dumps({"supported": False, "reason": reason})
    )
    assert proposal is None
    (record,) = refiner.rejected
    assert record["cluster_id"] == 7
    assert record["theme"] == _CLUSTER_PROPOSAL_JSON["theme"]
    assert record["soften"] == _CLUSTER_PROPOSAL_JSON["soften"]
    assert record["suggested_edit"] == _CLUSTER_PROPOSAL_JSON["suggested_edit"]
    assert record["verdict"] == "dropped"
    assert record["reason"] == reason  # NOT truncated, unlike the log line
    assert record["cluster_size"] == 3


@pytest.mark.parametrize(
    "verify_reply,why",
    [
        (RuntimeError("proxy 502"), "verifier LLM raised"),
        ("not json at all", "verifier reply is unparseable"),
        (json.dumps({"reason": "forgot the verdict"}), "verdict field missing"),
        (json.dumps(["not", "a", "dict"]), "verdict is not an object"),
    ],
)
async def test_gate_fails_open_on_verifier_failure(verify_reply, why):
    """An infrastructure failure is not evidence of unsupportedness.

    A fail-CLOSED gate would silently discard real signal every time the proxy
    hiccuped, and the run would look like a clean small-cluster result.
    """
    refiner, llm, proposal = await _propose(verify_reply)
    assert llm.verify_calls == 1
    assert proposal is not None, f"fail-open branch not taken: {why}"
    assert refiner.rejected == [], why


async def test_a_falsy_non_boolean_verdict_still_drops():
    """`supported` is coerced with bool(), so "" / 0 / null are rejections."""
    refiner, _, proposal = await _propose(
        json.dumps({"supported": None, "reason": "explicit null verdict"})
    )
    assert proposal is None
    assert len(refiner.rejected) == 1


async def test_the_gate_does_not_run_when_it_is_switched_off():
    """The `gating` ablation: every negative lesson reaches the skill."""
    refiner, llm, proposal = await _propose(
        json.dumps({"supported": False, "reason": "would have dropped it"}), verify=False
    )
    assert llm.verify_calls == 0
    assert proposal is not None
    assert refiner.rejected == []


async def test_the_gate_does_not_run_on_the_positive_partition():
    """_verify is `verify and polarity == "negative"`; positives are already valid."""
    refiner, llm, proposal = await _propose(
        json.dumps({"supported": False, "reason": "would have dropped it"}),
        polarity="positive",
    )
    assert llm.verify_calls == 0
    assert proposal is not None
    assert proposal.soften == []  # positive path zeroes the other polarity's field
    assert refiner.rejected == []
