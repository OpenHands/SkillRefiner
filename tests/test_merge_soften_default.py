import inspect

from openhands.sdk.skills import Skill

import skill_refiner.merge.synthesize as s
from skill_refiner.merge import merge_polarity_proposals


def test_merge_has_no_soften_at_source_switch():
    """Soften-at-source is what the merge does, not something a caller elects."""
    params = inspect.signature(merge_polarity_proposals).parameters
    assert "soften_at_source" not in params
    assert "allow_restructure" not in params


def test_the_merge_prompt_carries_the_soften_at_source_framing():
    """The merge prompt carries _MERGE_FRAMING_SAS, the paper's conflict rule (§2.6)."""
    prompt = s._merge_prompt([], Skill(name="x", content="SKILL"), total_traces=0)
    assert f"{s._MERGE_FRAMING_SAS}\n\nReply as JSON" in prompt
