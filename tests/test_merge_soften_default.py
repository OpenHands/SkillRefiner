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
    """The merge must use _MERGE_FRAMING_SAS — the additive variant measured worse.

    The effect size is deliberately not quoted: the additive branch is not shipped,
    so this artifact cannot rerun the comparison it came from.
    """
    prompt = s._merge_prompt([], Skill(name="x", content="SKILL"), total_traces=0)
    assert f"{s._MERGE_FRAMING_SAS}\n\nReply as JSON" in prompt


def test_additive_and_per_partition_framings_are_not_shipped():
    for name in ("_MERGE_FRAMING", "_NEGATIVE_FRAMING", "_NEGATIVE_FRAMING_SAS",
                 "_POSITIVE_FRAMING", "PolaritySynthesizer"):
        assert not hasattr(s, name), name
