import importlib.util
from pathlib import Path

import pytest

from skill_refiner.pipeline import RefineConfig

RUN = Path(__file__).resolve().parent.parent / "scripts" / "ablations" / "run.py"
spec = importlib.util.spec_from_file_location("ablations_run", RUN)
ablations = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablations)


def _base():
    return RefineConfig(
        skill_file=Path("s.md"), skill_name="xlsx", raw_jsonl=Path("t.jsonl"),
        binary_rewards=Path("b.json"), output_dir=Path("out"),
    )


def test_exactly_five_ablations_ship():
    assert set(ablations.ABLATIONS) == {
        "summary", "clustering", "positive", "negative", "gating"
    }


def test_merge_softening_is_not_an_ablation():
    assert "merge-softening" not in ablations.ABLATIONS
    assert "merge_softening" not in ablations.ABLATIONS


@pytest.mark.parametrize(
    "ablate,field,value",
    [
        ("summary", "summarizer", "none"),
        ("clustering", "cluster_method", "single"),
        ("positive", "use_positive", False),
        ("negative", "use_negative", False),
        ("gating", "verify_negative_clusters", False),
    ],
)
def test_each_ablation_swaps_its_own_stage(ablate, field, value):
    assert getattr(ablations.build_config(_base(), ablate), field) == value


@pytest.mark.parametrize("ablate", sorted(ablations.ABLATIONS))
def test_each_ablation_changes_exactly_one_field(ablate):
    base, out = _base(), ablations.build_config(_base(), ablate)
    changed = [f for f in vars(base) if getattr(base, f) != getattr(out, f)]
    assert len(changed) == 1, f"{ablate} changed {changed}"


def test_unknown_ablation_is_rejected():
    with pytest.raises(ValueError):
        ablations.build_config(_base(), "merge-softening")


def _ablation_base():
    return ablations.build_base_config(
        skill_file=Path("s.md"), skill_name="xlsx", raw_jsonl=Path("t.jsonl"),
        binary_rewards=Path("b.json"), output_dir=Path("out"),
    )


def test_ablation_base_config_equals_plain_refine_config():
    """The guardrail block is gone, so base and ablation configs no longer diverge.

    The five ablation MANIFESTs recorded extra_constraints=""; with the mechanism
    removed, that is now the only thing the technique can do, and build_base_config
    has nothing left to override.
    """
    assert _ablation_base() == _base()


@pytest.mark.parametrize("ablate", sorted(ablations.ABLATIONS))
def test_no_ablation_config_carries_a_guardrail_field(ablate):
    out = ablations.build_config(_ablation_base(), ablate)
    assert not hasattr(out, "extra_constraints_file")
