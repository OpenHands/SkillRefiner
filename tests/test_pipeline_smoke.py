from pathlib import Path

from skill_refiner.pipeline import RefineConfig, refine


def test_refine_is_a_coroutine_function():
    import inspect
    assert inspect.iscoroutinefunction(refine)


def test_config_requires_its_inputs():
    cfg = RefineConfig(
        skill_file=Path("s.md"), skill_name="xlsx", raw_jsonl=Path("t.jsonl"),
        binary_rewards=Path("b.json"), output_dir=Path("out"),
    )
    assert cfg.skill_name == "xlsx"
    assert cfg.ground_truth_file is None
