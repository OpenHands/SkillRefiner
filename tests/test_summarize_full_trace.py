from skill_refiner.summarize.llm import _skill_run_header


def test_header_names_the_skill_and_passing_outcome():
    out = _skill_run_header("positive", "xlsx")
    assert "`xlsx`" in out
    assert "PASSED" in out


def test_header_names_failing_outcome():
    assert "FAILED" in _skill_run_header("negative", "xlsx")


def test_header_falls_back_without_a_skill_name():
    assert "a skill" in _skill_run_header(None, None)
