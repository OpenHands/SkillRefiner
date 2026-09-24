"""The summarize stage must build byte-identical prompts on captured traces.

Fixtures are real SpreadsheetBench inputs and the summarizer prompts captured for
them, with the working-directory path and other identifying strings anonymized
consistently across both raw_content.txt and summarizer_prompt.txt in every pair.
The test checks that the prompt built by `_build_prompt` is byte-identical on these
traces: the template is what is under test. If any case diverges, the summarizer
prompt no longer matches the configuration the paper describes (Appendix A.5.1).
"""

from pathlib import Path

import pytest

from skill_refiner.summarize.llm import _build_prompt

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "golden"

# The winning run summarized SpreadsheetBench traces under the "xlsx" skill name.
SKILL_NAME = "xlsx"


def _cases():
    for polarity in ("positive", "negative"):
        d = GOLDEN / polarity
        if not d.is_dir():
            continue
        for case in sorted(d.iterdir()):
            if (case / "raw_content.txt").is_file():
                yield pytest.param(polarity, case, id=f"{polarity}/{case.name}")


CASES = list(_cases())


def test_fixtures_are_present():
    """A silently empty fixture dir would make every case below vacuously pass."""
    assert len(CASES) >= 12, f"expected >=12 golden cases, found {len(CASES)}"


def BUILD(polarity: str, trace_content: str) -> str:
    return _build_prompt(trace_content, polarity, SKILL_NAME)


@pytest.mark.parametrize("polarity,case", CASES)
def test_built_prompt_matches_the_original_run(polarity, case):
    expected = (case / "summarizer_prompt.txt").read_text()
    actual = BUILD(polarity, (case / "raw_content.txt").read_text())
    if actual == expected:
        return
    pairs = zip(actual, expected, strict=False)
    first_diff = next(
        (i for i, (a, b) in enumerate(pairs) if a != b),
        min(len(actual), len(expected)),
    )
    raise AssertionError(
        f"{case.name}: built prompt diverges from the captured prompt at char {first_diff}"
    )
