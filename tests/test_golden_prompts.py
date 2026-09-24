"""The summarize stage must build byte-identical prompts on captured traces.

Fixtures are real inputs and outputs captured from
tmp_synthesis/refine_minclust2/full_trace_gt/ — the run that produced
minclust2_soften.md — with the working-directory path and other identifying
strings anonymized consistently across both raw_content.txt and
summarizer_prompt.txt in every pair, so the equality this test checks is
unaffected by the substitution. This test no longer proves byte-identity
against the original captured prompts; it proves the prompt *scaffolding*
built by `_build_prompt` is byte-identical on these (anonymized) traces. That
is still the real guarantee that matters here — the template is what is
under test, not the literal trace content. If any case here diverges, a moved
body is no longer faithful to the configuration the paper measured.

This test is also the gate Task 12c's loose-edge prune runs before and after.
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
        f"{case.name}: built prompt diverges from the 81.0% run at char {first_diff}"
    )
