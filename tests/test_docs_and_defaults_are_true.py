"""A documented claim that was false, pinned so it cannot drift back.

`baselines/README.md` told readers to point `--data-dir` at their own
SpreadsheetBench checkout. No such flag exists on any of the three SpreadsheetBench
scripts; two of them have no dataset flag at all.
"""

import ast
from pathlib import Path

import pytest

ART = Path(__file__).resolve().parent.parent
BASELINES = ART / "baselines"


# ---------------------------------------------------------------------------
# I5: the dataset flags named in baselines/README.md have to be the real ones.
# ---------------------------------------------------------------------------


def _argparse_flags(path: Path) -> set[str]:
    """Every string literal passed positionally to an `add_argument` call."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    flags: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    flags.add(arg.value)
    return flags


_SPREADSHEETBENCH_BASELINES = (
    "gepa/gepa_tune_skill.py",
    "gepa/gepa_bootstrap_pipeline.py",
    "trace2skill/trace2skill_bootstrap_pipeline.py",
)


@pytest.mark.parametrize("rel", _SPREADSHEETBENCH_BASELINES)
def test_no_spreadsheetbench_baseline_has_a_data_dir_flag(rel):
    assert "--data-dir" not in _argparse_flags(BASELINES / rel)


def test_gepa_tune_skill_is_the_one_that_takes_data_path():
    assert "--data-path" in _argparse_flags(BASELINES / "gepa" / "gepa_tune_skill.py")
    assert "--data-path" not in _argparse_flags(
        BASELINES / "gepa" / "gepa_bootstrap_pipeline.py"
    )
    assert "--data-path" not in _argparse_flags(
        BASELINES / "trace2skill" / "trace2skill_bootstrap_pipeline.py"
    )


def test_gepa_bootstrap_forwards_unrecognized_flags_so_data_path_reaches_the_tuner():
    """The README's "pass --data-path and it arrives there" rests on this."""
    source = (BASELINES / "gepa" / "gepa_bootstrap_pipeline.py").read_text(encoding="utf-8")
    assert "parse_known_args" in source
    assert '"--data-path"' not in source  # i.e. not intercepted by the wrapper


def test_the_baselines_readme_does_not_name_a_flag_that_does_not_exist():
    text = (BASELINES / "README.md").read_text(encoding="utf-8")
    real = set().union(*(_argparse_flags(BASELINES / rel) for rel in _SPREADSHEETBENCH_BASELINES))
    assert "--data-dir" not in real  # guard the guard
    # The README may still mention `--data-dir` to say it does NOT exist.
    for line in text.splitlines():
        if "--data-dir" in line:
            assert "There is no `--data-dir` flag" in line, (
                f"baselines/README.md names --data-dir as if it existed: {line!r}"
            )
