import importlib.util
from pathlib import Path

import pytest

SEEDS = Path(__file__).resolve().parent.parent / "scripts" / "parametric_seed"
NAMES = ["spreadsheetbench", "dapo"]


@pytest.mark.parametrize("name", NAMES)
def test_parametric_seed_writer_imports_and_has_a_main(name):
    path = SEEDS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"ps_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(getattr(mod, "main", None)), f"{name}.py has no main()"


def test_crustbench_is_not_shipped():
    assert not (SEEDS / "crustbench.py").exists()
