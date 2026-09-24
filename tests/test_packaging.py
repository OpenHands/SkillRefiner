import tomllib
from pathlib import Path

ART = Path(__file__).resolve().parent.parent


def test_package_imports():
    import skill_refiner
    assert skill_refiner.__version__ == "1.0.0"


def test_no_server_or_cli_dependencies():
    deps = tomllib.loads((ART / "pyproject.toml").read_text())["project"]["dependencies"]
    names = {d.split(">")[0].split("[")[0] for d in deps}
    assert {"fastapi", "uvicorn", "click"}.isdisjoint(names)


def test_results_is_empty():
    entries = [p.name for p in (ART / "results").iterdir()]
    assert entries == [".gitkeep"], f"results/ must ship empty, found {entries}"
