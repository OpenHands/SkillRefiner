import ast
import re
from pathlib import Path

ART = Path(__file__).resolve().parent.parent
CODE = [p for p in ART.rglob("*.py") if ".venv" not in p.parts]

SECRET = re.compile(r"sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}")


def test_no_secrets_anywhere():
    hits = [p for p in CODE if SECRET.search(p.read_text())]
    assert not hits, f"possible credential in {hits}"
    assert not list(ART.rglob("*.secrets.json"))


def _module_level_path_literals(path: Path):
    """Yield (lineno, string literal) for every module-level `NAME = Path("...")`.

    Deliberately narrow: only a bare `Path("...")` call with a single string
    constant argument, assigned (plainly or with an annotation) at module scope.
    Anything built from `__file__`/`REPO_ROOT` is already computed relative to
    where the file actually lives and is not the bug class this guards against.
    """
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError:
        return
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if value is None:
            continue
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "Path"
            and len(value.args) == 1
            and isinstance(value.args[0], ast.Constant)
            and isinstance(value.args[0].value, str)
        ):
            yield node.lineno, value.args[0].value


def test_default_paths_in_shipped_scripts_resolve():
    """A default pointing at a non-existent file fails only when someone runs it.

    Task 9 selected files by transitive Python imports and missed every asset
    directory, leaving DAPO -- the benchmark the README calls self-contained --
    defaulting to a skill file that did not ship.
    """
    # results/... defaults name OUTPUT destinations the scripts that default to
    # them create (e.g. prepare_dapo_math.py writes results/dapo_math_17k/...);
    # they are correctly absent until something has been run, so a missing path
    # here is not the relocation bug this test guards against.
    OUTPUT_PREFIXES = ("results/",)

    # A relative Path literal resolves against the process CWD, which is not
    # always the artifact root. skill_evolver/skill_evolving_agent.py's
    # QUICK_VALIDATE_SCRIPT is the one such case: both
    # baselines/trace2skill/*_bootstrap_pipeline.py launch the evolver with
    # `cwd=TRACE2SKILL_DIR`, so its literal resolves under baselines/trace2skill.
    # These are checked against the right base, not waived: the file is required
    # (run_parallel_skill_evolution.py and run_parallel_combined_skill_evolution.py
    # both sys.exit(1) without it).
    CWD_RELATIVE_BASES = {"baselines/trace2skill/skill_evolver": "baselines/trace2skill"}

    hits = []
    for directory in ("scripts", "baselines"):
        for path in sorted((ART / directory).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            base = ART / CWD_RELATIVE_BASES.get(str(path.parent.relative_to(ART)), "")
            for lineno, literal in _module_level_path_literals(path):
                if literal.startswith(OUTPUT_PREFIXES):
                    continue
                if "{" in literal:  # format placeholder, not a literal path
                    continue
                if not (base / literal).exists():
                    hits.append(f"{path.relative_to(ART)}:{lineno} -> {literal!r}")
    assert not hits, "stale default path(s) -- fix the path or extend an exclusion:\n" + "\n".join(
        hits
    )


_USAGE_SCRIPT_RE = re.compile(r"uv run python ([^\s\\]+\.py)")


def test_usage_examples_name_scripts_that_exist():
    """A `uv run python <path>` example is the first thing a reader copy-pastes.

    Every GEPA and Trace2Skill baseline docstring named its own pre-relocation
    path (e.g. `scripts/dapo/gepa_tune_dapo_skill.py` for a script that now
    lives at `baselines/gepa/gepa_tune_dapo_skill.py`) and would fail with
    "No such file or directory" on the very first command a reader tries.
    """
    hits = []
    for directory in ("scripts", "baselines"):
        for path in sorted((ART / directory).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            for match in _USAGE_SCRIPT_RE.finditer(path.read_text()):
                script_path = match.group(1)
                if "..." in script_path:  # deliberate shorthand, not a literal path
                    continue
                if not (ART / script_path).exists():
                    hits.append(f"{path.relative_to(ART)} -> {script_path!r}")
    assert not hits, "Usage: example names a script that does not exist:\n" + "\n".join(hits)
