from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "baselines"


def test_both_baselines_ship():
    assert (BASE / "trace2skill" / "skill_evolver").is_dir()
    assert (BASE / "gepa" / "gepa_tune_skill.py").is_file()


def test_trace2skill_data_is_not_shipped():
    entries = [p.name for p in (BASE / "trace2skill" / "data").iterdir()]
    assert entries == [".gitkeep"], f"T2S data/ must be empty, found {len(entries)} entries"


def test_no_nested_git_repositories():
    assert not list(BASE.rglob(".git"))


def test_crustbench_baselines_are_not_shipped():
    assert not list(BASE.rglob("*crustbench*"))


def test_quick_validate_is_present_at_the_path_the_evolver_looks_up():
    """All three bootstrap pipelines sys.exit(1) at the evolve stage without it.

    `skill_evolving_agent.py:59` declares
    QUICK_VALIDATE_SCRIPT = Path("skills/skill-creator/scripts/quick_validate.py"),
    resolved against the evolver's CWD -- and all three
    baselines/trace2skill/*_bootstrap_pipeline.py launch it with
    `cwd=TRACE2SKILL_DIR`, none of them passing --dry-run. Both
    run_parallel_skill_evolution.py:659 and
    run_parallel_combined_skill_evolution.py:341 then exit 1 when it is missing,
    long before skill_evolving_agent.validate_skill()'s graceful-degradation
    branch is reachable. It is required, not optional.
    """
    t2s = BASE / "trace2skill"
    script = t2s / "skills" / "skill-creator" / "scripts" / "quick_validate.py"
    assert script.is_file(), f"Trace2Skill evolver's format checker missing: {script}"


def test_quick_validate_accepts_a_well_formed_skill_and_rejects_a_broken_one(tmp_path):
    import subprocess
    import sys

    script = BASE / "trace2skill" / "skills" / "skill-creator" / "scripts" / "quick_validate.py"

    good = tmp_path / "good"
    good.mkdir()
    (good / "SKILL.md").write_text("---\nname: xlsx\ndescription: d\n---\n\nbody\n")
    assert subprocess.run([sys.executable, str(script), str(good)]).returncode == 0

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "SKILL.md").write_text("# no frontmatter\n")
    assert subprocess.run(
        [sys.executable, str(script), str(bad)], capture_output=True
    ).returncode == 1
