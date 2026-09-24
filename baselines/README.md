# Baselines

The paper's two comparison methods (§3.1). Every method starts from the same initial
skill S0 and gets the same pool of N training examples.

## Trace2Skill (`trace2skill/`)

The closest prior offline method: it extracts lessons from individual trajectories
and consolidates them into the skill. Like SkillRefiner, it reads only the fixed trace
corpus. Bootstrap pipelines for SpreadsheetBench and DAPO-Math. The upstream code is
vendored under `trace2skill/`; see the note at the top of `trace2skill/README.md`.

> **Warning.** Trace2Skill mutates the seed skill **in place** at
> `<arm>/seed_skill/<name>/SKILL.md`, backing the original up to
> `seed_skill/<name>_backup_<timestamp>/` first. Any arm sequenced after it against the
> same directory will read the mutated file, not the seed. Stage each arm from a
> pristine copy of the seed and assert its checksum before running.

## GEPA (`gepa/`)

Reflective evolutionary search over the skill text. Unlike the offline methods, it
replays tasks to score each candidate. The paper runs it with a budget of five
candidate revisions (`gepa/GEPA_BUDGET.md` explains the budget arithmetic), and
splits the N examples evenly into GEPA's train and validation subsets. The validation
subset is separate from the held-out eval set. GEPA does not apply to PR review,
where candidates cannot be replayed. Takes `--parametric-seed-file` so it can start
from the same S0. Uses the pinned `gepa==0.1.4` package.

## Portability

These scripts are vendored from the authors' research code, verbatim apart from their
imports. A few classes of rewrite were needed to make them import and run
cleanly against this artifact's layout, all mechanical (no logic changed):

- `skill_lab.*` imports were rewritten to the `skill_refiner` package this artifact
  ships.
- Each of the six entry points that reaches `openhands`/`litellm` at import gained a
  four-line offline preamble, `__import__("skill_refiner")`, immediately before its
  import block. `litellm`'s module-level `get_model_cost_map()` makes a live HTTPS GET
  unless `LITELLM_LOCAL_MODEL_COST_MAP` is set, `skill_refiner/__init__.py` sets it, and
  a plain `import skill_refiner` would be sorted after the import it has to precede.
  No baseline logic is touched; `tests/test_no_network_on_import.py` sweeps all of them.
- Every bootstrap-pipeline entry point's `sys.path`/`REPO_ROOT`-relative imports were
  repointed at where the artifact actually keeps things: `_llm_config` (LLM
  connection glue — `_resolve_llm_config`, `_register_extra_models`) and
  `skill_refiner.pipeline._load_traces` replace the original research code's
  `run_summarizer_ablation` (not shipped, see below); `scripts/benchmarks/<bench>/`
  and `scripts/parametric_seed/<bench>.py` replace `scripts/<bench>/`;
  `baselines/trace2skill/` replaces `Trace2Skill/` at the repo root.
- Four `--llm-provider` defaults were changed from a provider alias naming a specific
  hosted deployment to `eval_proxy`, the only provider this artifact ships (see the
  repo README's "Local configuration"). No base URL was ever baked in, so the default
  selected a name, not an endpoint; the alias itself has been removed, so leaving the
  old default would have made these four scripts exit with "unknown --llm-provider".

`pyproject.toml`'s `[tool.ruff.lint.per-file-ignores]` silences 20 codes for
`baselines/**` so this vendored code isn't reformatted. One of them, `B905`
(`zip()` called without `strict=`), is not purely cosmetic like the rest —
a length mismatch between the two iterables it zips would silently truncate
instead of raising. It stays silenced on purpose: adding `strict=` would
change the runtime behavior of the exact code the paper's baseline numbers
were measured with, which the verbatim-vendoring policy above forbids. This
is a disclosed trade-off, not an oversight.

Every bootstrap-pipeline / GEPA entry point imports cleanly and reach their `argparse`
under this artifact's layout — verified by loading each one and confirming it either
runs to completion or exits via `--help`/argument parsing rather than `ImportError`.
None of them execute end-to-end without external inputs, same as the benchmark
drivers under `scripts/benchmarks/`:

| Script | Needs, beyond `--secrets-file`/`--llm-provider` |
|---|---|
| `trace2skill/trace2skill_bootstrap_pipeline.py` | A SpreadsheetBench dataset checkout (not shipped — see `trace2skill/data/` note below) |
| `trace2skill/dapo_trace2skill_bootstrap_pipeline.py` | DAPO-Math data (`scripts/benchmarks/dapo/prepare_dapo_math.py`); `--logs-dir` must point at a prior `run_dapo_agent.py` rollout of the *same* seed skill |
| `gepa/gepa_tune_skill.py`, `gepa/gepa_bootstrap_pipeline.py` | A SpreadsheetBench dataset checkout (not shipped) |
| `gepa/gepa_tune_dapo_skill.py`, `gepa/gepa_bootstrap_dapo.py` | DAPO-Math data |

The SpreadsheetBench-verified dataset that `trace2skill/data/` would otherwise hold
(192MB, 6666 files) is deliberately not shipped (see `trace2skill/data/.gitkeep`).
`datasets/README.md` shows how to link the 400-task copy into place. There is no `--data-dir` flag on any of the three
SpreadsheetBench scripts; each reaches the dataset differently, so point it at your own
checkout accordingly:

| Script | How it finds the dataset |
|---|---|
| `gepa/gepa_tune_skill.py` | `--data-path`, defaulting to `baselines/trace2skill/data/spreadsheetbench_verified/spreadsheetbench_verified_400` |
| `gepa/gepa_bootstrap_pipeline.py` | No dataset flag of its own; it forwards every unrecognized flag verbatim to `gepa_tune_skill.py`, so pass `--data-path` and it arrives there |
| `trace2skill/trace2skill_bootstrap_pipeline.py` | No dataset flag, and none is forwarded: it shells out to `scripts/benchmarks/spreadsheetbench/spreadsheetbench_agent_runner.py` without one, so that runner's own `--data-path` default (the same path as above) is what is used. Put the checkout there, or edit `DEFAULT_DATA_PATH` in the runner |

