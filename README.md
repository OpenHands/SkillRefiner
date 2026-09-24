# SkillRefiner

Code for *SkillRefiner: Offline Skill Refinement from Historical Agent Traces*.
It refines a deployed agent skill **offline**, using only historical execution
traces and their observed outcomes. It never generates new rollouts.

SkillRefiner turns a corpus of labeled agent execution traces into an edited skill
file. It runs in the paper's five stages (§2):

1. **Feedback-augmented summarization.** Each trace is summarized with a
   polarity-specific prompt. A success reconstructs the behavioral recipe. A
   failure diagnoses the failure chain, with the evaluator's feedback attached
   (`ground_truth_file`) before summarization.
2. **Embedding and clustering.** Summaries are embedded (`qwen3-embedding:4b`)
   and clustered with UMAP + HDBSCAN, separately for the success and failure
   partitions. Noise and singleton points do not produce an edit.
3. **Proposal generation.** One candidate edit per cluster. Positive clusters
   reinforce the skill; negative clusters soften it or add guardrails.
4. **Evidence gating.** A separate LLM call keeps a failure-derived proposal only
   if the cluster's traces support it.
5. **Skill merging.** Positive proposals and gated negative proposals are merged
   into the initial skill (prompt: paper §A.5.4, mirrored in `prompts/merge.md`).

The paper evaluates on SpreadsheetBench, DAPO-Math and production pull-request
review, with GPT-5.4-mini and MiniMax-M2.7 as agent models (§3). Refinement uses
only a fixed corpus of past traces, and the refined skill is scored on a disjoint
held-out set.

`skill_refiner/pipeline.py::refine()` is the single entry point; the defaults on
its `RefineConfig` dataclass are the configuration of record described below.

## Quickstart

```
uv sync                                  # install (pinned; see "Install" below)
make doctor                              # check every prerequisite at once
uv run python examples/minimal/run.py    # refine a real skill end to end, ~2 min
```

`make doctor` checks Python version, dependency pins, API key, LLM base URL,
`tmux` (used by the OpenHands terminal tool), and the **embedding endpoint** — a local Ollama serving
`qwen3-embedding:4b` by default, the one prerequisite nothing else announces
until a refine run is minutes in. Each failure prints its fix; missing
requirements exit non-zero.

The example needs no benchmark dataset: it bundles a 12-trace corpus derived
from the captured traces in `tests/fixtures/golden/` (see
`examples/minimal/README.md`).

**New here? [`RUNNING.md`](RUNNING.md) is the start-to-finish runbook** — install,
configure the three external services, first run, how to read the output
directory, and how to point the pipeline at your own traces. This README is the
reference companion to it.

## Layout

```
SkillRefiner/
├── README.md                 this file (reference)
├── RUNNING.md                start-to-finish runbook
├── pyproject.toml            deps trimmed; no [project.scripts]
├── uv.lock  Makefile  LICENSE  .python-version  .gitignore
│
├── skill_refiner/
│   ├── trace/                store.py serialization.py context.py
│   │                         event_normalization.py
│   ├── summarize/            core.py llm.py no_summary.py runner.py
│   ├── cluster/               umap_hdbscan.py single.py
│   ├── propose/               core.py proposal.py protocols.py clustered.py
│   │                          polarity.py budget.py
│   ├── merge/                synthesize.py
│   └── pipeline.py           end-to-end refine loop
│
├── prompts/                  every prompt the method sends, mirrored and drift-tested
├── examples/minimal/         bundled 12-trace corpus + one-command runner
├── scripts/
│   ├── doctor.py             preflight check (`make doctor`)
│   ├── parametric_seed/      spreadsheetbench dapo
│   ├── benchmarks/           spreadsheetbench dapo pr_review
│   └── ablations/            one runner, --ablate <stage>
├── baselines/
│   ├── trace2skill/
│   └── gepa/
├── tests/
├── datasets/                 refinement corpora + initial skills; gitignored except its README
└── results/                  empty + .gitkeep
```

## Install

```
uv sync
```

Every dependency in `pyproject.toml` is pinned with `==`, not bounded, and the
committed `uv.lock` resolves to exactly those versions — clustering and
prompt token-budgeting are both version-sensitive, so an unpinned dependency
can change output without raising an error.

### Tests

```
make doctor           # prerequisites, all reported at once; non-zero if any are missing
make check            # ruff + pytest; the default suite, fully offline
uv run pytest -m live   # opt-in: one real completion, needs an API key + base URL
```

`make check` stubs every LLM call, so it cannot catch dependency-level LLM
breakage; `-m live` exists for that and skips (rather than fails) without
credentials configured.

## Local configuration

Nothing machine-specific is hardcoded in tracked code. API keys, deployment
URLs, dataset roots and the embedding cache directory are each an environment
variable, a CLI flag, a gitignored `*.secrets.json` file, or (for LLM base
URLs) the gitignored `artifact.local.json` — copy
`artifact.local.example.json` to `artifact.local.json` to set it up. There is
no baked-in default base URL; if nothing resolves one, the run fails loudly
and names the missing provider.

| key | how it's actually wired |
|---|---|
| `embedding_cache_dir` | not configurable via env var; defaults to `<repo_root>/.cache/skill_refiner/summary_clustering` (gitignored). Override by passing `cache_dir=` explicitly. |
| `embedding_base_url`, `embedding_model` | env vars `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL` |
| `embedding_api_key_env_var` | env var `EMBEDDING_API_KEY` |
| `llm_base_urls` | read from `artifact.local.json` as `{"llm_base_urls": {"eval_proxy": ...}}`, falling back from `--base-url` / `SKILL_REFINER_BASE_URL_EVAL_PROXY` (`skill_refiner/pipeline.py`) / `LLM_BASE_URL` (`scripts/_llm_config.py`). `eval_proxy` is the only provider shipped. |
| `llm_api_key_env_var` | env vars `LLM_API_KEY` or `OPENAI_API_KEY` (checked in that order), or `--secrets-file` |
| `llm_secrets_file` | `--secrets-file` CLI flag; any `*.secrets.json` file you create is already gitignored |
| `openai_base_url`, `openai_api_key_env_var` | env vars `OPENAI_BASE_URL`, `OPENAI_API_KEY` (Trace2Skill baseline, under `baselines/trace2skill/`) |
| `spreadsheetbench_data_path` | `--data-path` CLI flag on `spreadsheetbench_agent_runner.py`; default points at a repo-relative path that only exists after you fetch the official SpreadsheetBench dataset yourself |

## Benchmarks

The paper evaluates three settings: SpreadsheetBench, DAPO-Math and PR review. Not all
are runnable from this artifact alone.

| benchmark | status |
|---|---|
| **SpreadsheetBench** | Needs the 400-task set (200 train / 200 eval) and its rollout, which are not in git. See `datasets/README.md` for the expected layout. |
| **DAPO-Math** | Evaluation is self-contained: `scripts/benchmarks/dapo/prepare_dapo_math.py` pulls `DAPO-Math-17k` from HuggingFace. Refinement needs a train rollout (400 problems in the paper). |
| **PR review** | The training gold labels are in `datasets/pr_review/` (not in git). You supply the review rollouts and eval PRs; `build_parametric_signal.py` turns them into the training signal. Steps in `datasets/README.md`. |

`datasets/README.md` lists the splits and initial skills (S0) the paper used.

`baselines/README.md` has the equivalent table for Trace2Skill and GEPA.

## Reproducing the configuration of record

`skill_refiner.pipeline.refine(RefineConfig(...))` is the single call; its
defaults already are the configuration of record (pinned by
`tests/test_config_of_record.py`):

```python
summarizer                 = "full_trace"
cluster_method              = "umap_hdbscan"
min_cluster_size            = 2                # not the HDBSCAN default of 3
verify_negative_clusters    = True
use_positive = use_negative = True
model                       = "openai/gpt-5.4-mini"
llm_provider                = "eval_proxy"
embedding_model             = "qwen3-embedding:4b"
embedding_base_url          = "http://localhost:11434/v1"
```

```python
import asyncio
from pathlib import Path
from skill_refiner.pipeline import RefineConfig, refine

config = RefineConfig(
    skill_file=Path("datasets/spreadsheetbench/seed_skills/parametric_seed.md"),
    skill_name="xlsx",
    raw_jsonl=Path("results/spreadsheetbench/train/traces.jsonl"),
    binary_rewards=Path("results/spreadsheetbench/train/binary_rewards.json"),
    # Evaluator feedback for failed traces -- see "Grader feedback" below.
    ground_truth_file=Path("results/spreadsheetbench/train/ground_truth_from_grader_diff.json"),
    output_dir=Path("results/spreadsheetbench/refine_out"),
    limit=200,
)
proposal = asyncio.run(refine(config))
```

`raw_jsonl` / `binary_rewards` for SpreadsheetBench come from running
`scripts/benchmarks/spreadsheetbench/spreadsheetbench_agent_runner.py --split train`
against your own dataset checkout, then
`scripts/benchmarks/spreadsheetbench/convert_traces_to_ablation_format.py` on
its output. Score the resulting skill with
`scripts/benchmarks/spreadsheetbench/eval_skill_on_spreadsheetbench.py` (once
with `--skill-file` pointed at the proposal, once without it for baseline) and
diff with `compare_results.py`.

There is no packaged CLI for the plain (non-ablated) configuration — only
`scripts/ablations/run.py --ablate <stage>`, which starts from this same base
and swaps one field away from it. To run the base configuration itself, call
`refine()` directly as above.

### Grader feedback (`ground_truth_file`)

This is the paper's feedback-augmented trajectory representation (§2.2): `refine()` appends each entry of `ground_truth_file` to its
trace as a SCORING REFERENCE event **before** summarization, so a failing run's
summarizer reads the evaluator's own `expected 'X', got 'Y'` cell diff instead
of inferring the failure from the transcript. The map is keyed to one rollout's
trace ids, so it is not shipped — build your own from that rollout's evaluator
output:

```
uv run python scripts/benchmarks/spreadsheetbench/build_grader_diff_ground_truth.py \
    --results results/spreadsheetbench/train/eval_official_results.json \
    --out results/spreadsheetbench/train/ground_truth_from_grader_diff.json
```

Without it, failed traces are summarized from the transcript alone, with no
evaluator feedback attached. `examples/minimal/` runs this way because it has no
rollout to score.

The merge always applies the soften-at-source conflict rule from the paper's
merge prompt. It is not a flag (`merge/synthesize.py`,
`tests/test_merge_soften_default.py`).

## Ablations

The paper's five ablations (Table 2), each a one-stage swap off the
configuration of record. See `scripts/ablations/README.md`. `scripts/ablations/run.py`'s base config is a
plain `RefineConfig`, so the ablation arms and the base configuration agree by
construction.

## Baselines

The paper's two comparison methods (§3.1): Trace2Skill, the closest offline
method, and GEPA, which replays tasks under each candidate skill. GEPA was run
with a budget of five candidate revisions and an even train/validation split of
the same N examples, and was not applicable to PR review. See
`baselines/README.md` for what each entry point needs. GEPA lives at
`baselines/gepa/` and uses the pinned `gepa==0.1.4` package.

## Using your own agent harness

Nothing about SkillRefiner is tied to the OpenHands SDK. Its input contract is
plain JSONL, not an SDK type: `_jsonl_trace_to_trace` in
`skill_refiner/pipeline.py` reads each line as

```json
{"trace_id": "...", "spans": [{"span_id": "...", "name": "...", "input_text": "...", "output_text": "...", "start_time": "...", "end_time": "..."}]}
```

keeps whichever spans have a non-empty `output_text`, and drops the rest. Any
harness that can be made to emit that shape works.

Whatever harness you use, a second input is required regardless: a
`binary_rewards.json` mapping `{trace_id: bool}`, which `refine()` uses to
split the corpus into the positive and negative partitions each stage works
from. That file comes from the benchmark's own scorer, not from the harness.
A missing or empty rewards file isn't silently ignored — `refine()` raises.

Produce `traces.jsonl` and `binary_rewards.json` from your own harness's
transcripts; no SDK is involved in the trace path.

The fastest correct start is to copy the SpreadsheetBench reference converter,
`scripts/benchmarks/spreadsheetbench/convert_traces_to_ablation_format.py`. It
reads the benchmark's native transcripts plus the evaluator's pass/fail results
and writes out exactly `traces.jsonl` + `binary_rewards.json`.

One `traces.jsonl` line for a two-span trace, plus its matching rewards
entry, looks like this:

```json
{"trace_id": "task-042", "spans": [{"span_id": "s1", "name": "agent_step", "input_text": "Fix the failing test", "output_text": "Ran pytest; test_foo failed on a missing import", "start_time": "2026-09-24T00:00:00Z", "end_time": "2026-09-24T00:00:04Z"}, {"span_id": "s2", "name": "agent_step", "input_text": "Add the missing import and rerun", "output_text": "Added `import os`; pytest now passes", "start_time": "2026-09-24T00:00:05Z", "end_time": "2026-09-24T00:00:09Z"}]}
```

```json
{"task-042": true}
```
