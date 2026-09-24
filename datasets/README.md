# Datasets

> **The data files are not included; only this README is.** It documents the layout
> the code expects, so you can place a copy of the data at `datasets/` (or point
> `RefineConfig` at wherever you keep it).

This directory holds the refinement corpora and initial skills for the three settings
in the paper (§3.2). Code is not stored here. Everything that reads this data lives in
`skill_refiner/`, `scripts/benchmarks/` and `baselines/`.

For prerequisites (LLM key and base URL, the local `qwen3-embedding:4b` endpoint), see
[`RUNNING.md`](../RUNNING.md).

## Splits used in the paper

| setting | refinement (train) | held-out eval | outcome signal |
|---|---|---|---|
| SpreadsheetBench | 200 tasks | 200 tasks | official evaluator. Failed traces get the cell-level diff. |
| DAPO-Math | 400 problems | 200 problems | rule-based answer match |
| PR review | 639 PRs (763 reflected suggestions) | 145 PRs (121 reflected suggestions) | developer revisions before merge, judged by GPT-5.4-mini |

SpreadsheetBench uses the same 400-task subset as Trace2Skill. Task ids `0–199` are
train and `200–399` are eval.

## Expected layout

```
datasets/
├── README.md                                   (tracked; this file)
├── MANIFEST.json                               sha256 of every file below
├── spreadsheetbench/
│   ├── rollout/
│   │   ├── traces.jsonl                        200 GPT-5.4-mini train traces
│   │   ├── binary_rewards.json                 {trace_id: bool}, 160 pass / 40 fail
│   │   └── ground_truth_from_grader_diff.json  grader diffs for the 40 failures
│   ├── seed_skills/
│   │   └── parametric_seed.md                  LLM-generated initial skill
│   └── spreadsheetbench_verified_400/
│       ├── dataset.json                        400 tasks
│       └── spreadsheet/<id>/                   1_<id>_init.xlsx, 1_<id>_golden.xlsx, prompt.txt
├── dapo/
│   ├── rollout/
│   │   ├── traces.jsonl                        400 GPT-5.4-mini train traces
│   │   └── binary_rewards.json                 320 pass / 80 fail
│   └── seed_skills/
│       ├── parametric_seed.md                  LLM-generated initial skill (paper's DAPO S0)
│       └── human_seed.dapo-math-base.md
└── pr_review/
    ├── seed_skills/
    │   ├── human_seed.code-review.md           deployed human-written skill (paper's PR S0)
    │   └── parametric_seed.md
    └── train639_gold/
        ├── cases/<case_id>/gold_reflected_issues.json
        ├── composition.json
        ├── train639_caselist.json
        └── train639_trace_manifest.json
```

## Initial skills (S0)

All refinement methods start from the same S0 (paper §3.2, appendices A.6–A.7).

| setting | S0 source | where |
|---|---|---|
| SpreadsheetBench | LLM-generated | `spreadsheetbench/seed_skills/parametric_seed.md` |
| SpreadsheetBench | human-written | `baselines/trace2skill/spreadsheet_agent/skills/xlsx/SKILL.md`. Gitignored, because its licence forbids redistribution. See [that folder's README](../baselines/trace2skill/spreadsheet_agent/skills/README.md). |
| DAPO-Math | LLM-generated (no human skill existed) | `dapo/seed_skills/parametric_seed.md` |
| PR review | human-written (the deployed skill) | `pr_review/seed_skills/human_seed.code-review.md` |

## SpreadsheetBench

Refine:

```python
import asyncio
from pathlib import Path
from skill_refiner.pipeline import RefineConfig, refine

DATA = Path("datasets")
proposal = asyncio.run(refine(RefineConfig(
    skill_file        = DATA / "spreadsheetbench/seed_skills/parametric_seed.md",
    skill_name        = "xlsx",
    raw_jsonl         = DATA / "spreadsheetbench/rollout/traces.jsonl",
    binary_rewards    = DATA / "spreadsheetbench/rollout/binary_rewards.json",
    ground_truth_file = DATA / "spreadsheetbench/rollout/ground_truth_from_grader_diff.json",
    output_dir        = Path("results/spreadsheetbench/refine_out"),
    limit             = 200,
)))
```

`ground_truth_file` is the evaluator feedback the paper attaches to failed traces before
summarization (§2.2).

Evaluate: `spreadsheetbench_agent_runner.py` defaults to
`baselines/trace2skill/data/spreadsheetbench_verified/spreadsheetbench_verified_400`.
Link the task set there once:

```bash
mkdir -p baselines/trace2skill/data/spreadsheetbench_verified
ln -s "$PWD/datasets/spreadsheetbench/spreadsheetbench_verified_400" \
      baselines/trace2skill/data/spreadsheetbench_verified/spreadsheetbench_verified_400

uv run python scripts/benchmarks/spreadsheetbench/eval_skill_on_spreadsheetbench.py \
    --skill-file results/spreadsheetbench/refine_out/combined/proposed_skill.md \
    --output-dir results/spreadsheetbench/eval_refined
uv run python scripts/benchmarks/spreadsheetbench/eval_skill_on_spreadsheetbench.py \
    --output-dir results/spreadsheetbench/eval_baseline          # no --skill-file
uv run python scripts/benchmarks/spreadsheetbench/compare_results.py \
    results/spreadsheetbench/eval_baseline results/spreadsheetbench/eval_refined
```

The task set is third-party data (SpreadsheetBench,
[arXiv:2406.14991](https://arxiv.org/abs/2406.14991)), taken from the Trace2Skill
release. It has no licence file of its own, so check the upstream terms before sharing it.

## DAPO-Math

```python
RefineConfig(
    skill_file     = DATA / "dapo/seed_skills/parametric_seed.md",
    skill_name     = "dapo-math",
    raw_jsonl      = DATA / "dapo/rollout/traces.jsonl",
    binary_rewards = DATA / "dapo/rollout/binary_rewards.json",
    output_dir     = Path("results/dapo/refine_out"),
    limit          = 400,
)
```

No `ground_truth_file` is stored for DAPO. For the paper's DAPO feedback (the agent's
answer against the reference answer, §3.2), pass the reference-answer map that
`scripts/benchmarks/dapo/prepare_dapo_math.py` writes to
`results/dapo_math_17k/ablation/train/answers.json`. It is keyed by the same
`dapo-math-…` trace ids as the rollout, and the agent's own answer is already in each
trace. To evaluate on the paper's 200 held-out problems (this pulls `DAPO-Math-17k`
from HuggingFace):

```bash
uv run python scripts/benchmarks/dapo/prepare_dapo_math.py
uv run python scripts/benchmarks/dapo/run_dapo_agent.py \
    --split eval --skill-file results/dapo/refine_out/combined/proposed_skill.md
```

## PR review

The PR-review signal is delayed: a review suggestion counts as useful when the
developer's later revisions before merge address the same issue (§3.2). Running the
setting takes three inputs besides the gold labels stored here:

| input | what it is |
|---|---|
| review rollouts | your review agent's run on each training PR, with its suggestions (`reflected_suggestions.jsonl`, one row per suggestion carrying a `reflection_label`) |
| miss list | gold issues each review did not raise (`miss_list.json`, `trace_id -> [{body, path, line}]`) |
| eval PRs | the 145 held-out PRs, for replaying a skill and scoring it |

`train639_gold/` holds the gold labels for the training PRs.

**1. Build the training signal.** This writes the trace corpus, the per-trace rewards,
and per-trace feedback listing which suggestions were confirmed and which issues were
missed. A PR counts as a success when most of its scored suggestions were reflected
(`trace_passes`).

```bash
uv run python scripts/benchmarks/pr_review/build_parametric_signal.py \
    --manifest datasets/pr_review/train639_gold/train639_trace_manifest.json \
    --reflected-suggestions <reflected_suggestions.jsonl> \
    --miss-list <miss_list.json> \
    --output-dir results/pr_review/signal
```

`--manifest` is optional. When given, the run aborts if its trace count does not match
the manifest's case count.

**2. Refine** from the deployed human-written skill, with the feedback as `ground_truth_file`:

```python
RefineConfig(
    skill_file        = DATA / "pr_review/seed_skills/human_seed.code-review.md",
    skill_name        = "code-review",
    raw_jsonl         = Path("results/pr_review/signal/suggestion_learning_traces.jsonl"),
    binary_rewards    = Path("results/pr_review/signal/binary_rewards.json"),
    ground_truth_file = Path("results/pr_review/signal/reflection_feedback.json"),
    output_dir        = Path("results/pr_review/refine_out"),
)
```

**3. Evaluate** by replaying the refined skill on the eval PRs and matching its
suggestions against the issues the developers addressed. The paper uses GPT-5.4-mini as
the matching judge and reports F1.

## Verifying a copy

`MANIFEST.json` records the size and sha256 of every file. From inside `datasets/`:

```bash
python3 - <<'PY'
import hashlib, json, pathlib
m = json.load(open("MANIFEST.json")); bad = 0
for f in m["files"]:
    p = pathlib.Path(f["path"]); h = hashlib.sha256()
    if not p.is_file(): print("MISSING", p); bad += 1; continue
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""): h.update(c)
    if h.hexdigest() != f["sha256"]: print("CORRUPT", p); bad += 1
print(f"{len(m['files'])} files checked, {bad} problem(s)")
PY
```

`MANIFEST.json` also lists `README.md`, so this README will now show as `CORRUPT`
against an older manifest. That one mismatch is expected.
