# Running SkillRefiner

A start-to-finish path: install, verify, refine a skill, read the output, then
point it at your own data. `README.md` is the reference — layout, every config
key, per-benchmark status. This file is the order to do things in.

The whole pipeline is one function, `skill_refiner.pipeline.refine()`. Everything
below is either preparing its two inputs or reading its output.

---

## 1. Install

```
git clone https://github.com/OpenHands/SkillRefiner.git && cd SkillRefiner
uv sync
```

Dependencies are pinned with `==` and `uv.lock` is committed. Clustering and
token-budgeting are version-sensitive, so don't relax the pins — an unpinned
dependency can change results without raising an error.

Python version comes from `.python-version`. If `uv` is missing:
`curl -LsSf https://astral.sh/uv/install.sh | sh`.

---

## 2. Configure the three external services

SkillRefiner needs an LLM, an embedding endpoint, and (for benchmarks only) a
dataset. Nothing machine-specific is hardcoded; there is no default base URL.

**LLM API key** — either environment variable, checked in this order:

```
export LLM_API_KEY=...        # or OPENAI_API_KEY
```

**LLM base URL** — an OpenAI-compatible `/chat/completions` endpoint. Either:

```
export SKILL_REFINER_BASE_URL_EVAL_PROXY=https://your-llm-proxy.example.com
```

or copy the example config and fill it in (the real file is gitignored):

```
cp artifact.local.example.json artifact.local.json
# set llm_base_urls.eval_proxy
```

**Embedding endpoint** — the clustering stage needs one. The default expects a
local Ollama serving `qwen3-embedding:4b`:

```
ollama serve
ollama pull qwen3-embedding:4b
```

Override with `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL` if you host it elsewhere.
This is the prerequisite most likely to bite you: nothing else announces it
until a refine run is already minutes in and has spent money on summaries.

---

## 3. Verify before spending anything

```
make doctor
```

Checks Python version, dependency pins, API key, LLM base URL, `tmux`, and the
embedding endpoint — all at once, each failure printing its own fix. Exits
non-zero if anything is missing. Run this before every first run on a new
machine.

```
make check
```

ruff + the full test suite, entirely offline (every LLM call is stubbed). Should
be green on a clean checkout. Because it stubs the LLM, it cannot catch
dependency-level LLM breakage — for that:

```
uv run pytest -m live
```

One real completion. Skips rather than fails when credentials aren't configured.

---

## 4. First run — the bundled example

```
uv run python examples/minimal/run.py
```

About two minutes and ~20 LLM calls. It refines `examples/minimal/seed_skill.md`
against a bundled 12-trace corpus, so it needs no benchmark dataset and no
rollout. Output lands in `examples/minimal/output/` (gitignored) and the runner
prints the path of the refined skill.

If this works, your configuration is correct and everything below is a matter of
supplying different inputs.

What you should see — stage by stage, cluster counts varying between runs since
an LLM writes the summaries:

```
[positive] 6 summaries → clustering (umap_hdbscan) → 2 clusters
[negative] 6 summaries → clustering (umap_hdbscan) → 3 clusters (1 noise), all 3 kept by the verify gate
[combined] merging 2 positive + 2 negative cluster(s) over 11 traces
```

See `examples/minimal/README.md` for where those traces come from and how the
example differs from the measured configuration.

---

## 5. Read the output

`refine()` writes into `output_dir`:

```
output_dir/
├── combined/
│   ├── proposed_skill.md        ← the refined skill. this is the deliverable
│   └── manifest.json            what ran: cluster counts, model, confidence, rationale
├── positive/
│   ├── cluster_proposals.json   one proposed edit per surviving cluster
│   ├── manifest.json            partition settings and counts
│   └── summaries.json           per-trace summaries that fed clustering
└── negative/
    ├── cluster_proposals.json   edits that passed the evidence gate
    ├── rejected_cluster_proposals.json   ← dropped by the evidence gate
    ├── manifest.json
    └── summaries.json
```

Read these in this order when a result looks wrong:

1. `combined/manifest.json` — did both partitions actually contribute? A
   `n_positive_clusters` or `n_negative_clusters` of 0 means that whole polarity
   was skipped, and the "refined" skill is a one-sided edit.
2. `negative/rejected_cluster_proposals.json` — lessons the verify gate threw
   out. A long list here with a thin `cluster_proposals.json` means the gate is
   doing most of the work.
3. `*/summaries.json` — if the lessons are vague, the summaries usually are too,
   and the cause is upstream in the traces.

---

## 6. Run it on your own traces

Two inputs, both plain files. No SDK type is involved.

**`traces.jsonl`** — one JSON object per line. Spans with an empty `output_text`
are dropped:

```json
{"trace_id": "task-042", "spans": [{"span_id": "s1", "name": "agent_step", "input_text": "Fix the failing test", "output_text": "Ran pytest; test_foo failed on a missing import", "start_time": "2026-09-24T00:00:00Z", "end_time": "2026-09-24T00:00:04Z"}]}
```

**`binary_rewards.json`** — `{trace_id: bool}`, from your benchmark's scorer,
not from the harness. This is what splits the corpus into the positive and
negative partitions every stage works from. A missing or empty rewards file
raises rather than being silently ignored:

```json
{"task-042": true, "task-043": false}
```

Then call `refine()` directly — there is no CLI for the base configuration:

```python
import asyncio
from pathlib import Path
from skill_refiner.pipeline import RefineConfig, refine

config = RefineConfig(
    skill_file=Path("my_seed_skill.md"),
    skill_name="my-skill",
    raw_jsonl=Path("traces.jsonl"),
    binary_rewards=Path("binary_rewards.json"),
    output_dir=Path("results/my_refine_out"),
)
proposal = asyncio.run(refine(config))
print(config.output_dir / "combined" / "proposed_skill.md")
```

`RefineConfig`'s defaults *are* the configuration of record, pinned by
`tests/test_config_of_record.py`. You do not need to set the stage fields to
reproduce it; you set them to depart from it.

**Corpus size.** Each partition needs **≥ 3 summaries** or it is skipped with a
printed warning — so ≥ 3 passing *and* ≥ 3 failing traces, minimum, and that
floor produces clusters too small to be meaningful. The bundled example uses 6
a side. The paper's corpora were 200 traces (SpreadsheetBench), 400 (DAPO-Math)
and 639 (PR review).

If your harness can't emit that JSONL shape directly, copy the reference
converter, which reads native transcripts plus evaluator results and writes both
files: `scripts/benchmarks/spreadsheetbench/convert_traces_to_ablation_format.py`.

---

## 7. Optional: grader feedback

The paper attaches evaluator feedback to failing traces *before* summarization
(§2.2), so a failure's summarizer reads the evaluator's own
`expected 'X', got 'Y'` instead of inferring the failure from the transcript.

```
uv run python scripts/benchmarks/spreadsheetbench/build_grader_diff_ground_truth.py \
    --results results/spreadsheetbench/train/eval_official_results.json \
    --out results/spreadsheetbench/train/ground_truth_from_grader_diff.json
```

Then pass `ground_truth_file=Path(...)` to `RefineConfig`. The map is keyed to
one rollout's trace ids, so it cannot ship — build your own.

`refine()` runs fine without it, but omitting it is a departure from the record,
not a neutral simplification. `examples/minimal/` runs in that reduced mode
because it has no rollout to score.

---

## 8. Benchmarks, ablations, baselines

Which benchmarks run from this artifact alone, and which need an external
dataset, is the table in `README.md` § Benchmarks. Short version: **DAPO-Math
evaluation is self-contained** (it pulls `DAPO-Math-17k` from HuggingFace
itself). SpreadsheetBench needs the task set and rollout described in
`datasets/README.md`. PR review needs your own review rollouts on top of the
stored gold labels; the steps are in `datasets/README.md`.

```
# five one-stage ablations off the base configuration
uv run python scripts/ablations/run.py --ablate summary \
  --skill-file <seed.md> --skill-name <name> \
  --raw-jsonl <traces.jsonl> --binary-rewards <rewards.json> \
  --output-dir results/ablations/without_summary
```

`--ablate` takes `summary`, `clustering`, `positive`, `negative`, or `gating`
(the paper's Table 2 columns).
Details in `scripts/ablations/README.md`; the two comparison methods
(Trace2Skill, GEPA) are in `baselines/README.md`.

---

## Troubleshooting

| symptom | cause |
|---|---|
| `no base URL for llm_provider='eval_proxy'` | Step 2. No default is baked in — set the env var or `artifact.local.json`. |
| Run dies at clustering after paying for summaries | Embedding endpoint unreachable. `make doctor` catches this in advance; it is the most common and most expensive miss. |
| `[positive] Only N summaries — need ≥3, skipping` | Corpus too small on that polarity, or `binary_rewards.json` labels are lopsided. A skipped partition means a one-sided edit, not a failed run. |
| `refine()` raises on rewards | The file is missing, unreadable, or resolves to an empty mapping. Deliberate — an unsplit corpus would silently produce a meaningless result. |
| Cluster counts differ between identical runs | Expected. An LLM writes the summaries, so the embedding inputs differ run to run. |
| `make check` green but a real run fails | `make check` stubs every LLM call. Use `uv run pytest -m live`. |

---

## Reproducibility caveats

Four things in this artifact differ from the runs behind the paper's numbers,
all deliberate:

- **The merge guardrail block was removed.** The measured SpreadsheetBench run
  passed a 1280-char hand-written, SpreadsheetBench-specific block into the merge
  prompt, and the artifact shipped it as a *default* — so every benchmark got
  SpreadsheetBench instructions. It is gone: no config field, no parameter, no
  file. SpreadsheetBench results have **not** been re-measured without it. See
  `prompts/README.md`.
- **The cross-polarity merge prompt is now exactly the paper's (§A.5.4).** It
  used to insert a constraint block ("Synthesise these cluster proposals...",
  the `cluster_size >= 5` must-represent rule, the soften-at-source clause,
  PR-review placeholder names) between the merge framing and the JSON contract.
  The paper's merge prompt has no such block, so it was removed. Mirrored
  byte-for-byte in `prompts/merge.md`.
- **There is no per-partition synthesis step.** The research code also synthesized
  each partition into its own skill before the merge, and dropped a partition's
  proposals if that call failed. The paper's pipeline (§2) sends the cluster
  proposals straight to the merge, so this artifact does too: a default run makes
  one synthesis call, the merge.
- **The parametric-seed prompts changed wording** ("a Claude skill" → "an Agent
  skill"). The seeds in `datasets/` were generated with the old wording, so
  regenerating produces a different seed than the one the results were measured
  against.

The stored artifacts remain valid. The recipe no longer regenerates them byte
for byte.
