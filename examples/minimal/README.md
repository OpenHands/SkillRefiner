# Minimal end-to-end example

A 12-trace corpus bundled so `refine()` can be run with nothing but an API key
and an embedding endpoint — no SpreadsheetBench download, no rollout.

```
make doctor                              # confirm the prerequisites first
uv run python examples/minimal/run.py    # ~2 min, ~20 LLM calls
```

The runner prints the path of the refined skill; it lands at
`examples/minimal/output/combined/proposed_skill.md` (the whole `output/`
directory is gitignored).

## What's here

| file | |
|---|---|
| `traces.jsonl` | 12 traces in the pipeline's input shape, ~75 KB |
| `binary_rewards.json` | `{trace_id: bool}` — 6 `true`, 6 `false` |
| `seed_skill.md` | the skill to refine: a short `xlsx` skill with YAML frontmatter |
| `run.py` | the one-command runner |
| `build_corpus.py` | regenerates the two corpus files from the golden fixtures |

## Where the traces come from

They are **real captured agent traces**, not synthetic text: the bodies are
lifted from `tests/fixtures/golden/{positive,negative}/*/raw_content.txt`, the
24 already-anonymized SpreadsheetBench rollouts that
`tests/test_golden_prompts.py` pins the summarizer prompt against. Two
reductions were applied, both reproducible by rerunning `build_corpus.py`:

* **Selection** — the 6 smallest fixtures of each polarity, by file size.
  Six a side is not arbitrary: `_run_partition` skips any partition with fewer
  than 3 summaries, and a 3-trace partition gives HDBSCAN nothing to separate.
* **Truncation** — each trace keeps its first 3000 and last 3000 characters
  with an explicit `[middle of trace elided ...]` marker between them (the
  originals run to 154 KB). Head *and* tail, because the head holds the task
  prompt and the tail holds the outcome — truncating only the head would hand
  the negative summarizer a failure trace with no visible failure.

Each trace is one JSONL span, so `serialize_trace` reproduces the fixture text
verbatim: `_clean_span_output` only rewrites spans whose output parses as JSON,
and these do not.

## What a run looks like

Stage output is non-deterministic (an LLM writes the summaries and the
lessons), so cluster counts move between runs. One observed run:

```
[positive] 6 summaries → clustering (umap_hdbscan) → 2 clusters
[negative] 6 summaries → clustering (umap_hdbscan) → 3 clusters (1 noise), all 3 kept by the verify gate
[combined] merging 2 positive + 2 negative cluster(s) over 11 traces
```

It produced a skill that kept the seed's frontmatter and workflow and added
guardrails the negative traces actually earned — forcing a recalculation before
verifying formula cells, and not normalizing labels or shifting data away from
the requested answer range.

## Differences from the configuration of record

Two, both consequences of the corpus being small and shipped:

* **No `ground_truth_file`.** The record injects grader diagnostics into
  negative traces before summarization. That map is keyed to a specific rollout,
  so it cannot ship — see the main README's reproduction section and
  `scripts/benchmarks/spreadsheetbench/build_grader_diff_ground_truth.py`.
* **12 traces, truncated.** The paper's SpreadsheetBench corpus is 200 full-length
  traces. This example shows the full pipeline running on a small corpus.

Everything else — summarizer, clustering method, `min_cluster_size=2`, the
evidence-verify gate, both polarities, the model — is left at its
`RefineConfig` default, which is the record.
