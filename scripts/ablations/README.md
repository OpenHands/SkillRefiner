# Ablations

The paper's five ablations (§4.1, Table 2). Each removes exactly one stage of
SkillRefiner, holding everything else at the configuration of record.

    uv run python scripts/ablations/run.py --ablate summary \
      --skill-file <base.md> --skill-name xlsx \
      --raw-jsonl <traces.jsonl> --binary-rewards <rewards.json> \
      --output-dir results/ablations/without_summary

Everything else stays at `RefineConfig`'s defaults: the ablation base config is a plain
`RefineConfig`.

`--binary-rewards` must point at a non-empty `{trace_id: bool}` JSON mapping — it is
what splits traces into the positive/negative partitions every ablation runs against.
A missing file, or one that resolves to an empty mapping, makes `refine()` raise
rather than silently proceeding on an unsplit corpus. This applies to `positive` and
`negative` too: each drops only the named partition and leaves the other untouched, so
both still need the split to exist before there is a surviving partition to keep.

| `--ablate` | Table 2 column | stage swapped |
|---|---|---|
| `summary` | w/o Summ. | `full_trace` -> `NoSummaryTraceSummarizer` (raw trace text) |
| `clustering` | w/o Clusters | `UmapHdbscanClusterer` -> `SingleClusterer` (one cluster per partition) |
| `positive` | w/o C+ | positive partition dropped before summarization |
| `negative` | w/o C− | negative partition dropped before summarization |
| `gating` | w/o Gating | evidence gate off; every failure-derived proposal reaches the merge |

Held constant: ground-truth injection, merge prompt, held-out eval harness and judges.
