# GEPA: budget and candidate count

How GEPA's metric-call budget turns into a number of candidate skills, how to set it for
the paper's comparison, and what it costs. GEPA's cost is agent rollouts, so the budget
is the main thing that sets its position on the paper's performance–cost plot (Figure 2).

## The paper's GEPA setup (§3.1)

- **Initial skill.** The same S0 as every other method (see `datasets/README.md`).
- **Data.** The same pool of N training examples, split **evenly** into GEPA's train
  (reflection) subset and its validation (candidate selection) subset. The validation
  subset is separate from the held-out eval set.
- **Budget.** Five candidate revisions.
- **PR review.** Not run. Candidate skills there cannot be replayed with an automated
  score.

The entry points that take a separate S0 file:

| setting | entry point | S0 flag |
|---|---|---|
| SpreadsheetBench, LLM S0 | `gepa_bootstrap_pipeline.py` | `--parametric-seed-file datasets/spreadsheetbench/seed_skills/parametric_seed.md` |
| SpreadsheetBench, human S0 | `gepa_tune_skill.py` | `--seed-skill` (defaults to the gitignored `xlsx` skill; see `../trace2skill/spreadsheet_agent/skills/README.md`) |
| DAPO-Math, LLM S0 | `gepa_bootstrap_dapo.py` | `--parametric-seed-file datasets/dapo/seed_skills/parametric_seed.md` |

Both bootstrap wrappers forward every other flag to the tuning script underneath. Set
the split and budget explicitly; the tuning scripts' defaults are smoke-test sized
(`gepa_tune_dapo_skill.py`: `--val-size 20 --max-metric-calls 80`;
`gepa_tune_skill.py`: `--max-metric-calls 60`).

- `--val-size N/2` for the even split: 100 for SpreadsheetBench (N=200), 200 for
  DAPO-Math (N=400).
- `--max-metric-calls ≈ 7 × val-size` for about five candidates, using the rule below.

These two values come from the budget rule, not from the logs of the paper's runs.
Check `num_candidates` in the run's output to confirm how many candidates you got.

## Budget → candidate relationship

A "candidate" is one proposed skill version (candidate 0 is the seed). Each candidate
costs about one full pass over the validation set, so:

```
num_candidates ≈ max_metric_calls / val_size − overhead (≈1–2)
```

A ratio of 7 gives about five candidates. A ratio of 2 gives only 2–4.

GEPA can end up keeping the seed when no candidate beats it on validation (in the run
output, `best_idx=0` and `changed_from_seed=False`). The paper reports this in two of
its settings (Appendix A.3).

## Resuming and extending a run

GEPA resumes from `<output-dir>/gepa_state/` automatically; only `--fresh` wipes it.
To add candidates to a finished run instead of restarting:

1. Rerun with the same `--output-dir`.
2. Raise `--max-metric-calls` **above what was already spent**. A run that already used
   about 7 × val gains nothing from a budget of 7 × val.
3. Do not pass `--fresh`.

GEPA reloads the existing candidates, the Pareto front and the cached validation
scores, so you only pay rollouts for the new candidates.

## Reasoning models: `<think>` leakage (fixed)

Reasoning models used as the reflector (MiniMax-M2.7, for example) emit
`<think>…</think>`. GEPA stored that raw output as the candidate, which leaked chain of
thought into `proposed_skill.md` and from there into the agent's system prompt.
`_strip_think()` in `gepa_tune_dapo_skill.py` now strips it, both in the reflection LM
wrapper and where the skill is written. Do not resume a `gepa_state/` created before this
fix: its corrupted candidates stay in the selection pool. Re-tune with `--fresh` instead.

## Cost anchors (GPT-5.4-mini, DAPO-Math, measured)

- **About $0.04 per paid rollout** (about 26k tokens). Cost tracks *paid* rollouts
  (`n_agent_rollouts`), not the nominal `metric_calls`. Cached re-evaluations are free.
- **Held-out eval (200 problems):** about $5 per skill.
- **Reflection:** negligible, about $0.02–0.09 per run.

The paper's token totals for GEPA and the other methods are in Appendix A.4 (Table 6).

## Caveat: candidates vs selection

More candidates widen the search, but selection still happens on the validation set.
A small validation set makes the choice noisy, so keep `--val-size` large (the paper's
even split gives 100 for SpreadsheetBench and 200 for DAPO-Math).
