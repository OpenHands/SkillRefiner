# Runtime skills for the Trace2Skill spreadsheet agent

| directory | tracked in git | what it is |
|---|---|---|
| `xlsx/` | **no** (gitignored) | Anthropic's `xlsx` skill: the human-written initial skill S0 for SpreadsheetBench |
| `xlsx-35B/` | yes | Trace2Skill's released skill evolved from 35B error traces |
| `xlsx-122B/` | yes | Trace2Skill's released skill evolved from 122B error traces |

## Why `xlsx/` is not in the repository

`xlsx/` holds Anthropic's proprietary `xlsx` skill (`SKILL.md`, `recalc.py`,
`LICENSE.txt`). Its `LICENSE.txt` reads "© 2025 Anthropic, PBC. All rights reserved"
and forbids reproducing or copying the materials, so this repository does not
redistribute it. The directory is listed in the top-level `.gitignore`.

The code still expects it at this path:
- `scripts/benchmarks/spreadsheetbench/eval_skill_on_spreadsheetbench.py` (`CANONICAL_XLSX_SKILL_DIR`)
- `baselines/trace2skill/trace2skill_bootstrap_pipeline.py` (`CANONICAL_XLSX_SKILL_DIR`)
- `baselines/trace2skill/run_spreadsheetbench.py` (`ALLOWED_SKILL_DIR_NAMES`)

To restore it, copy `spreadsheet_agent/skills/xlsx/` from a clone of the Trace2Skill
repository (`Qwen-Applications/Trace2Skill`) into this directory. The file the paper's
human-S0 SpreadsheetBench runs used is that repo's `SKILL.md`, md5 `c224a3c4…`
(288 lines). It is Anthropic's stock `xlsx` skill with one token changed ("When
Claude needs…" → "When Qwen-Agent needs…").
