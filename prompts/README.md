# Prompts

Every prompt SkillRefiner sends, one file each. These files are a **mirror** of the
strings the code builds at runtime. The code does not read them.
`tests/test_prompt_drift.py` keeps them honest: it fails if a file differs by even one
byte from what its builder produces.

Placeholder values (`xlsx` as the skill name, `SKILL` / `SUMMARIES` standing in for the
current skill and the packed summary block, a one-cluster example) were chosen for
readability. They do not change the prompt's structure, which is what the drift test
checks.

## Prompts in the paper (Appendix A.5)

| file | paper | built by |
|---|---|---|
| `summarize_positive.md` | A.5.1, polarity=positive | `skill_refiner/summarize/llm.py::_build_prompt` |
| `summarize_negative.md` | A.5.1, polarity=negative | `skill_refiner/summarize/llm.py::_build_prompt` |
| `skill_run_header.md` | the `{header}` line in A.5.1 | `skill_refiner/summarize/llm.py::_skill_run_header` |
| `positive_cluster.md` | A.5.2, polarity=positive | `skill_refiner/propose/polarity.py::_positive_cluster_prompt` |
| `negative_cluster.md` | A.5.2, polarity=negative | `skill_refiner/propose/polarity.py::_negative_cluster_prompt` |
| `evidence_verify_gate.md` | A.5.3 | `skill_refiner/propose/polarity.py::_verify_prompt` |
| `merge.md` | A.5.4, the full merge prompt | `skill_refiner/merge/synthesize.py::_merge_prompt` |
| `merge_framing.md` | A.5.4, the framing paragraph inside `merge.md` | `skill_refiner/merge/synthesize.py::_MERGE_FRAMING_SAS` |

`merge.md` is the prompt that produces the refined skill. It contains the current
skill, the clusters, `_MERGE_FRAMING_SAS`, and the JSON contract, and nothing else.
That framing paragraph carries the paper's conflict rule (§2.6): when a failure cluster
limits a behavior that a success cluster endorses, the merge qualifies that behavior
where it is stated rather than deleting it.

Every prompt the pipeline sends is one of the files above; there are no other prompt
strings in the code. The final skill is produced by a single merge call over both
partitions' cluster proposals (paper §2.6).

## Removed: `extra_constraints.md`

Earlier versions shipped one file the code *read* rather than mirrored: a 1280-character
hand-written "ADDITIONAL DOMAIN GUARDRAIL" block that was spliced into the merge prompt
by default. It was removed, along with the `extra_constraints` parameter and the
`--extra-constraints-file` flag, for two reasons:

1. **It was SpreadsheetBench-specific** (formula recalculation, `#NAME?`, headless
   LibreOffice), yet it was a global default. DAPO and PR-review runs got
   SpreadsheetBench instructions in their merge prompt.
2. **It was hand-written domain knowledge** injected into an automatic method, and it
   is not part of the method the paper describes.

The SpreadsheetBench base run had this block in its merge prompt, and that number has
not been re-measured without it (see `RUNNING.md`, "Reproducibility caveats").
`tests/test_prompt_drift.py::test_no_hand_written_guardrail_block_ships` pins that
nothing hand-written and benchmark-specific comes back into this directory.

## Coverage checks

- `test_every_prompt_file_is_covered_or_listed`: every file here has a drift test, and
  every drift test has a file.
- `test_every_shipped_framing_constant_is_mirrored`: every module-level `*FRAMING*`
  constant in `synthesize.py` has a file here. The only one left is
  `_MERGE_FRAMING_SAS`. The additive framings it replaced, and the per-partition
  framings used by the removed per-partition synthesis step, were deleted from the
  code.
