# Prompts

The prompts SkillRefiner sends, one file each, matching Appendix A.5 of the paper.
Each file mirrors the string the code builds at runtime, filled in with example
values (`xlsx` as the skill name, `SKILL` / `SUMMARIES` for the current skill and the
summary block, and a one-cluster example). `tests/test_prompt_drift.py` checks that
each file matches its builder exactly.

| file | paper | built by |
|---|---|---|
| `summarize_positive.md` | A.5.1, polarity=positive | `skill_refiner/summarize/llm.py::_build_prompt` |
| `summarize_negative.md` | A.5.1, polarity=negative | `skill_refiner/summarize/llm.py::_build_prompt` |
| `skill_run_header.md` | the `{header}` line in A.5.1 | `skill_refiner/summarize/llm.py::_skill_run_header` |
| `positive_cluster.md` | A.5.2, polarity=positive | `skill_refiner/propose/polarity.py::_positive_cluster_prompt` |
| `negative_cluster.md` | A.5.2, polarity=negative | `skill_refiner/propose/polarity.py::_negative_cluster_prompt` |
| `evidence_verify_gate.md` | A.5.3 | `skill_refiner/propose/polarity.py::_verify_prompt` |
| `merge.md` | A.5.4 | `skill_refiner/merge/synthesize.py::_merge_prompt` |
| `merge_framing.md` | A.5.4, the framing paragraph in `merge.md` | `skill_refiner/merge/synthesize.py::_MERGE_FRAMING_SAS` |

`merge.md` produces the refined skill: it combines the current skill, the positive
and evidence-gated negative cluster proposals, and the merge framing (§2.6). The
framing tells the model how to resolve conflicts: when a failure cluster limits a
behavior that a success cluster endorses, the merge qualifies that behavior where it
is stated rather than deleting it.
