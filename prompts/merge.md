Current skill content:
```markdown
SKILL
```

The following 1 cluster(s) of agent behavior were identified and analyzed across 7 traces:

[Cluster 0 — agent skipped validating the diff before commenting (cluster_size=7, shown_to_llm=5, polarity=negative)]
Reinforce: []
Soften: ['commented without re-reading the diff']
Suggested edit: Require a diff re-read immediately before writing any comment.

Clusters marked polarity=positive come from PASSING traces — validated behaviors to preserve and strengthen. Clusters marked polarity=negative come from FAILING traces — failure modes that ACTUALLY occurred, and must be weighed as seriously as the positive signal (a failure is at least as strong evidence as a success pattern of equal cluster_size). Validated behaviors are the default, NOT sacrosanct. Conflict rule: when a negative cluster shows that a behavior a positive cluster endorses is being over-applied, over-trusted, or misused, do NOT keep the positive text loud and unchanged — qualify, scope, or condition that behavior AT ITS SOURCE so the caveat sits together with the behavior it limits, editing the existing text down rather than appending a distant caveat. Delete a validated behavior outright only when the failure evidence directly contradicts it; otherwise narrow it to the cases where it still holds.

Reply as JSON (no markdown wrapper):
{"proposed_content": "<full revised skill, placeholders intact>", "rationale": "<1-3 sentence explanation of what changed and why>", "key_observations": ["<recurring behavior that drove a change>", "..."], "confidence": <self-assessed confidence, 0.0 to 1.0>}