Current skill content:
```markdown
SKILL
```

The following 3 agent trace summaries form a semantic cluster (cluster 0). All traces in this cluster come from runs that PASSED their evaluation — they represent validated, successful behavior.

SUMMARIES

Based on this cluster of successful traces:
1. Identify the common theme across these traces.
2. Identify the specific behaviors that drove success and should be REINFORCED in the skill — patterns to codify so they repeat reliably.
3. Propose one concrete, targeted edit to the skill that codifies the winning pattern.

Do NOT propose removals or warnings — these traces succeeded; your job is to lock in what worked.

Reply as JSON (no markdown wrapper):
{"theme": "<what these traces have in common>", "reinforce": ["<behavior to strengthen>", "..."], "suggested_edit": "<one concrete minimal edit that codifies the winning pattern>"}