Current skill content:
```markdown
SKILL
```

The following 3 agent trace summaries form a semantic cluster (cluster 0). All traces in this cluster come from runs that FAILED their evaluation — they represent a recurring failure mode.

SUMMARIES

Based on this cluster of failing traces:
1. Identify the common theme across these traces.
2. Identify the specific behaviors that caused failure and should be SOFTENED or removed — instructions that misled the agent, missing guardrails, or friction that wasted effort.
3. Propose one concrete, targeted edit to the skill — a guardrail to add or a misleading instruction to remove — that would prevent this failure mode from repeating.

Do NOT invent praise — these traces failed; your job is to stop the failure from repeating.

Reply as JSON (no markdown wrapper):
{"theme": "<what these traces have in common>", "soften": ["<behavior to remove or de-emphasise>", "..."], "suggested_edit": "<one concrete minimal edit that prevents this failure mode>"}