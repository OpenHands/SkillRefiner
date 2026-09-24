Below is a cleaned execution trace of an AI agent run. Each section is one agent turn or tool result. Internal model reasoning has been stripped — you are seeing only tool calls, tool results, and agent outputs.

--- TRACE START ---
<trace content>
--- TRACE END ---

This trace comes from an AI agent run that invoked the `xlsx` skill and then FAILED its downstream evaluation. Diagnose the failure chain.

Analyze this trace and answer:
1. What did the agent do? List the key tool calls (name + what it was trying to accomplish) and what each returned.
2. Diagnose the failure chain: what was the earliest decision that led the run astray, and what would a correct trace have done there?
3. What instruction or guardrail would have prevented this failure?

Reply as JSON (no markdown wrapper):
{"summary": "<3-5 sentence account of what the agent did and where it went wrong>", "observations": ["<failure mode phrased as a preventable rule>", "..."]}

observations must be failure modes phrased as preventable rules, e.g. "agent skipped reading the diff before commenting — require a diff read first" not "agent made mistakes".