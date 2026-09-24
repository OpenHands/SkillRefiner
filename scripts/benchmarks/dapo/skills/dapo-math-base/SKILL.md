---
name: dapo-math-base
description: "This skill should be used when the user asks to solve DAPO-Math, olympiad-style math prompts, answer math competition problems, run DAPO rollouts, or evaluate math-agent traces that require a final `Answer: ...` line."
---

# DAPO-Math Base Skill

Solve one DAPO-Math prompt carefully, efficiently, and in the required answer format. Treat the task as a contest-style math problem with hidden evaluation against a reward-model ground truth. Optimize for a correct final answer, not for lengthy exposition.

## Required Output Contract

End the final response with exactly one standalone line:

```text
Answer: <final answer>
```

Use the simplest equivalent form for `<final answer>`:

- Use an integer for integer answers.
- Use a reduced fraction like `17/12` for rational answers.
- Use exact symbolic forms when needed, such as `sqrt(3)`, `2*pi`, or a compact expression.
- Avoid trailing punctuation on the answer line.
- Do not include multiple competing final answers.

## Core Workflow

1. **Classify the problem.** Identify whether the prompt is mainly algebra, number theory, combinatorics, probability, geometry, sequences, inequalities, or optimization.
2. **Extract givens and the target.** Rewrite the key quantities, constraints, and requested output before calculating.
3. **Choose the shortest reliable method.** Prefer invariants, symmetry, counting complements, modular arithmetic, factorization, recurrence simplification, or direct algebra over brute-force reasoning.
4. **Carry exact quantities.** Avoid decimal approximations unless the final answer is explicitly numerical or approximation-based.
5. **Check edge cases.** Verify boundary values, positivity/integrality constraints, parity, divisibility, and whether objects are counted once or multiple times.
6. **Sanity-check the result.** Substitute the answer back into the original conditions when possible. For counting/probability, confirm totals and denominators. For geometry, confirm units and scale.
7. **Finish with the required answer line.** Provide concise reasoning above it, then the final `Answer: ...` line.

## Problem-Type Tactics

### Number Theory

- Factor early. For expressions involving divisors, gcd, lcm, valuations, or unit fractions, write prime factorizations and track exponents.
- Use p-adic valuations to control divisibility and denominators.
- For modular constraints, reduce to the smallest relevant modulus and check invertibility before dividing.
- For divisor counts or sums, separate independent prime exponents.
- For Diophantine conditions, derive bounds before enumerating cases.

### Algebra and Equations

- Normalize equations before expanding: move all terms to one side, factor if possible, and look for substitutions.
- Preserve domain restrictions from square roots, logarithms, denominators, and inverse functions.
- For symmetric polynomials, switch to sums/products or known identities.
- For systems, eliminate variables in the order that keeps expressions smallest.
- After solving, reject extraneous roots introduced by squaring or multiplying by variable expressions.

### Combinatorics and Probability

- Define the sample space explicitly before counting favorable cases.
- Prefer complements when forbidden configurations are simpler.
- Use bijections or recursive states when direct enumeration becomes messy.
- Check whether order matters, whether repetition is allowed, and whether rotations/reflections are considered identical.
- For probability, reduce the final fraction.

### Geometry

- Draw a mental diagram and name key points, lengths, and angles.
- Look for similarity, cyclic quadrilaterals, power of a point, angle bisectors, area ratios, and coordinate placements.
- Choose coordinates when the diagram has perpendiculars, circles, or many length constraints.
- Use vectors or complex numbers only when they shorten the solution.
- Verify that the final answer has the requested dimension: length, area, angle, ratio, or count.

### Sequences, Recurrences, and Functional Patterns

- Compute a few initial terms to identify structure, then prove or justify the pattern.
- Solve linear recurrences with characteristic roots when applicable.
- For floor/ceiling problems, split into intervals where the expression is constant or monotone.
- For iterative definitions, search for invariants, fixed points, or telescoping expressions.

## Arithmetic and Verification Discipline

- Keep fractions exact and reduce only when helpful.
- Label intermediate variables clearly.
- Recalculate the final arithmetic independently once before finishing.
- If terminal access is available, use Python only for exact arithmetic, symbolic checks, or bounded enumeration; do not rely on floating-point approximations for proof.
- If terminal access is unavailable, keep computations small and structured enough to audit manually.

## DAPO-Specific Notes

- The prompt already contains the required final-answer instruction. Follow it exactly even if the system prompt repeats it.
- The hidden scorer usually compares the extracted final answer to `reward_model.ground_truth`; equivalent simple forms are safest.
- Avoid saying that the ground truth is known. Derive the result from the prompt.
- Prefer a complete but compact solution. Long exploratory text increases risk of contradictions and token cost.
- If the problem is difficult, still return the best derived answer with the required `Answer:` line rather than ending without a parseable answer.

## Final Response Template

Use this structure unless the problem requires an unusual format:

```text
[Concise derivation with exact arithmetic and a quick check.]

Answer: <final answer>
```
