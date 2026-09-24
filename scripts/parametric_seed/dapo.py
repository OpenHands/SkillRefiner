"""Parametric-knowledge seed skill: a dapo-math SKILL.md authored from pretrained
knowledge, no traces, no ground truth, no human authoring.

This is the paper's LLM-generated DAPO-Math initial skill. It is shared by all three
DAPO arms so they differ by *engine* (SkillRefiner, Trace2Skill's skill_evolver, GEPA),
not by prompt wording or seed-generation sampling luck:

  - ``skill_refiner.pipeline.refine``  (``RefineConfig(skill_file=...)``)
  - ``baselines/trace2skill/dapo_trace2skill_bootstrap_pipeline.py``  (skill_evolver)
  - ``baselines/gepa/gepa_bootstrap_dapo.py``  (GEPA)

Unlike SpreadsheetBench's xlsx skill, a DAPO-Math skill is a single SKILL.md with no
protected aux files (no LICENSE.txt/recalc.py to preserve), so there is no separate
"stage into a skill directory" step: the seed IS the file every engine's
``--seed-skill``/``--parametric-seed-file``/``skill_file`` points at directly.

``resolve_or_generate_seed`` caches the seed to disk: seed generation is a
non-deterministic single sample, and GEPA resumes from its ``gepa_state`` by default, so
a re-run must reuse the byte-identical seed its state belongs to, and so that all three
arms can share one seed by pointing at the same cache path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Wording kept in one place (and deliberately trace-free) so all three bootstrap arms
# are seeded identically.
PARAMETRIC_SEED_PROMPT = (
    "Write a complete SKILL.md for an Agent skill named 'dapo-math' that helps an agent "
    "solve competition-style math word problems step by step. Base this entirely on "
    "your own general knowledge of mathematical problem-solving best practices — do not "
    "reference any specific traces or examples, none are provided. Include: (1) YAML "
    "frontmatter with `name: dapo-math` and a `description` field describing when to "
    "use this skill, (2) a body covering how to parse the problem, choose a solution "
    "strategy, show working, and verify the final answer before finishing. Output ONLY "
    "the SKILL.md content, starting with the YAML frontmatter delimiter `---`."
)


def bare_model(model: str) -> str:
    """Strip a leading litellm ``openai/`` prefix.

    The rollout stages (litellm/openhands-sdk) take a provider-prefixed model id
    (``openai/minimax-m2.7``); the raw ``openai.OpenAI``-format client used for one-off
    seed-generation calls needs the bare name (``minimax-m2.7``) or the eval proxy
    404s. Only the leading prefix is removed, so an id that legitimately contains a
    slash elsewhere keeps the rest.
    """
    prefix = "openai/"
    return model[len(prefix) :] if model.startswith(prefix) else model


def generate_parametric_seed(client, model: str) -> str:
    """One chat completion authoring the seed SKILL.md; returns its raw text.

    ``client`` is injected and duck-typed to ``openai.OpenAI``
    (``client.chat.completions.create(model=..., messages=...)``), so callers own the
    wiring and tests need no network. ``model`` must be the **bare** model string the raw
    OpenAI-format proxy expects (``gpt-5.4-mini``, not ``openai/gpt-5.4-mini``).
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PARAMETRIC_SEED_PROMPT}],
    )
    content = response.choices[0].message.content or ""
    if not content.strip():
        raise RuntimeError(f"parametric seed generation returned empty content (model={model})")
    return content


def resolve_or_generate_seed(
    seed_path: Path,
    client,
    model: str,
    regenerate: bool = False,
) -> str:
    """Return the seed text at ``seed_path``, generating and caching it if needed.

    Reusing an on-disk seed is what makes a bootstrap run idempotent across restarts
    and what lets all three bootstrap arms share a single seed: point ``seed_path`` at
    one shared file. ``regenerate=True`` forces a fresh sample and overwrites.
    """
    if seed_path.exists() and not regenerate:
        return seed_path.read_text(encoding="utf-8")
    seed_content = generate_parametric_seed(client, model)
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    seed_path.write_text(seed_content, encoding="utf-8")
    return seed_content


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed-path", type=Path, required=True, metavar="PATH")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--llm-provider", default="eval_proxy", metavar="PROVIDER")
    parser.add_argument("--base-url", default="", metavar="URL")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--secrets-file", default=".eval_proxy.secrets.json", metavar="PATH")
    parser.add_argument(
        "--seed-model",
        default="",
        metavar="NAME",
        help="Bare model string for the raw-client seed-generation call (default: "
        "--model with a leading 'openai/' stripped).",
    )
    parser.add_argument("--regenerate", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    import sys

    from openai import OpenAI

    _scripts_dir = Path(__file__).resolve().parents[1]  # scripts/ (holds _llm_config.py)
    if str(_scripts_dir) not in sys.path:
        sys.path.insert(0, str(_scripts_dir))
    from _llm_config import _resolve_llm_config

    args = parse_args()
    cfg = _resolve_llm_config(
        model=args.model,
        provider=args.llm_provider,
        base_url=args.base_url,
        api_key=args.api_key,
        secrets_file=args.secrets_file,
    )
    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    seed_model = args.seed_model or bare_model(cfg.model)
    existed = args.seed_path.exists() and not args.regenerate
    content = resolve_or_generate_seed(
        args.seed_path, client, seed_model, regenerate=args.regenerate
    )
    print(f"{'Reused' if existed else 'Generated'} seed ({len(content)} chars): {args.seed_path}")


if __name__ == "__main__":
    main()
