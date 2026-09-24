"""Parametric-knowledge seed skill: an xlsx SKILL.md authored from pretrained knowledge.

This is the paper's LLM-generated SpreadsheetBench initial skill. It is shared by the
arms that start from it so they differ by *engine*, not by prompt wording or sampling
luck:

  - ``baselines/gepa/gepa_bootstrap_pipeline.py``  (GEPA)
  - ``baselines/trace2skill/trace2skill_bootstrap_pipeline.py``  (skill_evolver)

Both engines require an existing seed SKILL.md and neither can author one from nothing.
One plain LLM call — no traces — supplies that seed, so a bootstrap run measures
"pretrained knowledge + behavioral refinement" rather than starting from the canonical
human-written skill.

``resolve_or_generate_seed`` caches the seed to disk: seed generation is a
non-deterministic single sample, and GEPA resumes from its ``gepa_state`` by default, so
a re-run must reuse the byte-identical seed its state belongs to.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# eval_skill_on_spreadsheetbench.py / spreadsheetbench_common.py live in the sibling
# scripts/benchmarks/spreadsheetbench/ driver directory in this artifact's layout.
_BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "spreadsheetbench"

if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from eval_skill_on_spreadsheetbench import (  # noqa: E402
    CANONICAL_XLSX_SKILL_DIR,
    stage_skill_dir,
)
from spreadsheetbench_common import strip_think  # noqa: E402

# Wording kept in one place (and deliberately trace-free) so the GEPA and Trace2Skill
# bootstrap arms are seeded identically.
PARAMETRIC_SEED_PROMPT = (
    "Write a complete SKILL.md for an Agent skill named 'xlsx' that helps an agent "
    "create, edit, and analyze Excel spreadsheets (.xlsx, .xlsm, .csv, .tsv) using "
    "openpyxl. Base this entirely on your own general knowledge of spreadsheet editing "
    "best practices — do not reference any specific traces or examples, none are "
    "provided. Include: (1) YAML frontmatter with `name: xlsx` and a `description` "
    "field describing when to use this skill, (2) a body covering how to inspect a "
    "workbook before editing, how to make targeted edits, how to handle formulas and "
    "recalculation, and how to verify the result before finishing. Output ONLY the "
    "SKILL.md content, starting with the YAML frontmatter delimiter `---`."
)


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
    content = strip_think(response.choices[0].message.content or "")
    if not content.strip():
        raise RuntimeError(f"parametric seed generation returned empty content (model={model})")
    return content


def stage_parametric_seed(seed_content: str, dest_root: Path) -> Path:
    """Stage ``<dest_root>/xlsx/`` = canonical aux files + this seed's SKILL.md.

    The canonical dir's ``LICENSE.txt``/``recalc.py`` are protected files the skill
    engines expect alongside SKILL.md, so they are copied verbatim rather than authored.
    Returns the staged SKILL.md path.
    """
    return stage_skill_dir(
        skill="xlsx",
        canonical_skill_dir=CANONICAL_XLSX_SKILL_DIR,
        candidate_md=seed_content,
        dest_root=dest_root,
    )


def resolve_or_generate_seed(
    seed_path: Path,
    client,
    model: str,
    regenerate: bool = False,
) -> str:
    """Return the seed text at ``seed_path``, generating and caching it if needed.

    Reusing an on-disk seed is what makes a bootstrap run idempotent across restarts
    (GEPA resumes from state tied to one seed) and what lets both bootstrap arms share a
    single seed: point ``seed_path`` at the other arm's file. ``regenerate=True`` forces a
    fresh sample and overwrites.
    """
    if seed_path.exists() and not regenerate:
        return seed_path.read_text(encoding="utf-8")
    seed_content = generate_parametric_seed(client, model)
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    seed_path.write_text(seed_content, encoding="utf-8")
    return seed_content


def bare_model(model: str) -> str:
    """Strip a leading litellm ``openai/`` prefix.

    The raw ``openai.OpenAI``-format client used for the one-off seed-generation call
    needs the bare model name (``minimax-m2.7``) or the eval proxy 404s.
    """
    prefix = "openai/"
    return model[len(prefix) :] if model.startswith(prefix) else model


# --------------------------------------------------------------------------- #
# CLI entry point
#
# Also importable as a library (gepa_bootstrap_pipeline.py and the Trace2Skill
# bootstrap pipeline call ``resolve_or_generate_seed``). This ``parse_args``/``main``
# pair gives it the same ``--seed-path`` CLI as the DAPO parametric-seed writer.
# --------------------------------------------------------------------------- #


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
