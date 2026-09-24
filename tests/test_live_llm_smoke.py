"""Opt-in live smoke test: one real completion through the pipeline's own LLM path.

Why this exists
---------------
Every other test in this suite stubs the LLM. That is deliberate — the default
suite must run offline (see ``conftest.py`` and ``test_no_network_on_import.py``)
— but it has a blind spot this test exists to cover: **the offline suite cannot
detect dependency-level LLM breakage.** A stubbed ``LLM`` never executes
openhands-sdk's response-handling path, so a transitive dependency can change an
API out from under that path and every offline test still passes.

For example, with litellm 1.95.0 openhands-sdk's ``normalize_usage`` raises
``AttributeError`` on every completion. The proposal stage treats a failed call as
"no proposal" (``skill_refiner/propose/polarity.py``), so such a run would exit 0
with zero clusters. ``pyproject.toml`` pins both ``openhands-sdk`` and ``litellm``;
this test notices if a similar break appears.

Running it
----------
``uv run pytest -m live``

It is deselected from the default run by the ``-m 'not live'`` entry in
``pyproject.toml``'s ``addopts``, so ``uv run pytest`` stays offline and
network-free.

It needs the same configuration a real run needs (see README > "Local
configuration"):

* ``LLM_API_KEY`` (or ``OPENAI_API_KEY``), and
* ``SKILL_REFINER_BASE_URL_EVAL_PROXY`` — or ``llm_base_urls.eval_proxy`` in the
  gitignored ``artifact.local.json``.

Without those it skips rather than fails: a missing credential is a local setup
gap, not a defect in the artifact.
"""

import os
from pathlib import Path

import pytest

from skill_refiner.pipeline import (
    _API_KEY_ENV_CANDIDATES,
    RefineConfig,
    _build_llm,
    _resolve_provider_base_url,
)

pytestmark = pytest.mark.live

# Stated in full here because it is the thing a reader most needs to understand
# when this test is skipped: the rest of the suite passing proves nothing about
# whether a real LLM call works.
_WHY = (
    "live LLM smoke test is not configured. The offline suite cannot detect "
    "dependency-level LLM breakage -- it stubs every LLM call, so a transitive "
    "dependency (litellm/openhands-sdk) can break every real completion while "
    "all offline tests still pass. Configure %s and a base URL "
    "($SKILL_REFINER_BASE_URL_EVAL_PROXY or artifact.local.json's "
    "llm_base_urls.eval_proxy) and rerun with `uv run pytest -m live`."
)


def _missing_config() -> str | None:
    if not any(os.environ.get(v, "").strip() for v in _API_KEY_ENV_CANDIDATES):
        return _WHY % (" / ".join("$" + v for v in _API_KEY_ENV_CANDIDATES))
    try:
        _resolve_provider_base_url("eval_proxy")
    except RuntimeError:
        return _WHY % "an API key env var"
    return None


def test_real_completion_returns_content(tmp_path: Path) -> None:
    """One real call through ``_build_llm`` comes back with non-empty content.

    Deliberately minimal: this is a correctness check on the dependency stack,
    not a benchmark or a quality assertion about the model's answer.
    """
    skip_reason = _missing_config()
    if skip_reason is not None:
        pytest.skip(skip_reason)

    from openhands.sdk.llm import Message, TextContent

    config = RefineConfig(
        skill_file=tmp_path / "skill.md",
        skill_name="live-smoke",
        raw_jsonl=tmp_path / "traces.jsonl",
        binary_rewards=tmp_path / "rewards.json",
        output_dir=tmp_path / "out",
    )
    llm = _build_llm(config)

    response = llm.completion(
        messages=[Message(role="user", content=[TextContent(text="Reply with one word.")])]
    )

    text = "".join(getattr(part, "text", "") for part in (response.message.content or []))
    assert text.strip(), (
        "the LLM returned no content. If this raised instead, read the traceback: "
        "an AttributeError from openhands-sdk's normalize_usage means the "
        "litellm upper bound in pyproject.toml has been loosened or defeated."
    )
