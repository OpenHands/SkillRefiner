"""Shared LLM connection glue for the benchmark drivers and parametric-seed writers.

Pure LLM base-url / api-key resolution (``_LLMConfig``, ``_register_extra_models``,
``_resolve_llm_config``) used by the benchmark rollout runners, the baselines and the
parametric-seed writers.

Every ``scripts/parametric_seed/*.py`` and ``scripts/benchmarks/**/*.py`` module that
needs this inserts ``scripts/`` (this file's directory) onto ``sys.path`` and imports
from here.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Offline guard: skill_refiner/__init__.py sets LITELLM_LOCAL_MODEL_COST_MAP before
# litellm's import-time HTTPS GET to raw.githubusercontent.com. It must run ahead of the
# openhands/litellm imports below, and a plain `import skill_refiner` cannot -- ruff's
# isort sorts `openhands` first, which is precisely how the guard got skipped here.
__import__("skill_refiner")  # noqa: F401 - imported for its import-time side effect

import litellm  # noqa: E402

# No deployment URL is baked in here; you supply your own endpoint. See
# artifact.local.example.json / README "Local configuration" --
# artifact.local.json's llm_base_urls.<provider> or the LLM_BASE_URL env var
# are the way to supply one.
_ARTIFACT_LOCAL_CONFIG_FILE = Path(__file__).resolve().parent.parent / "artifact.local.json"
_API_KEY_ENV_CANDIDATES = ("LLM_API_KEY", "OPENAI_API_KEY")


def _load_local_config() -> dict:
    if not _ARTIFACT_LOCAL_CONFIG_FILE.is_file():
        return {}
    try:
        data = json.loads(_ARTIFACT_LOCAL_CONFIG_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _local_base_url(provider: str) -> str:
    local = _load_local_config().get("llm_base_urls", {})
    return str(local.get(provider, "")).strip() if isinstance(local, dict) else ""


# An unknown --llm-provider exits loudly below, naming what is valid.
_PROVIDER_ALIASES = {
    "eval": "eval_proxy",
    "eval_proxy": "eval_proxy",
}

_MINIMAX_SPEC = {
    "max_input_tokens": 99_072,
    "max_output_tokens": 32_000,
    "litellm_provider": "openai",
    "mode": "chat",
}
_EXTRA_MODEL_REGISTRATIONS = {
    "openai/minimax-m2.7": _MINIMAX_SPEC,
    "openai/MiniMaxAI/MiniMax-M2.7": _MINIMAX_SPEC,
}


@dataclass(frozen=True)
class _LLMConfig:
    model: str
    base_url: str
    api_key: str


def _register_extra_models() -> None:
    """Register the MiniMax model spec with litellm (unpriced, chat-mode, openai-shaped).

    Purely cosmetic for cost accounting — litellm has no built-in pricing for
    MiniMax, so without this it would raise on an unknown model instead of
    reporting `$0.00`.
    """
    litellm.register_model(_EXTRA_MODEL_REGISTRATIONS)


def _resolve_llm_config(
    *,
    model: str,
    provider: str,
    base_url: str,
    api_key: str,
    secrets_file: str,
) -> _LLMConfig:
    normalized = _PROVIDER_ALIASES.get(provider.strip().lower())
    if normalized is None:
        valid = ", ".join(sorted(_PROVIDER_ALIASES))
        print(f"ERROR: unknown --llm-provider '{provider}'. Valid: {valid}", file=sys.stderr)
        sys.exit(1)

    resolved_url = (
        base_url.strip()
        or os.environ.get("LLM_BASE_URL", "").strip()
        or _local_base_url(normalized)
    )
    if not resolved_url:
        print(
            f"ERROR: no base URL configured for provider '{normalized}'. Set --base-url, "
            f"LLM_BASE_URL, or artifact.local.json's "
            f"llm_base_urls.{normalized} (see artifact.local.example.json).",
            file=sys.stderr,
        )
        sys.exit(1)

    resolved_key = api_key.strip()
    if not resolved_key:
        for var in _API_KEY_ENV_CANDIDATES:
            val = os.environ.get(var, "").strip()
            if val:
                resolved_key = val
                break

    if not resolved_key and secrets_file:
        secrets_path = Path(secrets_file)
        if not secrets_path.is_absolute():
            secrets_path = Path.cwd() / secrets_path
        if secrets_path.exists():
            data = json.loads(secrets_path.read_text(encoding="utf-8"))
            for var in _API_KEY_ENV_CANDIDATES:
                val = str(data.get(var, "")).strip()
                if val:
                    resolved_key = val
                    break

    if not resolved_key:
        candidates = ", ".join(f"${v}" for v in _API_KEY_ENV_CANDIDATES)
        print(
            f"ERROR: no API key found. Set --api-key, one of {candidates}, "
            f"or populate {secrets_file}.",
            file=sys.stderr,
        )
        sys.exit(1)

    return _LLMConfig(model=model, base_url=resolved_url, api_key=resolved_key)
