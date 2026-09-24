"""Preflight check: report every prerequisite at once, each with its remedy.

A fresh machine hits these one at a time -- install, then a missing key, then a
missing embedding model three stages into a refine run. This reports all of them
in one pass so the failure mode is "ah, pull that model" rather than "why doesn't
it work".

Required checks (a failure exits non-zero):
  * Python version against pyproject's ``requires-python``
  * every pinned dependency installed at its pinned version, and the pipeline +
    clustering stack actually importable
  * an API key in one of the accepted env vars
  * an LLM base URL resolvable for the ``eval_proxy`` provider
  * the embedding endpoint reachable AND serving the configured model

Informational checks (never fail the run):
  * ``tmux`` on PATH (the OpenHands terminal tool uses it)
  * ``artifact.local.json`` -- reported by KEY only. No value from it is ever
    printed: it is the file readers are told to put endpoints in.

Usage:
  uv run python scripts/doctor.py          # or: make doctor
"""

from __future__ import annotations

# skill_refiner/__init__.py sets LITELLM_LOCAL_MODEL_COST_MAP, which must be set
# before litellm is imported anywhere below. A statement, not an import, so ruff's
# isort cannot reorder it after the openhands/litellm-importing modules.
__import__("skill_refiner")  # noqa: F401

import importlib  # noqa: E402
import importlib.metadata  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tomllib  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402
from pathlib import Path  # noqa: E402

ART = Path(__file__).resolve().parents[1]

# Distributions whose import name differs from their PyPI name. Everything else
# is imported under its own name with '-' mapped to '_'.
_IMPORT_NAMES = {
    "openhands-sdk": "openhands.sdk",
    "openhands-tools": "openhands.tools",
    "scikit-learn": "sklearn",
    "umap-learn": "umap",
}

# Imported for real (not just checked present): the modules refine() loads on the
# way to a proposal. A dependency can be installed and still fail to import -- a
# mismatched numba/llvmlite pair is the usual way.
_MUST_IMPORT = (
    "skill_refiner.pipeline",
    "skill_refiner.cluster.umap_hdbscan",
    "openai",
    "umap",
    "hdbscan",
)

_API_KEY_ENV_CANDIDATES = ("LLM_API_KEY", "OPENAI_API_KEY")
_BASE_URL_ENV_CANDIDATES = ("SKILL_REFINER_BASE_URL_EVAL_PROXY", "LLM_BASE_URL")

_DEFAULT_EMBEDDING_BASE_URL = "http://localhost:11434/v1"
_DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:4b"

OK, FAIL, INFO, WARN = "  OK  ", " FAIL ", " info ", " warn "


class Report:
    """Collects one line per check; only `required=True` failures set the exit code."""

    def __init__(self) -> None:
        self.failed = False

    def add(self, status: str, label: str, detail: str, remedy: str = "") -> None:
        if status == FAIL:
            self.failed = True
        print(f"[{status}] {label:<26} {detail}")
        if remedy:
            for line in remedy.splitlines():
                print(f"{'':>9}{line}")


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def check_python(report: Report) -> None:
    raw = tomllib.loads((ART / "pyproject.toml").read_text(encoding="utf-8"))
    spec = raw["project"]["requires-python"]
    running = ".".join(str(p) for p in sys.version_info[:3])
    match = re.match(r">=\s*(\d+)\.(\d+)", spec)
    minimum = (int(match.group(1)), int(match.group(2))) if match else (0, 0)
    if sys.version_info[:2] >= minimum:
        report.add(OK, "python", f"{running} satisfies requires-python {spec}")
    else:
        report.add(
            FAIL, "python", f"{running} does not satisfy requires-python {spec}",
            f"Install Python {minimum[0]}.{minimum[1]}+ and re-run `uv sync` "
            "(uv will fetch it: `uv python install "
            f"{minimum[0]}.{minimum[1]}`).",
        )


def _pinned_dependencies() -> list[tuple[str, str]]:
    raw = tomllib.loads((ART / "pyproject.toml").read_text(encoding="utf-8"))
    out = []
    for entry in raw["project"]["dependencies"]:
        name, _, version = entry.partition("==")
        out.append((name.strip(), version.strip()))
    return out


def check_dependencies(report: Report) -> None:
    missing: list[str] = []
    mismatched: list[str] = []
    for name, pinned in _pinned_dependencies():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
            continue
        if pinned and installed != pinned:
            mismatched.append(f"{name} {installed} (pinned {pinned})")

    if missing:
        report.add(
            FAIL, "dependencies", f"{len(missing)} not installed: {', '.join(sorted(missing))}",
            "Run `uv sync` from the repo root.",
        )
    elif mismatched:
        report.add(
            FAIL, "dependencies", f"{len(mismatched)} off their pin: {'; '.join(mismatched)}",
            "Run `uv sync` -- the pins are not advisory. Clustering results and "
            "prompt token budgets both change silently across versions.",
        )
    else:
        report.add(OK, "dependencies", f"all {len(_pinned_dependencies())} pins installed")

    broken = []
    for module in _MUST_IMPORT:
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - any import failure is the finding
            broken.append(f"{module}: {type(exc).__name__}: {exc}")
    if broken:
        report.add(
            FAIL, "imports", f"{len(broken)} failed", "\n".join(broken) + "\nRun `uv sync`.",
        )
    else:
        report.add(
            OK, "imports",
            f"pipeline + clustering stack import ({len(_MUST_IMPORT)} modules)",
        )


def check_api_key(report: Report) -> None:
    for var in _API_KEY_ENV_CANDIDATES:
        if os.environ.get(var, "").strip():
            report.add(OK, "LLM api key", f"${var} is set")
            return
    report.add(
        FAIL, "LLM api key", "no API key in the environment",
        "Export one of " + ", ".join(f"${v}" for v in _API_KEY_ENV_CANDIDATES)
        + ".\nThe benchmark scripts also accept `--secrets-file <file>.secrets.json` "
        "carrying the\nsame key names (that filename pattern is gitignored).",
    )


def _local_config() -> dict:
    path = ART / "artifact.local.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def check_llm_base_url(report: Report) -> None:
    for var in _BASE_URL_ENV_CANDIDATES:
        if os.environ.get(var, "").strip():
            report.add(OK, "LLM base url", f"${var} is set")
            return
    urls = _local_config().get("llm_base_urls")
    if isinstance(urls, dict) and str(urls.get("eval_proxy", "")).strip():
        report.add(OK, "LLM base url", "artifact.local.json llm_base_urls.eval_proxy is set")
        return
    report.add(
        FAIL, "LLM base url", "no base URL for llm_provider='eval_proxy'",
        "Export $SKILL_REFINER_BASE_URL_EVAL_PROXY (or $LLM_BASE_URL for the "
        "scripts/ drivers),\nor copy artifact.local.example.json to "
        "artifact.local.json and fill in\nllm_base_urls.eval_proxy. There is no "
        "baked-in default -- an OpenAI-compatible\n/chat/completions endpoint is "
        "what is expected.",
    )


def _http_get_json(url: str, timeout: float = 5.0) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def check_embeddings(report: Report) -> None:
    """The hidden hard requirement: a local Ollama with one specific model pulled.

    Clustering cannot run without it, and nothing earlier in a refine run says so --
    the failure lands several minutes in, after every summary has been paid for.
    """
    base_url = os.environ.get("EMBEDDING_BASE_URL", "").strip() or _DEFAULT_EMBEDDING_BASE_URL
    model = os.environ.get("EMBEDDING_MODEL", "").strip() or _DEFAULT_EMBEDDING_MODEL
    remedy_point_elsewhere = (
        "Or point at a different OpenAI-compatible embedding endpoint with\n"
        "$EMBEDDING_BASE_URL / $EMBEDDING_MODEL (and $EMBEDDING_API_KEY if it needs one)."
    )
    try:
        payload = _http_get_json(base_url.rstrip("/") + "/models")
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        report.add(
            FAIL, "embedding endpoint", f"{base_url} unreachable ({type(exc).__name__}: {exc})",
            "The default is a LOCAL Ollama. Start it and pull the model:\n"
            "  ollama serve\n"
            f"  ollama pull {model}\n" + remedy_point_elsewhere,
        )
        return

    available = sorted(
        str(entry.get("id", "")) for entry in payload.get("data", []) if entry.get("id")
    )
    if model in available:
        report.add(OK, "embedding endpoint", f"{base_url} serving {model}")
        return
    listed = ", ".join(available) if available else "(none)"
    report.add(
        FAIL, "embedding endpoint", f"{base_url} is up but does not serve {model}",
        f"Models it does serve: {listed}\n"
        f"Pull the configured one:\n  ollama pull {model}\n" + remedy_point_elsewhere,
    )


def check_tmux(report: Report) -> None:
    path = shutil.which("tmux")
    if path:
        report.add(OK, "tmux", path)
    else:
        report.add(
            INFO, "tmux", "not on PATH (optional)",
            "The OpenHands terminal tool uses it (SpreadsheetBench rollouts, and DAPO "
            "with --use-terminal).\nInstall via your package manager (e.g. `brew "
            "install tmux`, `apt install tmux`).",
        )


def check_local_config(report: Report) -> None:
    """Reports KEYS only. Values in this file are endpoints and are never printed."""
    path = ART / "artifact.local.json"
    if not path.is_file():
        report.add(
            INFO, "artifact.local.json", "absent",
            "Optional: only needed if you prefer a file over env vars. "
            "Copy artifact.local.example.json\nto artifact.local.json -- it is "
            "gitignored. Only `llm_base_urls` is read from it.",
        )
        return
    data = _local_config()
    if not data:
        report.add(
            WARN, "artifact.local.json", "present but empty or unparseable",
            f"Check the JSON in {path.name}; it is being ignored as written.",
        )
        return
    keys = ", ".join(sorted(data))
    urls = data.get("llm_base_urls")
    providers = (
        ", ".join(sorted(k for k, v in urls.items() if str(v).strip()))
        if isinstance(urls, dict) else "(none)"
    )
    report.add(
        INFO, "artifact.local.json", f"present; keys: {keys}",
        f"llm_base_urls providers set: {providers}  (values not shown)",
    )


def main() -> int:
    print(f"SkillRefiner doctor — {ART}\n")
    report = Report()
    check_python(report)
    check_dependencies(report)
    check_api_key(report)
    check_llm_base_url(report)
    check_embeddings(report)
    print()
    check_tmux(report)
    check_local_config(report)
    print()
    if report.failed:
        print("FAILED — fix the [ FAIL ] lines above, then re-run `make doctor`.")
        return 1
    print("All required checks passed. Try: uv run python examples/minimal/run.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
