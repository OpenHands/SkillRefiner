"""Importing the package must not touch the network.

litellm fetches a model-cost map at import time unless LITELLM_LOCAL_MODEL_COST_MAP
is set. conftest.py sets it; this test is what stops that guard being removed.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ART_ROOT = Path(__file__).resolve().parent.parent


def test_importing_the_package_opens_no_socket():
    probe = (
        "import socket\n"
        "def _blocked(self, addr):\n"
        "    raise AssertionError(f'network attempted: {addr}')\n"
        "socket.socket.connect = _blocked\n"
        "import skill_refiner.pipeline\n"
        "print('clean')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
    )
    assert "clean" in result.stdout, (
        f"import reached the network:\n{result.stdout}\n{result.stderr}"
    )


# Every .py under scripts/ and baselines/ that can be exec'd standalone. A module
# that raises ImportError on its own (a package-internal baselines module reached
# only through its package) is reported as such and passes: it is not an entry
# point a reader can run, and it never gets far enough to load litellm. What this
# must NOT do is check one script and generalize -- the previous version checked
# scripts/ablations/run.py alone, which happens to be the only entry point that
# imports skill_refiner before openhands, and so was the only one already safe.
_ENTRY_POINT_DIRS = ("scripts", "baselines")


def _shipped_entry_points() -> list[Path]:
    out = []
    for directory in _ENTRY_POINT_DIRS:
        for path in sorted((ART_ROOT / directory).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            out.append(path)
    return out


def _probe_module(path: Path) -> subprocess.CompletedProcess:
    """Exec `path` as a standalone module with every socket connect() fatal.

    The environment is stripped of the two guard variables the Makefile and
    conftest.py export, because the point is whether the FILE sets them itself.
    """
    probe = (
        "import socket, sys, importlib.util\n"
        "def _blocked(self, addr):\n"
        "    raise AssertionError(f'network attempted: {addr}')\n"
        "socket.socket.connect = _blocked\n"
        "s = importlib.util.spec_from_file_location('probe_mod', sys.argv[1])\n"
        "m = importlib.util.module_from_spec(s)\n"
        "try:\n"
        "    s.loader.exec_module(m)\n"
        "except AssertionError:\n"
        "    raise\n"
        "except BaseException as exc:\n"
        "    print(f'not-standalone: {type(exc).__name__}')\n"
        "print('clean')\n"
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("LITELLM_LOCAL_MODEL_COST_MAP", "OPENHANDS_SUPPRESS_BANNER")
    }
    return subprocess.run(
        [sys.executable, "-c", probe, str(path)],
        capture_output=True,
        text=True,
        cwd=ART_ROOT,
        env=env,
    )


def test_there_are_entry_points_to_check():
    """A globbing bug that finds nothing would make the sweep below vacuously green."""
    assert len(_shipped_entry_points()) > 50


@pytest.mark.parametrize(
    "rel", [str(p.relative_to(ART_ROOT)) for p in _shipped_entry_points()]
)
def test_every_shipped_entry_point_opens_no_socket(rel):
    """13 of these reached raw.githubusercontent.com at import before the guard.

    litellm's module-level get_model_cost_map() does a live HTTPS GET unless
    LITELLM_LOCAL_MODEL_COST_MAP is set. skill_refiner/__init__.py sets it, but
    ruff's isort orders `openhands` before `skill_refiner`, so any script whose
    first import is an openhands/litellm one loaded litellm before the guard.
    Each such file now carries `__import__("skill_refiner")` as a statement ahead
    of its import block -- a statement, because an import would be re-sorted.
    """
    result = _probe_module(ART_ROOT / rel)
    assert "clean" in result.stdout, (
        f"{rel} reached the network at import:\n{result.stdout}\n{result.stderr}"
    )
