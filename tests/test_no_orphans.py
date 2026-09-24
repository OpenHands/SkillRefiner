"""The shipped package carries no unreachable module-level symbols.

The artifact advertises itself as the minimum code required to run SkillRefiner.
This test is what keeps that true as the package changes.

It runs a fixpoint reachability sweep, because a single reference-counting pass
under-reports. Removing one orphan can orphan the
symbols only it called (e.g. a dead `cluster_summaries` keeping four `cluster_*`
helpers looking "referenced"), so this test re-counts references while
ignoring any that originate inside a body already ruled dead, until nothing
new turns up dead.
"""

import ast
from collections import defaultdict
from pathlib import Path

ART = Path(__file__).resolve().parent.parent
PKG = ART / "skill_refiner"
ROOTS = [PKG, ART / "tests", ART / "scripts", ART / "baselines"]

# This file itself is excluded from every root below (see _py). Its
# ALLOWED_ORPHANS dict holds symbol names as dict keys -- string constants
# that would otherwise satisfy the "string literal equals a defined name"
# heuristic a few lines down, making this test's own "these are dead" list
# count as evidence that the named symbols are alive.

# Symbols kept deliberately despite zero static references, with the reason.
# Add here ONLY with evidence of dynamic reference or documented public surface.
ALLOWED_ORPHANS: dict[str, str] = {}

# See the module docstring above: this test's own ALLOWED_ORPHANS keys would
# otherwise self-satisfy the string-literal reference heuristic.
_EXCLUDED_FILENAMES = {"test_no_orphans.py"}


def _py(root):
    if not root.exists():
        return []
    return [
        p for p in root.rglob("*.py")
        if ".venv" not in p.parts
        and "__pycache__" not in p.parts
        and p.name not in _EXCLUDED_FILENAMES
    ]


def _refs_excluding(dead: set[str], defined: dict[str, list[str]]) -> dict[str, int]:
    """Recount references, ignoring any that originate inside a dead symbol's body."""
    counts: dict[str, int] = defaultdict(int)
    for root in ROOTS:
        for f in _py(root):
            tree = ast.parse(f.read_text())
            for top in tree.body:
                # Skip the whole body of a symbol already ruled dead.
                if (
                    isinstance(top, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
                    and top.name in dead
                ):
                    continue
                for node in ast.walk(top):
                    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                        continue
                    if isinstance(node, ast.Name):
                        counts[node.id] += 1
                    elif isinstance(node, ast.Attribute):
                        counts[node.attr] += 1
                    elif isinstance(node, ast.ImportFrom):
                        for a in node.names:
                            counts[a.name] += 1
                    elif (
                        isinstance(node, ast.Constant)
                        and isinstance(node.value, str)
                        and node.value in defined
                    ):
                        counts[node.value] += 1
    return counts


def test_no_unreachable_module_level_symbols():
    defined: dict[str, list[str]] = defaultdict(list)
    for f in _py(PKG):
        for node in ast.parse(f.read_text()).body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                defined[node.name].append(f"{f.relative_to(ART)}:{node.lineno}")

    # Iterate to a fixpoint: a single pass flags a dead root (e.g. an
    # unreferenced `cluster_summaries`) but not the helpers only it calls,
    # because those helpers still look referenced from inside the dead root's
    # own body. Re-count with dead bodies excluded until nothing new dies.
    dead: set[str] = set()
    while True:
        current = _refs_excluding(dead, defined)
        newly = {
            n for n in defined
            if not n.startswith("__") and n not in dead and current[n] == 0
        }
        if not newly:
            break
        dead |= newly

    orphans = {
        name: defined[name] for name in dead
        if name not in ALLOWED_ORPHANS
    }
    assert not orphans, (
        "unreachable symbols in the shipped package:\n"
        + "\n".join(f"  {n} at {', '.join(s)}" for n, s in sorted(orphans.items()))
        + "\nPrune them, or add to ALLOWED_ORPHANS with evidence of dynamic reference."
    )
