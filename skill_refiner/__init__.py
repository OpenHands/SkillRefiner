"""SkillRefiner — refining agent skills from execution traces."""

import os

# Set before any submodule import reaches openhands.sdk -> litellm, whose
# module-level get_model_cost_map() otherwise performs a live HTTPS GET to
# raw.githubusercontent.com. Every documented entry point here — pytest, make,
# and direct `uv run python scripts/...` invocations — must work offline.
#
# TRADE-OFF, chosen deliberately: this is a process-wide side effect of merely
# importing the package, not just of running its entry points. A host process
# that imports skill_refiner alongside other libraries inherits these settings
# for its whole lifetime. setdefault leaves an explicit caller override intact.
#
# This alone is NOT sufficient, and used not to be enough in practice. It only
# fires once something imports skill_refiner, and ruff's isort orders `openhands`
# before `skill_refiner`, so 13 entry points under scripts/ and baselines/ loaded
# litellm — and made the HTTPS GET — before ever reaching this file. Each of them
# now carries `__import__("skill_refiner")` as a statement ahead of its import
# block (a statement, because an import would be re-sorted right back). A newly
# added script that forgets the preamble is caught by
# tests/test_no_network_on_import.py, which sweeps every shipped entry point.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")

__version__ = "1.0.0"
