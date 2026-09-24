"""Test-session environment guards.

Set before any import pulls in openhands.sdk -> litellm, whose module-level
get_model_cost_map() otherwise performs a live HTTPS GET to raw.githubusercontent.com.
The suite must run offline: in a sandboxed CI runner or on a plane, that fetch is a
hang or a hard failure with no obvious cause.
"""

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
