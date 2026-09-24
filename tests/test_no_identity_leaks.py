"""Repo-wide leak guard: scans every git-tracked file, not just .py/.sh/.md.

An earlier hygiene sweep (before this artifact was anonymized) only checked
.py/.sh/.md files by hand, and test_artifact_hygiene.py's own scans walk
`ART.rglob("*.py")`. Both structurally cannot see a *.txt or *.json file. That
blind spot is exactly how the 48 golden fixture files in
tests/fixtures/golden/**/{raw_content.txt,summarizer_prompt.txt} shipped with
the author's absolute home path baked into captured agent traces -- no
extension-based scan was ever going to look at them.

This test instead walks `git ls-files`, so no tracked file is structurally
exempt by extension, and checks for four kinds of leak:

  1. The author's name, in any casing (content, not git commit metadata --
     the user has explicitly said their name may remain in commit history/
     authorship; this only guards file *content*).
  2. An absolute home-directory path (`/Users/...`, `/home/...`), which both
     identifies a specific machine and, combined with a username in the same
     path, doubles as a name leak.
  3. The three org-internal deployment endpoints that used to be baked into
     skill_refiner/pipeline.py and scripts/_llm_config.py as unremovable
     defaults -- see artifact.local.example.json / README "Local
     configuration" for the replacement (env vars / artifact.local.json,
     with a loud failure instead of a silent example.com fallback).
  4. Organisation-named configuration this artifact used to ship under its
     own naming: the removed provider aliases and the removed env vars.
     These are *our* names, not a dependency's, so they identify an
     employer to a blind reviewer. Deliberately narrow: the agent SDK is a
     public PyPI package and is kept, so `from openhands.sdk ...` imports,
     the `openhands-sdk` / `openhands-tools` dependencies, the SDK-defined
     `OPENHANDS_SUPPRESS_BANNER` env var and the SDK-defined `openhands`
     tmux socket / `openhands-pool-*` session names must all keep passing.
     Only names this repo itself chose are forbidden -- see
     `_REMOVED_ORG_CONFIG_RE`.

The endpoint check (3) doesn't hold the hostnames as literals or split
string fragments -- a split fragment still puts every character of the real
hostname in this file's source for anyone reading it, and the Modal service
id *is* the internal identifier. Instead it stores SHA-256 digests of the
hostnames and hashes hostname-shaped tokens pulled out of each scanned
file's text, comparing digests. See `_FORBIDDEN_HOST_DIGESTS` below.

Deliberately excludes: this file itself (it must name the patterns it looks
for), and genuinely binary tracked files (decoded as UTF-8; anything that
isn't skipped rather than false-positiving on garbage bytes).
"""

import hashlib
import re
import subprocess
from pathlib import Path

ART = Path(__file__).resolve().parent.parent
_SELF = Path(__file__).resolve()

_NAME_RE = re.compile(r"anirudh|khatry", re.IGNORECASE)
# Anchored so a directory merely *named* "home" does not match — e.g. a web app's
# "src/components/features/home/foo.tsx". Requires a path start (or a quote/space/
# equals before it) followed by a username segment, which is what a real home path
# looks like: /Users/<name>/... or /home/<name>/...
_HOME_PATH_RE = re.compile(r"""(?:^|[\s"'=:(])/(?:Users|home)/[A-Za-z0-9._-]+/""")

# Hostname-shaped token: dot-separated labels of alnum/hyphen, e.g.
# "sub.example.com" or a Modal-style "some-service-name.modal.run" id.
# Deliberately broad -- it will also match plenty of non-hostnames (e.g.
# "e.g." or "v1.2.3"), which is fine because a match only becomes a hit
# once its hash lands in _FORBIDDEN_HOST_DIGESTS below.
_HOSTNAME_TOKEN_RE = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)+"
)

# SHA-256 hex digests (of the lowercased hostname) of the three org-internal
# deployment endpoints. Stored as digests, not plaintext or split-string
# fragments: a split fragment (e.g. "host-part" + ".example.com") still
# spells out the full hostname to anyone reading this source, and a hosted
# service id is itself the sensitive part -- exactly what this artifact is
# being anonymized to avoid disclosing. A digest discloses nothing, while
# `_hostname_hash_hits` below recovers an exact match by hashing candidate
# tokens extracted from each scanned file, so detection stays real rather
# than becoming a guard that only looks strict. Do NOT "simplify" this back
# to plaintext or concatenated-string literals.
_FORBIDDEN_HOST_DIGESTS = {
    "e573d226233550f47f34a3032cef1b529203a9403357f1d1596c2dd8fe13abf6",
    "0a2a0dc441a41e542c3e2330e34590920c6e91b8bd78dbf8efccfafe40ba28f4",
    "8074e12039db87002f7ca7077bed023e715f4e697719eb304fef134f6ef712c6",
}

# Organisation-named configuration that was removed from this artifact and must
# not come back. Two shapes, both of them names this repo chose for itself:
#
#   * env vars we defined -- any OPENHANDS_* name (OPENHANDS_API_KEY, the
#     former OPENHANDS_LLM_* family of BASE_URL / MODEL / PROVIDER,
#     OPENHANDS_SECRETS_FILE, ...), plus SKILL_REFINER_BASE_URL_OPENHANDS.
#     LLM_API_KEY / LLM_BASE_URL / LLM_MODEL / LLM_PROVIDER /
#     SKILL_REFINER_BASE_URL_EVAL_PROXY / SKILL_REFINER_SECRETS_FILE replaced
#     them. Matched broadly (OPENHANDS_[A-Z_]+) rather than one name at a time,
#     with a single carve-out below for the one OPENHANDS_* var the SDK itself
#     defines -- so a *new* org-named env var we invent later still trips this,
#     not just the ones already caught once.
#   * the provider aliases "openhands" / "openhands_proxy", which only ever
#     appeared as a mapping *key* (`"openhands": ...` in _PROVIDER_ALIASES,
#     _PROVIDER_BASE_URL_ENV_VARS and artifact.local.json's llm_base_urls).
#     Matching the key form specifically is what keeps this from firing on the
#     SDK's own `["tmux", "-L", "openhands", ...]` argument list, which is a
#     public dependency's socket name and is deliberately kept.
#
# Everything the SDK itself defines stays legal: OPENHANDS_SUPPRESS_BANNER
# (the one name in _ALLOWED_OPENHANDS_ENV_VARS below), `from openhands.sdk
# ...`, the openhands-sdk/openhands-tools requirements, and the
# openhands-pool-* tmux session names. Using a public library is not a
# disclosure; naming our own infrastructure after our employer is.
_ALLOWED_OPENHANDS_ENV_VARS = {"OPENHANDS_SUPPRESS_BANNER"}

_REMOVED_ORG_CONFIG_RE = re.compile(
    r"OPENHANDS_[A-Z_]+"
    r"|SKILL_REFINER_BASE_URL_OPENHANDS"
    r"|[\"']openhands_proxy[\"']"
    r"|[\"']openhands[\"']\s*:"
)


def _hostname_hash_hits(line: str) -> list[str]:
    """Return the raw tokens in `line` whose sha256 digest is forbidden."""
    hits = []
    for token in _HOSTNAME_TOKEN_RE.findall(line):
        digest = hashlib.sha256(token.lower().encode("utf-8")).hexdigest()
        if digest in _FORBIDDEN_HOST_DIGESTS:
            hits.append(token)
    return hits


# The datasets/ bundle is local-use research data, deliberately not scrubbed
# (the user runs it on their own machines). It is excluded here for two reasons:
# the pr_review gold corpus IS reviews of specific repositories, so organisation
# identifiers are intrinsic to the data and cannot be removed without destroying
# it; and trace/task content is evidence, not authored artifact text. This guard
# exists to protect the shipped code artifact — see README before publishing the
# bundle anywhere.
_EXCLUDED_DIRS = ("datasets/",)


def _tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ART, capture_output=True, text=True, check=True
    ).stdout
    return [
        ART / line
        for line in out.splitlines()
        if line.strip() and not line.startswith(_EXCLUDED_DIRS)
    ]


def _iter_text_lines(path: Path):
    """Yield (lineno, line) for a tracked file; yields nothing for binary files."""
    try:
        data = path.read_bytes()
    except OSError:
        return
    if b"\x00" in data:
        return  # binary (e.g. .xlsx, images)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return  # not UTF-8 text; treat as opaque/binary
    yield from enumerate(text.splitlines(), start=1)


def test_no_tracked_file_leaks_identity_path_or_internal_endpoint():
    hits = []
    for path in _tracked_files():
        if not path.is_file() or path.resolve() == _SELF:
            continue
        rel = path.relative_to(ART)
        for lineno, line in _iter_text_lines(path):
            snippet = line.strip()[:160]
            if _NAME_RE.search(line):
                hits.append(f"{rel}:{lineno}: author name fragment -> {snippet!r}")
            if _HOME_PATH_RE.search(line):
                hits.append(f"{rel}:{lineno}: absolute home-directory path -> {snippet!r}")
            for token in _hostname_hash_hits(line):
                hits.append(
                    f"{rel}:{lineno}: internal deployment endpoint ({token}) -> {snippet!r}"
                )
            for removed in _REMOVED_ORG_CONFIG_RE.finditer(line):
                token = removed.group(0)
                if token in _ALLOWED_OPENHANDS_ENV_VARS:
                    continue
                hits.append(
                    f"{rel}:{lineno}: removed org-named config ({token}) "
                    f"-> {snippet!r}"
                )
    assert not hits, "leak(s) found in tracked files:\n" + "\n".join(hits)
