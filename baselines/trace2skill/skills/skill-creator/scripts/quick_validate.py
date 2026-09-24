#!/usr/bin/env python3
"""Minimal skill-format validator for the Trace2Skill evolver.

The evolver requires a validator at this path (skills/skill-creator/scripts/
quick_validate.py) and invokes it as `python quick_validate.py <skill_dir>`,
treating exit 0 as "valid". The upstream Anthropic skill-creator validator is
not vendored here, so this provides an equivalent lightweight check: the skill
dir must contain a SKILL.md with YAML frontmatter carrying `name` and
`description`. That is enough to catch gross corruption during patch application
while matching the evolver's own "missing script => skip validation" leniency.
"""
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: quick_validate.py <skill_dir>", file=sys.stderr)
        return 2
    skill_dir = Path(sys.argv[1])
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        print(f"INVALID: no SKILL.md in {skill_dir}", file=sys.stderr)
        return 1
    text = skill_md.read_text(encoding="utf-8")
    if not text.lstrip().startswith("---"):
        print("INVALID: SKILL.md missing YAML frontmatter (--- fence)", file=sys.stderr)
        return 1
    parts = text.split("---", 2)
    if len(parts) < 3:
        print("INVALID: SKILL.md frontmatter not closed", file=sys.stderr)
        return 1
    front = parts[1]
    for key in ("name:", "description:"):
        if key not in front:
            print(f"INVALID: SKILL.md frontmatter missing '{key}'", file=sys.stderr)
            return 1
    print("OK: skill format valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
