#!/usr/bin/env python
"""Showable-artifact bundle (guide §11, Phase 5 milestone): zip of
src + tests + configs + scripts + experiments + DESIGN_DECISIONS.md +
WALKTHROUGH.md + pyproject/README, EXCLUDING data/artifacts/results and the
internal constitution (docs/guide.md stays out of any shared bundle).

Writes artifacts/eager-showable-<label>.zip (gitignored).

Usage (from the repo root):
    python scripts/make_showable_zip.py --label phase-5
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

INCLUDE_TREES = ["src", "tests", "configs", "scripts", "experiments"]
INCLUDE_FILES = ["docs/DESIGN_DECISIONS.md", "docs/WALKTHROUGH.md",
                 "docs/BASELINE_FIDELITY.md", "pyproject.toml", "README.md",
                 ".gitignore", ".gitattributes"]
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="phase-5")
    args = parser.parse_args(argv)

    out = REPO / "artifacts" / f"eager-showable-{args.label}.zip"
    out.parent.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for tree in INCLUDE_TREES:
            for p in sorted((REPO / tree).rglob("*")):
                if p.is_dir() or (set(p.parts) & EXCLUDE_PARTS):
                    continue
                zf.write(p, p.relative_to(REPO))
                n += 1
        for f in INCLUDE_FILES:
            p = REPO / f
            if p.exists():
                zf.write(p, p.relative_to(REPO))
                n += 1
    print(f"wrote {out} ({n} files, {out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
