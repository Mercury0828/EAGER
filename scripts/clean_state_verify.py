#!/usr/bin/env python
"""Clean-state verification (guide §11): from a FRESH clone + FRESH venv,

  1. install the package,
  2. run the end-to-end episode script BEFORE pytest and record its output,
  3. run the full pytest suite,
  4. re-run the episode script and require byte-identical output,
  5. snapshot the file tree before/after pytest and require no new files
     outside interpreter/runner caches (no test pollution: tests write only
     to pytest tmp dirs; results/ belongs to experiments/).

Prints a PASS/FAIL verdict; exit code 0 only on full PASS.

Usage (from the repo root):
    python scripts/clean_state_verify.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

EPISODES = [
    ["--hardware", "configs/hardware/k2_line.yaml",
     "--circuit", "configs/circuits/golden_micro_1.yaml",
     "--seed", "0", "--policy", "jit"],
    ["--hardware", "configs/hardware/golden_k2_det.yaml",
     "--circuit", "configs/circuits/golden_micro_2.yaml",
     "--seed", "1", "--policy", "jit"],
]

# Interpreter/runner noise, not test-written data.
SNAPSHOT_EXCLUDE = {".git", ".venv", "__pycache__", ".pytest_cache"}


def run(cmd, cwd, check=True):
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(map(str, cmd))}")
    return proc


def snapshot_tree(root: Path) -> set[str]:
    files = set()
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(root)
        parts = set(rel.parts)
        if parts & SNAPSHOT_EXCLUDE or any(part.endswith(".egg-info")
                                           for part in rel.parts):
            continue
        files.add(str(rel))
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent),
                        help="repo to clone (default: this script's repo)")
    args = parser.parse_args(argv)

    tmp = Path(tempfile.mkdtemp(prefix="eager_clean_"))
    clone = tmp / "clone"
    print(f"clean-state workdir: {tmp}")

    run(["git", "clone", "--quiet", args.repo, str(clone)], cwd=tmp)
    run([sys.executable, "-m", "venv", str(clone / ".venv")], cwd=clone)
    venv_py = clone / (".venv/Scripts/python.exe" if sys.platform == "win32"
                       else ".venv/bin/python")
    print("installing into fresh venv ...")
    run([str(venv_py), "-m", "pip", "install", "-e", ".[dev]", "--quiet"],
        cwd=clone)

    def episode_outputs() -> list[str]:
        outs = []
        for ep in EPISODES:
            proc = run([str(venv_py), "scripts/run_episode.py", *ep], cwd=clone)
            outs.append(proc.stdout)
        return outs

    print("\n=== episode runs BEFORE pytest ===")
    before = episode_outputs()
    for out in before:
        print(out, end="")

    tree_before = snapshot_tree(clone)

    print("\n=== pytest (fresh clone) ===")
    proc = run([str(venv_py), "-m", "pytest", "-q"], cwd=clone, check=False)
    print(proc.stdout[-2000:])
    pytest_ok = proc.returncode == 0

    tree_after = snapshot_tree(clone)
    new_files = sorted(tree_after - tree_before)
    removed = sorted(tree_before - tree_after)

    print("=== episode runs AFTER pytest ===")
    after = episode_outputs()
    for out in after:
        print(out, end="")

    identical = before == after
    clean = not new_files and not removed

    print("\n=== clean-state verdict ===")
    print(f"pytest green:               {'PASS' if pytest_ok else 'FAIL'}")
    print(f"no test pollution:          {'PASS' if clean else 'FAIL'}"
          + (f"  new={new_files} removed={removed}" if not clean else ""))
    print(f"episode outputs identical:  {'PASS' if identical else 'FAIL'}")
    ok = pytest_ok and clean and identical
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
