#!/usr/bin/env python
"""Flaky-bug protocol (guide §11): run the stochastic test suite N times and
report per-test pass counts. Acceptance requires N/N for every test — a
single pass is NOT evidence of correctness.

Writes junit XML only to a temporary directory (test-hygiene rule); prints a
per-test table and exits non-zero if any test passes fewer than N times.

Usage (from the repo root):
    python scripts/run_repeat_suite.py            # 10 runs of -m stochastic
    python scripts/run_repeat_suite.py --runs 3 --marker statistical
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def run_once(marker: str, xml_path: Path) -> dict[str, str]:
    cmd = [sys.executable, "-m", "pytest", "-m", marker, "-q",
           f"--junitxml={xml_path}"]
    subprocess.run(cmd, capture_output=True, text=True)
    outcomes: dict[str, str] = {}
    root = ET.parse(xml_path).getroot()
    for case in root.iter("testcase"):
        name = f"{case.attrib['classname']}::{case.attrib['name']}"
        skipped = case.find("skipped")
        if case.find("failure") is not None or case.find("error") is not None:
            outcomes[name] = "FAIL"
        elif skipped is not None:
            # strict xfail is an EXPECTED outcome: stable iff it xfails N/N
            if skipped.attrib.get("type") == "pytest.xfail":
                outcomes[name] = "XFAIL"
            else:
                outcomes[name] = "SKIP"
        else:
            outcomes[name] = "PASS"
    return outcomes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--marker", default="stochastic")
    args = parser.parse_args(argv)

    per_test: dict[str, list[str]] = defaultdict(list)
    with tempfile.TemporaryDirectory() as td:
        for i in range(args.runs):
            outcomes = run_once(args.marker, Path(td) / f"run{i}.xml")
            if not outcomes:
                print(f"run {i + 1}: no tests collected for -m {args.marker}",
                      file=sys.stderr)
                return 2
            for name, outcome in outcomes.items():
                per_test[name].append(outcome)
            n_pass = sum(1 for o in outcomes.values() if o in ("PASS", "XFAIL"))
            print(f"run {i + 1:2d}/{args.runs}: {n_pass}/{len(outcomes)} as expected")

    width = max(len(n) for n in per_test)
    print(f"\n{'test':<{width}}  expected_outcome_count")
    print("-" * (width + 24))
    all_ok = True
    for name in sorted(per_test):
        outcomes = per_test[name]
        kinds = set(outcomes)
        # stable = the SAME expected outcome (PASS or strict XFAIL) every run
        if kinds == {"PASS"}:
            label = f"{len(outcomes)}/{args.runs} PASS"
        elif kinds == {"XFAIL"}:
            label = f"{len(outcomes)}/{args.runs} XFAIL(strict, expected)"
        else:
            label = "/".join(outcomes)
            all_ok = False
        flag = "" if all_ok or kinds in ({"PASS"}, {"XFAIL"}) else "   <-- NOT STABLE"
        if kinds not in ({"PASS"}, {"XFAIL"}):
            flag = "   <-- NOT STABLE"
        print(f"{name:<{width}}  {label}{flag}")

    print(f"\nverdict: {'ALL STABLE' if all_ok else 'UNSTABLE TESTS PRESENT'} "
          f"({len(per_test)} tests x {args.runs} runs)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
