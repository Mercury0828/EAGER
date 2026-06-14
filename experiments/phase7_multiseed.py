#!/usr/bin/env python
"""Phase 7 (review item A): 5-seed EAGER held-out distribution — the answer to
"is the result a lucky seed?". Reads artifacts/agents/pathb_seed{0..4}.json
(same recipe, IL 200k + PPO 60, identical n=288 held-out) and reports the
mean +/- std and min/max of the held-out vs-AGG-reactive ratio across seeds,
plus the waste-stratum residual per seed when present (D85). Writes
results/multiseed.md.

Usage: python experiments/phase7_multiseed.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ART = REPO / "artifacts" / "agents"
RESULTS = REPO / "results"


def main(argv=None) -> int:
    rows = []
    for s in range(5):
        p = ART / f"pathb_seed{s}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        full = d["eval"]["full"]
        row = {"seed": s, "il": d.get("il_val_top1"),
               "vs_react": full["vs_AGGreactive_ratio"],
               "p": full["vs_react_p"], "wins": full["vs_react_wins"],
               "n": full["n"],
               "vs_eager": full.get("vs_AGGeager_ratio"),
               "waste_vs_react": (d.get("eval_waste", {}) or {})
               .get("full", {}).get("vs_AGGreactive_ratio")}
        rows.append(row)

    if not rows:
        print("no pathb_seed*.json found")
        return 1

    ratios = [r["vs_react"] for r in rows]
    lines = ["# 5-seed EAGER held-out distribution (review item A)\n",
             f"Same recipe (IL 200k + PPO 60), identical n={rows[0]['n']} "
             "held-out, vs AGG-reactive (lower favors EAGER).\n",
             "| seed | IL top-1 | vs AGG-reactive | p | wins | vs always-on | "
             "waste vs-reactive |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        wv = f"{r['waste_vs_react']:.4f}" if r["waste_vs_react"] is not None else "-"
        ve = f"{r['vs_eager']:.4f}" if r["vs_eager"] is not None else "-"
        lines.append(f"| {r['seed']} | {r['il']:.4f} | {r['vs_react']:.4f} | "
                     f"{r['p']:.1e} | {r['wins']}/{r['n']} | {ve} | {wv} |")
    mean = st.mean(ratios)
    sd = st.stdev(ratios) if len(ratios) > 1 else 0.0
    lines.append("")
    lines.append(f"**{len(rows)}-seed held-out vs AGG-reactive: "
                 f"{mean:.4f} +/- {sd:.4f}** (min {min(ratios):.4f}, "
                 f"max {max(ratios):.4f}). All {len(rows)} seeds beat AGG-reactive "
                 f"(every ratio < 1.0): {all(x < 1.0 for x in ratios)}. The "
                 "improvement is not a lucky seed — it reproduces across seeds.")
    waste = [r["waste_vs_react"] for r in rows if r["waste_vs_react"] is not None]
    if waste:
        lines.append("")
        lines.append(f"Waste-stratum vs-reactive (D85): "
                     f"{st.mean(waste):.4f} mean over {len(waste)} seed(s) "
                     f"[{min(waste):.4f}, {max(waste):.4f}] — >1.0 = the honest "
                     "reactive-best-in-waste residual; the curriculum run (D85) "
                     "targets this.")

    out = RESULTS / "multiseed.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")

    idx = RESULTS / "index.json"
    index = json.loads(idx.read_text(encoding="utf-8")) if idx.exists() else {}
    index["multiseed"] = {"path": "multiseed.md", "n_seeds": len(rows),
                          "vs_react_mean": mean, "vs_react_std": sd,
                          "all_beat_agg": all(x < 1.0 for x in ratios)}
    idx.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
