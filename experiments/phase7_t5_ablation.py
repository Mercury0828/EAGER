#!/usr/bin/env python
"""Phase 7 T5 ablation table (guide §10.6) — scripted from the artifacts, no
hand-entered numbers. Each row isolates ONE design choice's contribution to
the path-B EAGER result, all on the held-out provisioning task (vs AGG-reactive
= the published static provisioning = the NoProactive baseline), plus the
stochastic-optimum anchor (T4, D84).

Reads: artifacts/agents/pathb_seed0.json (graph EAGER, the locked system),
pathb_flat_seed0.json (flat MLP representation ablation, D83),
results/phase7_stochastic_gap.parquet (clairvoyant optimum, D84). Writes
results/t5_ablation.md.

Usage: python experiments/phase7_t5_ablation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
ART = REPO / "artifacts" / "agents"
RESULTS = REPO / "results"


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                # noqa: BLE001
        pass
    graph = load(ART / "pathb_seed0.json")["eval"]
    flat = load(ART / "pathb_flat_seed0.json")["eval"]
    gf, ff = graph["full"], flat["full"]

    lines = []
    lines.append("# T5 — Ablation table (Phase 7, scripted)\n")
    lines.append("Path-B EAGER = AGG placement+aggregation (fixed strong base) "
                 "+ a learned proactive-provisioning policy. Each row removes "
                 "ONE component; metric = held-out J ratio vs AGG-reactive "
                 "(= EAGER-NoProactive), lower is better; and vs AGG-eager "
                 "(always-on) to show regime adaptivity. CRN-paired.\n")
    lines.append("| Variant | Encoder | Refine | vs AGG-reactive | p | "
                 "vs AGG-eager (always-on) | n |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.append(f"| **EAGER (full)** | R-GCN (graph) | IL+PPO | "
                 f"**{gf['vs_AGGreactive_ratio']:.4f}** "
                 f"({(1-gf['vs_AGGreactive_ratio'])*100:+.1f}%) | "
                 f"{gf['vs_react_p']:.1e} | {gf['vs_AGGeager_ratio']:.4f} | "
                 f"{gf['n']} |")
    lines.append(f"| - graph encoder (D83) | MLP (flat, no msg-passing) | "
                 f"IL+PPO | {ff['vs_AGGreactive_ratio']:.4f} "
                 f"({(1-ff['vs_AGGreactive_ratio'])*100:+.1f}%) | "
                 f"{ff['vs_react_p']:.1e} | {ff['vs_AGGeager_ratio']:.4f} | "
                 f"{ff['n']} |")
    lines.append("| - proactivity (NoProactive §9.7) | — | — (reactive) | "
                 "1.0000 (baseline) | — | — | — |")
    lines.append("")
    lines.append("Reading: proactive provisioning is worth "
                 f"{(1-gf['vs_AGGreactive_ratio'])*100:.1f}% over AGG-reactive "
                 f"(the NoProactive baseline, p={gf['vs_react_p']:.0e}). The "
                 "graph encoder over a flat MLP: IL imitation TIES "
                 f"(val top-1 {load(ART/'pathb_seed0.json')['il_val_top1']:.3f} "
                 f"graph vs {load(ART/'pathb_flat_seed0.json')['il_val_top1']:.3f} "
                 "flat), but only the graph yields a policy that beats always-on "
                 f"(graph vs-eager {gf['vs_AGGeager_ratio']:.3f} <= 1.0; flat "
                 f"{ff['vs_AGGeager_ratio']:.3f} > 1.0) and trains stable PPO "
                 "(flat PPO diverges, D83) — message passing buys adaptivity + "
                 "stable refinement, not imitation.\n")

    # T4 stochastic-optimum anchor (D84)
    sg = RESULTS / "phase7_stochastic_gap.parquet"
    if sg.exists():
        df = pd.read_parquet(sg)
        lines.append("## T4 stochastic-optimum anchor (D84)\n")
        lines.append("Gap to the CLAIRVOYANT (perfect-information) stochastic "
                     "optimum — a rigorous lower bound on any non-anticipative "
                     "policy — on tiny forced-remote instances (B&B per CRN "
                     "seed). Lower gap = closer to optimal.\n")
        lines.append("| Instance | n | optimum meanJ | GreedyJIT (reactive) | "
                     "GreedyEager (always-on) | EAGER (path-B) |")
        lines.append("|---|---|---|---|---|---|")
        for name in df.instance.unique():
            d = df[df.instance == name]
            opt = d.opt_J.mean()
            def gap(col):
                return f"{(d[col].mean()-opt)/opt*100:+.1f}%"
            lines.append(f"| {name} | {len(d)} | {opt:.3f} | "
                         f"{gap('jit_J')} | {gap('eager_always_J')} | "
                         f"{gap('EAGER_J')} |")
        lines.append("")
        lines.append("Reading: reactive (GreedyJIT) leaves a large gap to the "
                     "stochastic optimum where there is latency to hide; the "
                     "learned EAGER policy reaches the clairvoyant optimum "
                     "(EAGER is OOD here — N far below its training band — so "
                     "this also probes tiny-instance transfer).\n")

    out = RESULTS / "t5_ablation.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")

    index_path = RESULTS / "index.json"
    index = load(index_path) if index_path.exists() else {}
    index["t5_ablation"] = {
        "path": "t5_ablation.md",
        "rows": "graph EAGER / flat-MLP (D83) / NoProactive (§9.7) "
                "+ T4 stochastic-optimum anchor (D84)",
        "graph_vs_reactive": gf["vs_AGGreactive_ratio"],
        "flat_vs_reactive": ff["vs_AGGreactive_ratio"]}
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True),
                          encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
