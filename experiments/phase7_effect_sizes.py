#!/usr/bin/env python
"""Phase 7 (review item B): effect sizes + cluster-bootstrap 95% CIs for the
headline comparisons, to report alongside (not instead of) the p-values. With
n=5184 cells everything is "significant"; what matters is the EFFECT SIZE and
its uncertainty. CIs use an INSTANCE-cluster bootstrap (resample instances with
replacement, keep all their cells) so the correlation among cells of the same
instance is respected — a per-cell bootstrap would understate the width.

Reports, for EAGER vs each baseline (CRN-paired on results/phase6_main.parquet)
and for the K=4 zero-shot (results/phase6_zeroshot.parquet): aggregate mean
ratio, per-instance mean ratio (the robust headline) with 95% CI, the paired
win rate, and the median (to expose the 'median tie' honestly). Writes
results/effect_sizes.md.

Usage: python experiments/phase7_effect_sizes.py [--boot 10000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"


def cluster_bootstrap(per_inst_ratios, n_boot, rng):
    """per_inst_ratios: 1-D array of one ratio per instance. Returns the
    (2.5, 97.5) percentile CI of the mean over instance resamples."""
    k = len(per_inst_ratios)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, k, size=k)
        means[b] = per_inst_ratios[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(piv, num, den, cluster_col, n_boot, rng):
    e, a = piv[num].values, piv[den].values
    ratio_agg = e.mean() / a.mean()
    win = float((e < a).mean())
    med = float(np.median(e / a))
    # per-instance ratio = mean(E_i)/mean(A_i), then cluster bootstrap
    g = piv.groupby(level=cluster_col)
    per_inst = (g[num].mean() / g[den].mean()).values
    lo, hi = cluster_bootstrap(per_inst, n_boot, rng)
    return {"mean_ratio": float(ratio_agg), "per_inst_mean": float(per_inst.mean()),
            "ci": (lo, hi), "win_rate": win, "median": med,
            "n_cells": len(e), "n_clusters": len(per_inst)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(12345)

    lines = ["# Effect sizes + cluster-bootstrap 95% CIs (review item B)\n",
             "Ratio = J(EAGER) / J(baseline), lower favors EAGER. Per-instance "
             "mean is the robust headline; CI is an instance-cluster bootstrap "
             f"({args.boot} resamples). 'win rate' = fraction of CRN-paired "
             "cells where EAGER strictly wins; 'median' exposes ties honestly.\n"]

    # --- main grid: EAGER vs each baseline ---
    df = pd.read_parquet(RESULTS / "phase6_main.parquet")
    keys = ["p", "W", "T_cut", "instance", "seed"]
    piv = df.pivot_table(index=keys, columns="method", values="J")
    baselines = [m for m in ["agg", "mhsa_ls", "cloudqc", "greedy_regime_prov",
                             "greedy_eager", "greedy_jit", "greedy_adaptive",
                             "random_prog"] if m in piv.columns]
    lines.append("## T3 main grid — EAGER vs each baseline "
                 f"({len(piv)} CRN-paired cells, {piv.index.get_level_values('instance').nunique()} instances)\n")
    lines.append("| baseline | mean ratio | per-inst mean [95% CI] | win rate | median |")
    lines.append("|---|---|---|---|---|")
    for b in baselines:
        r = compare(piv, "eager", b, "instance", args.boot, rng)
        lines.append(f"| {b} | {r['mean_ratio']:.4f} | "
                     f"{r['per_inst_mean']:.4f} [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}] | "
                     f"{r['win_rate']*100:.1f}% | {r['median']:.4f} |")
    lines.append("")
    agg = compare(piv, "eager", "agg", "instance", args.boot, rng)
    lines.append(f"Headline (EAGER vs AGG, the strongest static): per-instance "
                 f"mean ratio {agg['per_inst_mean']:.4f} "
                 f"(95% CI [{agg['ci'][0]:.4f}, {agg['ci'][1]:.4f}]) — the CI "
                 f"excludes 1.0, so the ~{(1-agg['per_inst_mean'])*100:.1f}% "
                 f"improvement is robust; but the median is {agg['median']:.4f} "
                 f"and the win rate {agg['win_rate']*100:.0f}% — the gain is "
                 "CONCENTRATED (large wins where provisioning binds, ties in the "
                 "comfortable regime), NOT uniform dominance. The regime map (F2) "
                 "is the honest framing.\n")

    # --- zero-shot K=4: EAGER vs AGG-reactive (cluster = circuit) ---
    zs = RESULTS / "phase6_zeroshot.parquet"
    if zs.exists():
        z = pd.read_parquet(zs)
        z = z.assign(r=z.eager_J / z.reactive_J)
        per_c = z.groupby("circuit").apply(
            lambda d: d.eager_J.mean() / d.reactive_J.mean()).values
        lo, hi = cluster_bootstrap(per_c, args.boot, rng)
        lines.append("## Zero-shot K=4 — EAGER vs AGG-reactive "
                     f"({len(z)} cells, {z.circuit.nunique()} QASMBench circuits)\n")
        lines.append(f"per-circuit mean ratio {per_c.mean():.4f} "
                     f"(95% CI [{lo:.4f}, {hi:.4f}]); aggregate "
                     f"{z.eager_J.mean()/z.reactive_J.mean():.4f}; win rate "
                     f"{(z.eager_J<z.reactive_J).mean()*100:.0f}%; median "
                     f"{np.median(z.r):.4f}.\n")

    out = RESULTS / "effect_sizes.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")

    idx_path = RESULTS / "index.json"
    index = json.loads(idx_path.read_text(encoding="utf-8")) if idx_path.exists() else {}
    index["effect_sizes"] = {"path": "effect_sizes.md", "boot": args.boot,
                             "method": "instance-cluster bootstrap 95% CI",
                             "eager_vs_agg_per_inst": agg["per_inst_mean"],
                             "eager_vs_agg_ci": list(agg["ci"]),
                             "eager_vs_agg_win_rate": agg["win_rate"],
                             "eager_vs_agg_median": agg["median"]}
    idx_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
