#!/usr/bin/env python
"""Phase 5 IL run (guide §8.1): collect >= 50k GreedyJIT expert transitions
over the stage-A distribution, train behavioral cloning (<= 20 epochs, early
stop), report val top-1, then CRN-paired held-out evaluation vs GreedyJIT.

Writes the checkpoint to artifacts/agents/il_seed{S}.pt and a JSON summary
next to it. Acceptance gates: val top-1 >= 0.90; held-out mean-J ratio
<= 1.05 x GreedyJIT.

Usage (from the repo root):
    python scripts/train_il.py --transitions 50000 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from eager.model.policy import EagerPolicy
from eager.train.distribution import held_out_cases
from eager.train.evaluate import paired_eval
from eager.train.il import collect_expert_dataset, split_episodes, train_il

ART = Path("artifacts") / "agents"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--map-boost", type=float, default=1.0,
                        help="extra CE weight multiplier on Map decisions")
    parser.add_argument("--eval-cases", type=int, default=20)
    parser.add_argument("--eval-env-seeds", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    print(f"device: {device}")

    print("collecting expert dataset ...")
    episodes, stats = collect_expert_dataset(
        min_transitions=args.transitions, seed=args.seed)
    print(f"dataset: {stats}")
    train_data, val_data = split_episodes(episodes, val_frac=0.1,
                                          seed=args.seed + 1)
    print(f"split: {len(train_data)} train / {len(val_data)} val transitions")

    policy = EagerPolicy()
    result = train_il(policy, train_data, val_data, device,
                      max_epochs=args.max_epochs, seed=args.seed,
                      batch_size=args.batch_size, patience=args.patience,
                      boost={0: args.map_boost})
    from eager.train.il import evaluate_breakdown
    breakdown = evaluate_breakdown(policy, val_data, device)
    print(f"val per-type top-1: {breakdown}")
    print(f"best val top-1: {result['best_val_top1']:.4f} "
          f"(epoch {result['best_epoch']})  "
          f"[acceptance >= 0.90: "
          f"{'PASS' if result['best_val_top1'] >= 0.90 else 'FAIL'}]")

    print("held-out CRN-paired evaluation vs GreedyJIT ...")
    cases = held_out_cases(args.eval_cases)
    ev = paired_eval(policy, cases, list(range(args.eval_env_seeds)), device)
    ratio_ok = ev["ratio"] <= 1.05
    print(f"held-out: agent J={ev['mean_J_agent']:.2f} "
          f"greedy J={ev['mean_J_greedy']:.2f} ratio={ev['ratio']:.4f} "
          f"won {ev['pairs_won']}/{ev['n_pairs']} "
          f"trunc={ev['agent_truncations']}  "
          f"[acceptance ratio <= 1.05: {'PASS' if ratio_ok else 'FAIL'}]")

    ART.mkdir(parents=True, exist_ok=True)
    ckpt = ART / f"il_seed{args.seed}.pt"
    torch.save({"state_dict": policy.state_dict(),
                "il": {k: v for k, v in result.items() if k != "history"},
                "dataset": stats}, ckpt)
    with open(ART / f"il_seed{args.seed}.json", "w", encoding="utf-8") as fh:
        json.dump({"dataset": stats, "il": result,
                   "val_breakdown": breakdown, "held_out": {
                       k: v for k, v in ev.items() if not k.startswith("j_")}},
                  fh, indent=2)
    print(f"checkpoint -> {ckpt}")
    ok = result["best_val_top1"] >= 0.90 and ratio_ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
