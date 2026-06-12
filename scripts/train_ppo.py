#!/usr/bin/env python
"""Phase 5 PPO run (guide §8.2): fine-tune one training seed from the IL
checkpoint on the stage-A small config; periodically evaluate CRN-paired vs
GreedyJIT on the held-out set and stop once the acceptance criterion holds
(mean J strictly below GreedyJIT AND paired Wilcoxon p < 0.05), else run to
the iteration budget. Writes artifacts/agents/ppo_seed{S}.pt + JSON summary.

Usage (from the repo root):
    python scripts/train_ppo.py --seed 1 --il-ckpt artifacts/agents/il_seed0.pt
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
from eager.train.ppo import PPOConfig, train_ppo

ART = Path("artifacts") / "agents"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--il-ckpt", default="artifacts/agents/il_seed0.pt")
    parser.add_argument("--total-iters", type=int, default=150)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-cases", type=int, default=20)
    parser.add_argument("--eval-env-seeds", type=int, default=20)
    parser.add_argument("--val-env-seeds", type=int, default=4)
    parser.add_argument("--sil-gen-weight", type=float, default=1.0)
    parser.add_argument("--anchor-coef", type=float, default=0.0)
    parser.add_argument("--paired-advantage", action="store_true")
    parser.add_argument("--regime-stage1-iters", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    print(f"device: {device}  ppo training seed: {args.seed}")

    policy = EagerPolicy()
    ckpt = torch.load(args.il_ckpt, map_location="cpu", weights_only=False)
    policy.load_state_dict(ckpt["state_dict"])
    policy.to(device)

    # model selection on a SEPARATE validation case set (seed 778); the
    # held-out acceptance set (seed 777) is touched only once at the end
    # (no selection-on-test contamination, D59)
    val_cases = held_out_cases(args.eval_cases, seed=778)
    val_seeds = list(range(args.val_env_seeds))
    held_cases = held_out_cases(args.eval_cases)          # seed 777
    held_seeds = list(range(args.eval_env_seeds))
    evals = []
    best = {"ratio": float("inf"), "iter": -1, "state": None}

    def on_eval(it: int):
        ev = paired_eval(policy, val_cases, val_seeds, device)
        ev["iter"] = it
        evals.append({k: v for k, v in ev.items() if not k.startswith("j_")})
        # selection candidacy requires a sane win STRUCTURE too (val p<0.4),
        # not just a lucky mean ratio (D66)
        if (ev["ratio"] < best["ratio"] and ev["agent_truncations"] == 0
                and ev["wilcoxon_p_less"] < 0.4):
            best.update(ratio=ev["ratio"], iter=it,
                        state={k: v.detach().cpu().clone()
                               for k, v in policy.state_dict().items()})
        strong = (ev["ratio"] < 0.97 and ev["wilcoxon_p_less"] < 0.02)
        print(f"  val@{it}: ratio {ev['ratio']:.4f} won "
              f"{ev['pairs_won']}/{ev['n_pairs']} "
              f"p={ev['wilcoxon_p_less']:.2e} trunc={ev['agent_truncations']}"
              f"{'  <- strong, stopping' if strong else ''}", flush=True)
        return {"stop": strong}

    cfg = PPOConfig(total_iters=args.total_iters,
                    sil_gen_weight=args.sil_gen_weight,
                    anchor_coef=args.anchor_coef,
                    paired_advantage=args.paired_advantage,
                    regime_stage1_iters=args.regime_stage1_iters)
    anchor_policy = None
    if args.anchor_coef > 0:
        anchor_policy = EagerPolicy()
        anchor_policy.load_state_dict(ckpt["state_dict"])
        anchor_policy.to(device)
        anchor_policy.eval()
    result = train_ppo(policy, cfg, device, seed=args.seed,
                       on_eval=on_eval, eval_every=args.eval_every,
                       anchor_policy=anchor_policy)

    if best["state"] is not None:
        policy.load_state_dict(best["state"])
        print(f"selected best-validation checkpoint from iter {best['iter']} "
              f"(val ratio {best['ratio']:.4f})")

    final = paired_eval(policy, held_cases, held_seeds, device)
    final["iter"] = "final_held_out"
    evals.append({k: v for k, v in final.items() if not k.startswith("j_")})
    accepted = (final["mean_J_agent"] < final["mean_J_greedy"]
                and final["wilcoxon_p_less"] < 0.05
                and final["agent_truncations"] == 0)
    print(f"FINAL seed {args.seed} (held-out, {final['n_pairs']} pairs): "
          f"agent J={final['mean_J_agent']:.2f} vs "
          f"greedy {final['mean_J_greedy']:.2f} ratio={final['ratio']:.4f} "
          f"won {final['pairs_won']}/{final['n_pairs']} "
          f"p={final['wilcoxon_p_less']:.2e} trunc={final['agent_truncations']} "
          f"-> {'ACCEPTED' if accepted else 'NOT ACCEPTED'}")

    ART.mkdir(parents=True, exist_ok=True)
    out = ART / f"ppo_seed{args.seed}.pt"
    torch.save({"state_dict": policy.state_dict(), "seed": args.seed,
                "final_eval": {k: v for k, v in final.items()
                               if not k.startswith("j_")}}, out)
    with open(ART / f"ppo_seed{args.seed}.json", "w", encoding="utf-8") as fh:
        json.dump({"seed": args.seed, "train": {
            "episodes": result["episodes"],
            "truncations": result["truncations"],
            "wall_s": result["wall_s"], "iters": len(result["history"])},
            "best_val": {"iter": best["iter"], "ratio": best["ratio"]},
            "evals": evals,
            "accepted": accepted}, fh, indent=2)
    print(f"checkpoint -> {out}")
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
