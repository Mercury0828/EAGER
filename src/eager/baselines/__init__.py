"""Baseline policies over the env API (guide §9). All baselines share the env
and eager.env.metrics — no per-method cost reimplementation (guide §12)."""

from .greedy_jit import GreedyJITPolicy, compute_placement
from .partition import balanced_partition, interaction_graph
from .random_prog import RandomProgressivePolicy

__all__ = [
    "GreedyJITPolicy", "compute_placement",
    "balanced_partition", "interaction_graph",
    "RandomProgressivePolicy",
]
