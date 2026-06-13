"""Baseline policies over the env API (guide §9). All baselines share the env
and eager.env.metrics — no per-method cost reimplementation (guide §12)."""

from .agg import detect_bursts, make_agg_method, transform_instance
from .cloudqc import CloudQCPolicy
from .greedy_jit import (
    GreedyAdaptivePolicy,
    GreedyEagerPolicy,
    GreedyJITPolicy,
    GreedyRegimeProvisionPolicy,
    compute_placement,
)
from .mhsa import make_mhsa_policy, mhsa_placement
from .partition import balanced_partition, interaction_graph
from .random_prog import RandomProgressivePolicy

__all__ = [
    "GreedyJITPolicy", "GreedyEagerPolicy", "GreedyAdaptivePolicy",
    "GreedyRegimeProvisionPolicy", "CloudQCPolicy", "compute_placement",
    "balanced_partition", "interaction_graph",
    "RandomProgressivePolicy",
    "mhsa_placement", "make_mhsa_policy",
    "detect_bursts", "transform_instance", "make_agg_method",
]
