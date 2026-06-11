"""Random-Progressive — the honest lower bound (guide §9.5).

Uniform over valid NON-ADVANCE actions; ADVANCE only when nothing else is
valid (pure uniform over all actions stalls forever). The policy RNG (PCG64,
seeded per episode) is strictly separate from the env's CRN engine, so two
methods evaluated under the same env seed share generation luck (CRN pairing)
while the policy's own randomness stays independent.
"""

from __future__ import annotations

import numpy as np

from ..env.actions import Action
from ..env.env import EagerEnv


class RandomProgressivePolicy:
    name = "random_progressive"

    def __init__(self, policy_seed: int):
        self.policy_seed = policy_seed
        self.rng = np.random.default_rng(policy_seed)

    def __call__(self, env: EagerEnv) -> Action:
        acts = env.valid_actions()      # fixed enumeration order, ADVANCE last
        if len(acts) > 1:
            return acts[int(self.rng.integers(len(acts) - 1))]
        return acts[-1]
