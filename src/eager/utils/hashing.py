"""Stable trajectory fingerprinting (D19).

Chained SHA-256 over canonical JSON of each transition: (action repr,
integer-only obs snapshot, reward as float.hex(), done flag). float.hex()
sidesteps decimal-repr ambiguity; Python's salted hash() is never used.
Identical (config, seed, action sequence) must produce identical digests
across separate process invocations (Phase 1A acceptance).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


class TrajectoryHasher:
    def __init__(self) -> None:
        self._h = hashlib.sha256()

    def update_reset(self, obs: dict) -> None:
        self._h.update(canonical_json({"reset": obs}).encode("utf-8"))

    def update(self, action: Any, obs: dict, reward: float, done: bool) -> None:
        rec = {
            "a": repr(action),
            "o": obs,
            "r": float(reward).hex(),
            "d": bool(done),
        }
        self._h.update(canonical_json(rec).encode("utf-8"))

    def hexdigest(self) -> str:
        return self._h.hexdigest()
