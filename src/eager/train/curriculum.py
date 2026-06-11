"""Curriculum (guide §8.2): start stage A (N in [10,30]); unlock stage B
(N in [30,60]) after the agent beats GreedyJIT mean J on the held-out small
set for 3 CONSECUTIVE evaluations. Phase 5 acceptance lives entirely in
stage A; the unlock machinery is exercised by tests and used in Phase 6."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Curriculum:
    stage: str = "A"
    consecutive_wins: int = 0
    unlock_after: int = 3
    history: list = field(default_factory=list)

    def record_eval(self, mean_j_agent: float, mean_j_greedy: float) -> str:
        won = mean_j_agent < mean_j_greedy
        self.consecutive_wins = self.consecutive_wins + 1 if won else 0
        self.history.append({"stage": self.stage, "agent": mean_j_agent,
                             "greedy": mean_j_greedy, "won": won})
        if (self.stage == "A" and self.consecutive_wins >= self.unlock_after):
            self.stage = "B"
            self.consecutive_wins = 0
        return self.stage
