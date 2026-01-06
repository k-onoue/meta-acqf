"""Gym-like environment for meta Bayesian optimization tasks."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class StepResult:
    observation: Dict[str, torch.Tensor]
    reward: float
    done: bool
    info: Dict[str, object]


class MetaBOEnvironment:
    def __init__(
        self,
        features: torch.Tensor,
        rewards: torch.Tensor,
        initial_pool_size: int,
        query_budget: int,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if features.size(0) != rewards.size(0):
            raise ValueError("features and rewards must have matching first dimension")
        self.X = features
        self.y = rewards
        self.initial_pool_size = min(initial_pool_size, features.size(0))
        self.query_budget = min(query_budget, features.size(0) - self.initial_pool_size)
        self.generator = generator

        self.observed: List[int] = []
        self.available: List[int] = []
        self.t: int = 0
        self.global_best = float(self.y.max().item())
        self.reset()

    def reset(self) -> Dict[str, torch.Tensor]:
        num_points = self.X.size(0)
        perm = torch.randperm(num_points, generator=self.generator)
        self.observed = perm[: self.initial_pool_size].tolist()
        self.available = perm[self.initial_pool_size :].tolist()
        self.t = 0
        return self._observation()

    def _observation(self) -> Dict[str, torch.Tensor]:
        ctx_x = self.X[self.observed]
        ctx_y = self.y[self.observed]
        cand_x = self.X[self.available]
        return {"context_x": ctx_x, "context_y": ctx_y, "candidates_x": cand_x}

    def step(self, candidate_index: int) -> StepResult:
        if candidate_index >= len(self.available):
            raise IndexError("candidate_index out of range of available set")

        chosen_idx = self.available.pop(candidate_index)
        self.observed.append(chosen_idx)
        self.t += 1

        obs = self._observation()
        best_observed = float(obs["context_y"].max().item())
        gap = self.global_best - best_observed
        reward = -gap
        done = self.t >= self.query_budget or len(self.available) == 0

        return StepResult(
            observation=obs, reward=reward, done=done, info={"gap": gap, "step": self.t}
        )

    @property
    def context(self) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = self._observation()
        return obs["context_x"], obs["context_y"]

    @property
    def candidates(self) -> torch.Tensor:
        return self.X[self.available]
