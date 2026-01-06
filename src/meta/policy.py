"""Acquisition policy that wraps a surrogate and acquisition function."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from meta.acquisition import (expected_improvement, mutual_information,
                              upper_confidence_bound)
from meta.config import AcquisitionConfig
from meta.surrogate import AbstractSurrogate


class AcquisitionPolicy(nn.Module):
    def __init__(
        self, surrogate: AbstractSurrogate, acq_config: AcquisitionConfig
    ) -> None:
        super().__init__()
        self.surrogate = surrogate
        self.acq_config = acq_config

    @property
    def use_softmax(self) -> bool:
        return bool(self.acq_config.use_softmax)

    def forward(
        self,
        context_x: torch.Tensor,
        context_y: torch.Tensor,
        candidates_x: torch.Tensor,
        xi: Optional[torch.Tensor] = None,
        best_y: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor
    ]:
        mu, var = self.surrogate(context_x, context_y, candidates_x)

        name = self.acq_config.name.upper()
        if name == "MI":
            xi = xi if xi is not None else torch.tensor(1e-6, device=mu.device)
            scores = mutual_information(mu, var, xi, self.acq_config.nu)
        elif name == "UCB":
            scores = upper_confidence_bound(mu, var, self.acq_config.nu)
        elif name == "EI":
            best_y = best_y if best_y is not None else context_y.max()
            scores = expected_improvement(mu, var, best_y)
        else:
            raise ValueError(f"Unsupported acquisition type: {self.acq_config.name}")

        if self.use_softmax:
            probs = F.softmax(scores, dim=-1)
        else:
            probs = None
        return scores, probs, xi, var
