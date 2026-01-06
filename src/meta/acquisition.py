"""Acquisition functions for Bayesian optimization."""

from typing import Tuple

import torch
import torch.distributions as dist


def mutual_information(
    mu: torch.Tensor, var: torch.Tensor, xi: torch.Tensor, nu: float
) -> torch.Tensor:
    """Mutual information acquisition with scalar exploration term xi."""
    value = mu + torch.sqrt(torch.tensor(nu, device=mu.device)) * (
        torch.sqrt(var + xi) - torch.sqrt(xi)
    )
    return value


def upper_confidence_bound(
    mu: torch.Tensor, var: torch.Tensor, nu: float
) -> torch.Tensor:
    return mu + torch.sqrt(torch.tensor(nu, device=mu.device)) * torch.sqrt(var)


def expected_improvement(
    mu: torch.Tensor, var: torch.Tensor, best_y: torch.Tensor
) -> torch.Tensor:
    sigma = torch.sqrt(var)
    z = (mu - best_y) / sigma.clamp_min(1e-9)
    normal = dist.Normal(torch.zeros_like(z), torch.ones_like(z))
    return (mu - best_y) * normal.cdf(z) + sigma * torch.exp(normal.log_prob(z))
