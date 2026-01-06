"""Surrogate models for meta Bayesian optimization."""

from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


class AbstractSurrogate(nn.Module, ABC):
    """Interface for surrogate models producing predictive distributions."""

    @abstractmethod
    def forward(
        self, context_x: torch.Tensor, context_y: torch.Tensor, query_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return predictive mean and variance for ``query_x`` given observed data."""

    @abstractmethod
    def get_hyperparameters(self) -> dict:
        """Return a dictionary of learnable hyperparameters (for logging)."""


class FeatureExtractor(nn.Module):
    """Simple MLP used as feature extractor g(x)."""

    def __init__(
        self,
        input_dim: int,
        hidden_units: int,
        num_layers: int,
        output_dim: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        act = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }.get(activation.lower(), nn.ReLU)

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_units))
            layers.append(act())
            in_dim = hidden_units
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepRBFKernel(nn.Module):
    """RBF kernel applied on learned feature space."""

    def __init__(
        self,
        initial_alpha: float,
        initial_beta: float,
        initial_eta: float,
        learn_priors: bool = True,
    ) -> None:
        super().__init__()
        self.log_alpha = nn.Parameter(
            torch.tensor(initial_alpha).log(), requires_grad=learn_priors
        )
        self.log_beta = nn.Parameter(
            torch.tensor(initial_beta).log(), requires_grad=learn_priors
        )
        self.log_eta = nn.Parameter(
            torch.tensor(initial_eta).log(), requires_grad=learn_priors
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def beta(self) -> torch.Tensor:
        return self.log_beta.exp()

    @property
    def eta(self) -> torch.Tensor:
        return self.log_eta.exp()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        # Pairwise squared Euclidean distances: ||z1 - z2||^2
        z1_sq = (z1**2).sum(dim=-1, keepdim=True)
        z2_sq = (z2**2).sum(dim=-1).unsqueeze(-2)
        dists = z1_sq + z2_sq - 2 * z1 @ z2.transpose(-2, -1)
        return self.alpha * torch.exp(-0.5 * dists / self.eta)

    def trainable_parameters(self):
        return [
            p for p in [self.log_alpha, self.log_beta, self.log_eta] if p.requires_grad
        ]


class DKLGaussianProcess(AbstractSurrogate):
    """Exact GP with deep kernel for small candidate sets."""

    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        hidden_units: int,
        num_layers: int,
        kernel_params: dict,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.feature_extractor = FeatureExtractor(
            input_dim, hidden_units, num_layers, feature_dim, activation
        )
        self.kernel = DeepRBFKernel(**kernel_params)

    def compute_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        z1 = self.feature_extractor(x1)
        z2 = self.feature_extractor(x2)
        return self.kernel(z1, z2)

    def forward(
        self, context_x: torch.Tensor, context_y: torch.Tensor, query_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Shapes: context_x [Nctx, D], context_y [Nctx], query_x [Nq, D]
        if context_x.numel() == 0:
            raise ValueError("context_x cannot be empty")

        k_xx = self.compute_kernel(context_x, context_x)
        noise = self.kernel.beta * torch.eye(k_xx.size(-1), device=context_x.device)
        k_xx = k_xx + noise

        k_xs = self.compute_kernel(context_x, query_x)  # [Nctx, Nq]
        k_ss = self.compute_kernel(query_x, query_x).diagonal(dim1=-2, dim2=-1)  # [Nq]

        # Solve K^{-1} y via Cholesky
        L = torch.linalg.cholesky(
            k_xx + 1e-6 * torch.eye(k_xx.size(-1), device=context_x.device)
        )
        alpha = torch.cholesky_solve(context_y.unsqueeze(-1), L)
        mu = k_xs.transpose(-2, -1) @ alpha  # [Nq, 1]
        mu = mu.squeeze(-1)

        v = torch.linalg.solve_triangular(L, k_xs, upper=False)
        var = k_ss - (v**2).sum(dim=0)
        var = torch.clamp(var, min=1e-9)
        return mu, var

    def get_hyperparameters(self) -> dict:
        return {
            "alpha": self.kernel.alpha.detach(),
            "beta": self.kernel.beta.detach(),
            "eta": self.kernel.eta.detach(),
        }

    def _nll(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        k_xx = self.compute_kernel(x, x)
        noise = self.kernel.beta * torch.eye(k_xx.size(-1), device=x.device)
        k_xx = k_xx + noise
        L = torch.linalg.cholesky(
            k_xx + 1e-6 * torch.eye(k_xx.size(-1), device=x.device)
        )
        alpha = torch.cholesky_solve(y.unsqueeze(-1), L)
        data_fit = 0.5 * torch.sum(y.unsqueeze(-1) * alpha)
        complexity = torch.sum(torch.log(torch.diag(L)))
        n = x.size(0)
        const = 0.5 * n * torch.log(torch.tensor(2 * torch.pi, device=x.device))
        return data_fit + complexity + const

    def refit_mll(
        self, x: torch.Tensor, y: torch.Tensor, steps: int, lr: float
    ) -> None:
        self.train()
        params = (
            list(self.feature_extractor.parameters())
            + self.kernel.trainable_parameters()
        )
        if not params:
            return
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = self._nll(x, y)
            loss.backward()
            opt.step()
        self.eval()


class RBFGaussianProcess(AbstractSurrogate):
    """Exact GP with plain RBF kernel on raw inputs."""

    def __init__(self, input_dim: int, kernel_params: dict) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.kernel = DeepRBFKernel(**kernel_params)

    def compute_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.kernel(x1, x2)

    def forward(
        self, context_x: torch.Tensor, context_y: torch.Tensor, query_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if context_x.numel() == 0:
            raise ValueError("context_x cannot be empty")

        k_xx = self.compute_kernel(context_x, context_x)
        noise = self.kernel.beta * torch.eye(k_xx.size(-1), device=context_x.device)
        k_xx = k_xx + noise

        k_xs = self.compute_kernel(context_x, query_x)
        k_ss = self.compute_kernel(query_x, query_x).diagonal(dim1=-2, dim2=-1)

        L = torch.linalg.cholesky(
            k_xx + 1e-6 * torch.eye(k_xx.size(-1), device=context_x.device)
        )
        alpha = torch.cholesky_solve(context_y.unsqueeze(-1), L)
        mu = k_xs.transpose(-2, -1) @ alpha
        mu = mu.squeeze(-1)

        v = torch.linalg.solve_triangular(L, k_xs, upper=False)
        var = k_ss - (v**2).sum(dim=0)
        var = torch.clamp(var, min=1e-9)
        return mu, var

    def get_hyperparameters(self) -> dict:
        return {
            "alpha": self.kernel.alpha.detach(),
            "beta": self.kernel.beta.detach(),
            "eta": self.kernel.eta.detach(),
        }

    def _nll(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        k_xx = self.compute_kernel(x, x)
        noise = self.kernel.beta * torch.eye(k_xx.size(-1), device=x.device)
        k_xx = k_xx + noise
        L = torch.linalg.cholesky(
            k_xx + 1e-6 * torch.eye(k_xx.size(-1), device=x.device)
        )
        alpha = torch.cholesky_solve(y.unsqueeze(-1), L)
        data_fit = 0.5 * torch.sum(y.unsqueeze(-1) * alpha)
        complexity = torch.sum(torch.log(torch.diag(L)))
        n = x.size(0)
        const = 0.5 * n * torch.log(torch.tensor(2 * torch.pi, device=x.device))
        return data_fit + complexity + const

    def refit_mll(
        self, x: torch.Tensor, y: torch.Tensor, steps: int, lr: float
    ) -> None:
        self.train()
        params = self.kernel.trainable_parameters()
        if not params:
            return
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = self._nll(x, y)
            loss.backward()
            opt.step()
        self.eval()
