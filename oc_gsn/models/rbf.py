"""Gaussian radial basis functions."""

from __future__ import annotations
import torch
from torch import nn, Tensor


class GaussianRBF(nn.Module):
    """Expand distances with fixed Gaussian radial basis functions."""

    def __init__(self, num_basis: int, cutoff: float, eps: float = 1e-8) -> None:
        super().__init__()
        self.num_basis, self.cutoff, self.eps = num_basis, cutoff, eps
        centers = torch.linspace(0.0, cutoff, num_basis)
        delta = cutoff if num_basis <= 1 else centers[1] - centers[0]
        self.register_buffer("centers", centers)
        self.register_buffer("beta", torch.tensor(1.0 / (float(delta) ** 2 + eps)))

    def forward(self, d: Tensor) -> Tensor:
        """Return features with shape ``d.shape + (num_basis,)``."""
        return torch.exp(
            -self.beta
            * (d.unsqueeze(-1) - self.centers.to(dtype=d.dtype, device=d.device)) ** 2
        )
