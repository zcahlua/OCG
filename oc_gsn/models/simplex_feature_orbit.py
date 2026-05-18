"""Orbit pooling for permutation-expanded initial simplex features."""

from __future__ import annotations
import torch
from torch import nn, Tensor
from .mlp import MLP


class SimplexFeatureOrbitEncoder(nn.Module):
    """Encode ordered raw simplex features and pool over all vertex permutations."""

    def __init__(
        self,
        raw_dim: int,
        out_dim: int,
        mlp_hidden_dim: int,
        num_layers: int = 2,
        pool: str = "mean",
    ) -> None:
        super().__init__()
        self.pool = pool
        self.out_dim = out_dim
        self.mlp = MLP(raw_dim, mlp_hidden_dim, out_dim, num_layers=num_layers)

    def forward(self, raw_perm: Tensor) -> Tensor:
        """Return pooled features of shape [num_simplices, out_dim]."""
        n, p, d = raw_perm.shape
        y = self.mlp(raw_perm.reshape(n * p, d)).reshape(n, p, self.out_dim)
        if self.pool == "mean":
            return y.mean(dim=1)
        if self.pool == "sum":
            return y.sum(dim=1)
        if self.pool == "max":
            return y.max(dim=1).values
        raise ValueError("pool must be mean, sum, or max")
