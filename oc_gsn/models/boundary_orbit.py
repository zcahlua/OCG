"""Order-controlled boundary orbit encoder."""
from __future__ import annotations
import itertools
import torch
from torch import nn, Tensor
from .mlp import MLP

class BoundaryOrbitEncoder(nn.Module):
    """Encode fixed-arity boundary tuples with id or full symmetric orbit pooling."""
    def __init__(self, arity: int, hidden_dim: int, inc_dim: int, out_dim: int, mlp_hidden_dim: int, pi_mode: str = "full", pool: str = "mean") -> None:
        super().__init__(); self.arity=arity; self.pool=pool; self.out_dim=out_dim
        if pi_mode == "id": perms = [tuple(range(arity))]
        elif pi_mode == "full": perms = list(itertools.permutations(range(arity)))
        else: raise ValueError("pi_mode must be 'id' or 'full'")
        self.perms = perms
        self.mlp = MLP(arity * (hidden_dim + inc_dim) + hidden_dim, mlp_hidden_dim, out_dim)
    def forward(self, boundary_h: Tensor, boundary_inc: Tensor, self_h: Tensor) -> Tensor:
        """Compute an upward message for each upper simplex."""
        n = self_h.shape[0]
        if n == 0:
            return self_h.new_empty((0, self.out_dim))
        vals = []
        for p in self.perms:
            bh = boundary_h[:, p, :]
            bi = boundary_inc[:, p, :]
            x = torch.cat([torch.cat([bh, bi], dim=-1).reshape(n, -1), self_h], dim=-1)
            vals.append(self.mlp(x))
        y = torch.stack(vals, dim=1)
        if self.pool == "mean": return y.mean(dim=1)
        if self.pool == "sum": return y.sum(dim=1)
        if self.pool == "max": return y.max(dim=1).values
        raise ValueError("pool must be mean, sum, or max")
