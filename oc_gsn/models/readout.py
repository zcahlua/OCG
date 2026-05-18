"""Graph-level OC-GSN readout."""

from __future__ import annotations
import torch
from torch import nn, Tensor
from .mlp import MLP
from .scatter import graph_pool


class OCGSNReadout(nn.Module):
    """Pool each simplex dimension per graph, inserting learned per-dimension nulls for empty graphs."""

    def __init__(
        self, hidden_dim: int, target_dim: int = 1, readout_pool: str = "sum"
    ) -> None:
        super().__init__()
        self.readout_pool = readout_pool
        h = hidden_dim
        self.mlps = nn.ModuleList([MLP(h, h, h) for _ in range(4)])
        self.nulls = nn.ParameterList([nn.Parameter(torch.zeros(h)) for _ in range(4)])
        self.graph_mlp = MLP(4 * h, h, h)
        self.head = nn.Linear(h, target_dim)

    def _pool_dim(self, x: Tensor, b: Tensor, num_graphs: int, dim: int) -> Tensor:
        y = (
            self.mlps[dim](x)
            if x.shape[0]
            else x.new_empty((0, self.nulls[dim].numel()))
        )
        pooled = graph_pool(y, b, num_graphs, self.readout_pool)
        counts = x.new_zeros((num_graphs,))
        if b.numel():
            counts.index_add_(0, b, torch.ones_like(b, dtype=x.dtype))
        return torch.where(
            (counts == 0).view(-1, 1), self.nulls[dim].view(1, -1).to(pooled), pooled
        )

    def forward(
        self,
        h0: Tensor,
        h1: Tensor,
        h2: Tensor,
        h3: Tensor,
        batch0: Tensor,
        batch1: Tensor,
        batch2: Tensor,
        batch3: Tensor,
        num_graphs: int,
    ) -> Tensor:
        """Return graph predictions with shape [num_graphs, target_dim]."""
        r = [
            self._pool_dim(h, b, num_graphs, i)
            for i, (h, b) in enumerate(
                [(h0, batch0), (h1, batch1), (h2, batch2), (h3, batch3)]
            )
        ]
        return self.head(self.graph_mlp(torch.cat(r, dim=-1)))
