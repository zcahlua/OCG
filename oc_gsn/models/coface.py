"""Permutation-invariant upper-to-lower coface aggregation."""
from __future__ import annotations
import torch
from torch import nn, Tensor
from .mlp import MLP
from .scatter import scatter_sum, scatter_mean

class CofaceAggregator(nn.Module):
    """Aggregate messages from cofaces with learned empty-target convention."""
    def __init__(self, hidden_dim: int, inc_dim: int, out_dim: int, mlp_hidden_dim: int, aggr: str = "sum", learned_empty: bool = True) -> None:
        super().__init__(); self.aggr=aggr; self.learned_empty=learned_empty; self.out_dim=out_dim
        self.mlp = MLP(2 * hidden_dim + inc_dim, mlp_hidden_dim, out_dim)
        self.empty = nn.Parameter(torch.zeros(out_dim)) if learned_empty else None
    def forward(self, coface_h: Tensor, simplex_h: Tensor, inc_feat: Tensor, target_index: Tensor, num_targets: int) -> Tensor:
        """Return aggregated coface messages for ``num_targets`` lower simplices."""
        if num_targets == 0:
            return coface_h.new_empty((0, self.out_dim))
        if coface_h.shape[0] == 0:
            base = coface_h.new_zeros((num_targets, self.out_dim))
            return base + self.empty.view(1, -1) if self.empty is not None else base
        msg = self.mlp(torch.cat([coface_h, simplex_h, inc_feat], dim=-1))
        out = scatter_sum(msg, target_index, num_targets) if self.aggr == "sum" else scatter_mean(msg, target_index, num_targets)
        counts = coface_h.new_zeros((num_targets,)); counts.index_add_(0, target_index, torch.ones_like(target_index, dtype=coface_h.dtype))
        if self.empty is not None:
            out = torch.where((counts == 0).view(-1, 1), self.empty.view(1, -1).to(out), out)
        return out
