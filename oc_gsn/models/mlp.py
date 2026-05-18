"""Small configurable MLP block."""
from __future__ import annotations
import torch
from torch import nn, Tensor

class MLP(nn.Module):
    """Feed-forward network using Linear, SiLU, and optional LayerNorm."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2, activation: str = "silu", layer_norm: bool = False) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if activation != "silu":
            raise ValueError("Only silu activation is supported in v1")
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        mods: list[nn.Module] = []
        for i in range(len(dims) - 1):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if layer_norm:
                    mods.append(nn.LayerNorm(dims[i + 1]))
                mods.append(nn.SiLU())
        self.net = nn.Sequential(*mods)
    def forward(self, x: Tensor) -> Tensor:
        """Apply the MLP to the final feature dimension."""
        return self.net(x)
