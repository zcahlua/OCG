"""Pure PyTorch scatter and graph-pooling utilities."""
from __future__ import annotations
import torch
from torch import Tensor


def scatter_sum(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Sum rows of ``src`` into ``dim_size`` targets."""
    out = src.new_zeros((dim_size,) + tuple(src.shape[1:]))
    if src.numel() > 0:
        out.index_add_(0, index, src)
    return out


def scatter_mean(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Mean rows of ``src`` into ``dim_size`` targets."""
    out = scatter_sum(src, index, dim_size)
    counts = src.new_zeros((dim_size,))
    if index.numel() > 0:
        counts.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    return out / counts.clamp_min(1).view(dim_size, *([1] * (src.dim() - 1)))


def scatter_max(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Max rows of ``src`` into ``dim_size`` targets; empty targets are zero."""
    out = src.new_full((dim_size,) + tuple(src.shape[1:]), -torch.inf)
    for i in range(index.numel()):
        out[index[i]] = torch.maximum(out[index[i]], src[i])
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


def graph_pool(src: Tensor, batch: Tensor, num_graphs: int, reduce: str = "sum") -> Tensor:
    """Pool simplex rows into graph rows by sum or mean."""
    if reduce == "sum":
        return scatter_sum(src, batch, num_graphs)
    if reduce == "mean":
        return scatter_mean(src, batch, num_graphs)
    raise ValueError("reduce must be 'sum' or 'mean'")
