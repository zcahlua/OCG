"""Lightweight training utilities for small OC-GSN benchmarks."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor


class TargetNormalizer:
    """Minimal single-target mean/std normalizer."""

    def __init__(self) -> None:
        self.mean = torch.zeros(1, 1)
        self.std = torch.ones(1, 1)

    def fit(self, y_train: Tensor) -> "TargetNormalizer":
        y = self._as_column(y_train).detach()
        self.mean = y.mean().view(1, 1)
        self.std = y.std(unbiased=False).clamp_min(1.000001e-12).view(1, 1)
        return self

    def transform(self, y: Tensor) -> Tensor:
        y_col = self._as_column(y)
        return (y_col - self.mean.to(y_col.device)) / self.std.to(y_col.device)

    def inverse_transform(self, y_norm: Tensor) -> Tensor:
        y_col = self._as_column(y_norm)
        return y_col * self.std.to(y_col.device) + self.mean.to(y_col.device)

    def state_dict(self) -> dict[str, Tensor]:
        return {"mean": self.mean.detach().clone(), "std": self.std.detach().clone()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean = torch.as_tensor(state["mean"]).view(1, 1)
        self.std = torch.as_tensor(state["std"]).clamp_min(1.000001e-12).view(1, 1)

    def to(self, device: torch.device | str) -> "TargetNormalizer":
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    @staticmethod
    def _as_column(y: Tensor) -> Tensor:
        if y.ndim == 1:
            return y.view(-1, 1)
        if y.ndim == 2 and y.shape[1] == 1:
            return y
        if y.numel() == y.shape[0]:
            return y.reshape(-1, 1)
        raise ValueError("TargetNormalizer expects a single target shaped [N] or [N, 1].")


def make_split(
    n: int,
    train_size: int = 110000,
    val_size: int = 10000,
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Create a deterministic train/validation/test split covering all indices."""
    if n < 0:
        raise ValueError("n must be non-negative")

    if n > train_size + val_size:
        n_train = train_size
        n_val = val_size
    else:
        n_train = math.floor(0.8 * n)
        n_val = math.floor(0.1 * n)
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    if n >= 3 and n - n_train - n_val == 0:
        if n_val > 0:
            n_val -= 1
        else:
            n_train -= 1

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return train_idx, val_idx, test_idx


def _flatten_pair(pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    return pred.view(-1), target.view(-1)


def mae(pred: Tensor, target: Tensor) -> Tensor:
    pred_f, target_f = _flatten_pair(pred, target)
    return (pred_f - target_f).abs().mean()


def mse(pred: Tensor, target: Tensor) -> Tensor:
    pred_f, target_f = _flatten_pair(pred, target)
    return ((pred_f - target_f) ** 2).mean()


def rmse(pred: Tensor, target: Tensor) -> Tensor:
    return torch.sqrt(mse(pred, target))


def num_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
