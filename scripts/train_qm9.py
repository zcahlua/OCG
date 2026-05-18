#!/usr/bin/env python
"""Extremely lightweight QM9 training script for OC-GSN."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oc_gsn import OCGSN  # noqa: E402
from oc_gsn.training import (  # noqa: E402
    TargetNormalizer,
    mae,
    make_split,
    mse,
    num_parameters,
    rmse,
)

_PYG_IMPORT_ERROR = (
    "QM9 training requires torch_geometric. "
    "Install it with pip install -r requirements-qm9.txt "
    "or follow the official PyTorch Geometric installation instructions."
)
_TOO_MANY_SIMPLICES_MESSAGE = (
    "Simplicial complex exceeded max_num_simplices_per_graph. Try reducing cutoff, "
    "setting max_neighbors, reducing max_dim, or increasing the cap."
)


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")


def require_pyg() -> None:
    try:
        from torch_geometric.datasets import QM9 as _QM9  # noqa: F401
        from torch_geometric.loader import DataLoader as _DataLoader  # noqa: F401
    except ImportError as exc:
        raise ImportError(_PYG_IMPORT_ERROR) from exc


def load_qm9_dataset(data_root: str):
    require_pyg()
    from torch_geometric.datasets import QM9

    return QM9(root=data_root)


def get_pyg_dataloader():
    require_pyg()
    from torch_geometric.loader import DataLoader

    return DataLoader


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/QM9")
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=110000)
    parser.add_argument("--val-size", type=int, default=10000)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--limit-test-batches", type=int, default=None)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--rbf-dim", type=int, default=32)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--max-neighbors", type=int, default=8)
    parser.add_argument("--include-bonds", type=str2bool, default=True)
    parser.add_argument("--max-dim", type=int, default=2)
    parser.add_argument("--pi-mode", default="full")
    parser.add_argument("--orbit-pool", default="mean")
    parser.add_argument("--readout-pool", default="sum")
    parser.add_argument("--max-num-simplices-per-graph", type=int, default=None)
    parser.add_argument("--tie-tol", type=float, default=1e-8)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--output-dir", default="runs/qm9_light")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--ablation-max-dims", type=str2bool, default=False)
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def collect_targets(dataset) -> torch.Tensor:
    y_all = None
    data = getattr(dataset, "data", None)
    if data is not None and hasattr(data, "y"):
        candidate = getattr(data, "y")
        if isinstance(candidate, torch.Tensor) and candidate.ndim == 2:
            y_all = candidate
    if y_all is None:
        y_all = torch.cat(
            [dataset[int(i)].y.view(1, -1) for i in range(len(dataset))], dim=0
        )
    if y_all.ndim == 1:
        y_all = y_all.view(-1, 1)
    if y_all.ndim != 2:
        raise ValueError(
            f"Expected QM9 targets to have shape [N, T], got {tuple(y_all.shape)}"
        )
    return y_all


def validate_target(y_all: torch.Tensor, target: int) -> None:
    if not 0 <= target < y_all.shape[1]:
        raise ValueError(
            f"target must satisfy 0 <= target < {y_all.shape[1]}, got {target}"
        )


def batch_target(batch, target: int) -> torch.Tensor:
    y = batch.y
    if y.ndim == 1:
        y = y.view(1, -1)
    return y[:, target].view(-1, 1)


def forward_model(model, batch):
    try:
        return model(
            z=batch.z,
            pos=batch.pos,
            batch=batch.batch,
            bond_edge_index=batch.edge_index,
        )
    except RuntimeError as exc:
        if "max_num_simplices_per_graph" in str(exc):
            print(_TOO_MANY_SIMPLICES_MESSAGE)
        raise


def train_one_epoch(
    model,
    loader,
    optimizer,
    normalizer: TargetNormalizer,
    target: int,
    device,
    grad_clip: float = 5.0,
    limit_batches: Optional[int] = None,
) -> dict[str, float]:
    model.train()
    normalizer.to(device)
    total_loss = 0.0
    total_items = 0
    for step, batch in enumerate(loader):
        if limit_batches is not None and step >= limit_batches:
            break
        batch = batch.to(device)
        y = batch_target(batch, target)
        y_norm = normalizer.transform(y)
        pred_norm = forward_model(model, batch)
        loss = F.mse_loss(pred_norm, y_norm)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        items = int(y.shape[0])
        total_loss += float(loss.detach().cpu()) * items
        total_items += items
    return {"loss": total_loss / max(total_items, 1)}


def evaluate(
    model,
    loader,
    normalizer: TargetNormalizer,
    target: int,
    device,
    limit_batches: Optional[int] = None,
) -> dict[str, float]:
    model.eval()
    normalizer.to(device)
    preds = []
    targets = []
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if limit_batches is not None and step >= limit_batches:
                break
            batch = batch.to(device)
            y = batch_target(batch, target)
            pred_norm = forward_model(model, batch)
            pred = normalizer.inverse_transform(pred_norm)
            preds.append(pred.detach().cpu())
            targets.append(y.detach().cpu())
    if not preds:
        return {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan")}
    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)
    return {
        "mae": float(mae(pred_all, target_all)),
        "mse": float(mse(pred_all, target_all)),
        "rmse": float(rmse(pred_all, target_all)),
    }


def make_model(args: argparse.Namespace, max_dim: int) -> OCGSN:
    return OCGSN(
        max_atomic_number=100,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        rbf_dim=args.rbf_dim,
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
        include_bonds=args.include_bonds,
        max_dim=max_dim,
        pi_mode=args.pi_mode,
        orbit_pool=args.orbit_pool,
        readout_pool=args.readout_pool,
        target_dim=1,
        use_chirality=False,
        z_is_atomic_number=True,
        max_num_simplices_per_graph=args.max_num_simplices_per_graph,
        tie_tol=args.tie_tol,
    )


def json_safe_config(args: argparse.Namespace, max_dim: int) -> dict[str, Any]:
    config = vars(args).copy()
    config["max_dim"] = max_dim
    return config


def save_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    config: dict[str, Any],
    normalizer: TargetNormalizer,
    metrics: dict[str, Any],
    model_state_dict: Optional[dict[str, torch.Tensor]] = None,
) -> None:
    torch.save(
        {
            "model_state_dict": (
                model_state_dict if model_state_dict is not None else model.state_dict()
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "normalizer": normalizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def run_experiment(
    args: argparse.Namespace,
    *,
    max_dim: int,
    normalizer: TargetNormalizer,
    loaders: tuple[Any, Any, Any],
    device: torch.device,
    run_dir: Path,
) -> dict[str, Any]:
    train_loader, val_loader, test_loader = loaders
    run_dir.mkdir(parents=True, exist_ok=True)
    config = json_safe_config(args, max_dim)
    save_json(run_dir / "config.json", config)

    model = make_model(args, max_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_state = None
    best_optimizer_state = None
    best_epoch = -1
    best_val = {"mae": float("inf"), "mse": float("inf"), "rmse": float("inf")}
    train_loss_at_best = float("nan")
    best_val_mae = float("inf")
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            normalizer,
            args.target,
            device,
            grad_clip=args.grad_clip,
            limit_batches=args.limit_train_batches,
        )
        val_stats = evaluate(
            model,
            val_loader,
            normalizer,
            args.target,
            device,
            limit_batches=args.limit_val_batches,
        )
        if val_stats["mae"] < best_val_mae:
            best_val_mae = val_stats["mae"]
            best_val = val_stats.copy()
            best_epoch = epoch
            train_loss_at_best = train_stats["loss"]
            best_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        print(
            f"epoch={epoch} train_loss_norm={train_stats['loss']:.6g} "
            f"val_mae={val_stats['mae']:.6g} best_val_mae={best_val_mae:.6g}"
        )

    last_metrics = {
        "target": args.target,
        "max_dim": max_dim,
        "last_epoch": args.epochs,
    }
    save_checkpoint(
        run_dir / "last_model.pt",
        model,
        optimizer,
        args.epochs,
        config,
        normalizer,
        last_metrics,
    )

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        best_epoch = 0
        best_val = {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan")}
        train_loss_at_best = float("nan")

    model.load_state_dict(best_state)
    test_stats = evaluate(
        model,
        test_loader,
        normalizer,
        args.target,
        device,
        limit_batches=args.limit_test_batches,
    )
    elapsed = time.time() - started
    metrics = {
        "target": args.target,
        "max_dim": max_dim,
        "pi_mode": args.pi_mode,
        "cutoff": args.cutoff,
        "max_neighbors": args.max_neighbors,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "rbf_dim": args.rbf_dim,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_mae": best_val["mae"],
        "test_mae_at_best_val": test_stats["mae"],
        "best_val_mse": best_val["mse"],
        "train_loss_at_best": train_loss_at_best,
        "num_parameters": num_parameters(model),
        "elapsed_seconds": elapsed,
        "best_val_rmse": best_val["rmse"],
        "test_mse_at_best_val": test_stats["mse"],
        "test_rmse_at_best_val": test_stats["rmse"],
    }
    save_json(run_dir / "metrics.json", metrics)
    if best_optimizer_state is not None:
        optimizer.load_state_dict(best_optimizer_state)
    best_ckpt_metrics = {
        "validation": best_val,
        "test_at_best_val": test_stats,
        **metrics,
    }
    save_checkpoint(
        run_dir / "best_model.pt",
        model,
        optimizer,
        best_epoch,
        config,
        normalizer,
        best_ckpt_metrics,
        model_state_dict=best_state,
    )
    print(
        f"final max_dim={max_dim} best_val_mae={metrics['best_val_mae']:.6g} "
        f"test_mae_at_best_val={metrics['test_mae_at_best_val']:.6g}"
    )
    return metrics


def split_metadata(n: int, train_size: int, val_size: int, seed: int) -> dict[str, int]:
    return {
        "n": int(n),
        "train_size": int(train_size),
        "val_size": int(val_size),
        "seed": int(seed),
    }


def validate_split_metadata(
    split_path: Path,
    state: dict[str, Any],
    expected_metadata: dict[str, int],
) -> None:
    cached_metadata = state.get("metadata")
    if cached_metadata != expected_metadata:
        raise ValueError(
            "cached split was created with different n/train_size/val_size/seed; "
            "delete the cached split or use a different output directory."
        )


def load_or_create_split(
    split_path: Path, n: int, train_size: int, val_size: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    metadata = split_metadata(n, train_size, val_size, seed)
    if split_path.exists():
        state = torch.load(split_path, map_location="cpu")
        validate_split_metadata(split_path, state, metadata)
        return state["train_idx"], state["val_idx"], state["test_idx"]

    split = make_split(n, train_size=train_size, val_size=val_size, seed=seed)
    torch.save(
        {
            "metadata": metadata,
            "train_idx": split[0],
            "val_idx": split[1],
            "test_idx": split[2],
        },
        split_path,
    )
    return split


def make_run_name(run_name: Optional[str]) -> str:
    if run_name:
        return run_name
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset = load_qm9_dataset(args.data_root)

    target_dir = Path(args.output_dir) / f"target_{args.target}"
    target_dir.mkdir(parents=True, exist_ok=True)
    split_path = target_dir / f"split_seed{args.seed}.pt"
    train_idx, val_idx, test_idx = load_or_create_split(
        split_path, len(dataset), args.train_size, args.val_size, args.seed
    )

    y_all = collect_targets(dataset)
    validate_target(y_all, args.target)
    y_train = y_all[train_idx, args.target].view(-1, 1)
    normalizer = TargetNormalizer().fit(y_train).to(device)

    DataLoader = get_pyg_dataloader()
    train_subset = Subset(dataset, train_idx.tolist())
    val_subset = Subset(dataset, val_idx.tolist())
    test_subset = Subset(dataset, test_idx.tolist())
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    loaders = (train_loader, val_loader, test_loader)

    print(f"device={device} split={split_path}")
    print(
        f"split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
    )

    if args.ablation_max_dims:
        print("Warning: max_dim=3 may be slower because tetrahedra are included.")
        results = []
        for max_dim in (0, 1, 2, 3):
            args_for_run = copy.copy(args)
            args_for_run.max_dim = max_dim
            run_dir = target_dir / f"max_dim_{max_dim}" / make_run_name(args.run_name)
            results.append(
                run_experiment(
                    args_for_run,
                    max_dim=max_dim,
                    normalizer=normalizer,
                    loaders=loaders,
                    device=device,
                    run_dir=run_dir,
                )
            )
        print("max_dim | best_val_mae | test_mae_at_best_val | best_epoch")
        for row in results:
            print(
                f"{row['max_dim']} | {row['best_val_mae']:.6g} | "
                f"{row['test_mae_at_best_val']:.6g} | {row['best_epoch']}"
            )
    else:
        run_dir = target_dir / make_run_name(args.run_name)
        run_experiment(
            args,
            max_dim=args.max_dim,
            normalizer=normalizer,
            loaders=loaders,
            device=device,
            run_dir=run_dir,
        )


if __name__ == "__main__":
    main()
