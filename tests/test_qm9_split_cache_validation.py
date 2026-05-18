import pytest
import torch

from oc_gsn.training import make_split
from scripts.train_qm9 import load_or_create_split, split_metadata


def test_load_or_create_split_rejects_stale_metadata(tmp_path):
    split_path = tmp_path / "split_seed0.pt"
    original_split = make_split(n=20, train_size=10, val_size=5, seed=0)
    torch.save(
        {
            "metadata": split_metadata(n=20, train_size=10, val_size=5, seed=0),
            "train_idx": original_split[0],
            "val_idx": original_split[1],
            "test_idx": original_split[2],
        },
        split_path,
    )

    with pytest.raises(ValueError, match="Delete the cached split"):
        load_or_create_split(
            split_path,
            n=20,
            train_size=11,
            val_size=5,
            seed=0,
        )

    loaded_split = load_or_create_split(
        split_path,
        n=20,
        train_size=10,
        val_size=5,
        seed=0,
    )
    assert all(
        torch.equal(original_tensor, loaded_tensor)
        for original_tensor, loaded_tensor in zip(original_split, loaded_split)
    )
