import pytest
import torch

from oc_gsn.training import make_split
from scripts.train_qm9 import load_or_create_split, split_metadata


def write_split(path, *, n=20, train_size=10, val_size=5, seed=0):
    split = make_split(n=n, train_size=train_size, val_size=val_size, seed=seed)
    torch.save(
        {
            "metadata": split_metadata(
                n=n,
                train_size=train_size,
                val_size=val_size,
                seed=seed,
            ),
            "train_idx": split[0],
            "val_idx": split[1],
            "test_idx": split[2],
        },
        path,
    )
    return split


def assert_split_equal(left, right):
    assert all(
        torch.equal(left_tensor, right_tensor)
        for left_tensor, right_tensor in zip(left, right)
    )


def test_load_or_create_split_loads_matching_metadata(tmp_path):
    split_path = tmp_path / "split_seed0.pt"
    original_split = write_split(
        split_path,
        n=20,
        train_size=10,
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

    assert_split_equal(original_split, loaded_split)


@pytest.mark.parametrize(
    "overrides",
    [
        {"train_size": 11},
        {"val_size": 6},
        {"seed": 1},
        {"n": 21},
    ],
)
def test_load_or_create_split_rejects_stale_metadata(tmp_path, overrides):
    split_path = tmp_path / "split_seed0.pt"
    write_split(split_path, n=20, train_size=10, val_size=5, seed=0)
    kwargs = {"n": 20, "train_size": 10, "val_size": 5, "seed": 0}
    kwargs.update(overrides)

    with pytest.raises(
        ValueError,
        match=(
            "cached split was created with different "
            "n/train_size/val_size/seed"
        ),
    ):
        load_or_create_split(split_path, **kwargs)
