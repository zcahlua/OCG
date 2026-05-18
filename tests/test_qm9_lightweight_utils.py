import math

import torch

from oc_gsn.training import TargetNormalizer, mae, make_split, mse, rmse


def assert_split_covers_without_overlap(split, n):
    train_idx, val_idx, test_idx = split
    train = set(train_idx.tolist())
    val = set(val_idx.tolist())
    test = set(test_idx.tolist())
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(range(n))


def test_target_normalizer_roundtrip():
    y = torch.tensor([[1.0], [2.0], [4.0]])
    normalizer = TargetNormalizer().fit(y)
    y2 = normalizer.inverse_transform(normalizer.transform(y))
    assert torch.allclose(y2, y)


def test_target_normalizer_std_clamp():
    y = torch.ones(4, 1)
    normalizer = TargetNormalizer().fit(y)
    assert normalizer.std.item() >= 1e-12
    y_norm = normalizer.transform(y)
    y_back = normalizer.inverse_transform(y_norm)
    assert torch.isfinite(y_norm).all()
    assert torch.isfinite(y_back).all()
    assert torch.allclose(y_back, y)


def test_make_split_deterministic():
    split_a = make_split(100, seed=7)
    split_b = make_split(100, seed=7)
    split_c = make_split(100, seed=8)
    assert all(torch.equal(a, b) for a, b in zip(split_a, split_b))
    assert any(not torch.equal(a, c) for a, c in zip(split_a, split_c))
    assert_split_covers_without_overlap(split_a, 100)


def test_make_split_small_dataset_fallback():
    split = make_split(10)
    train_idx, val_idx, test_idx = split
    assert len(train_idx) == 8
    assert len(val_idx) == 1
    assert len(test_idx) == 1
    assert_split_covers_without_overlap(split, 10)


def test_make_split_avoids_empty_test_when_requested_sizes_sum_to_n():
    split = make_split(12, train_size=10, val_size=2)
    train_idx, val_idx, test_idx = split
    assert len(train_idx) == 9
    assert len(val_idx) == 1
    assert len(test_idx) == 2
    assert_split_covers_without_overlap(split, 12)


def test_metrics_known_values():
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 1.0, 5.0])
    assert torch.allclose(mae(pred, target), torch.tensor(1.0))
    assert torch.allclose(mse(pred, target), torch.tensor(5.0 / 3.0))
    assert torch.allclose(rmse(pred, target), torch.tensor(math.sqrt(5.0 / 3.0)))
