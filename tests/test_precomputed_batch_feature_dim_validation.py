import pytest
import torch

from oc_gsn.complex.lift import build_simplicial_batch
from oc_gsn.models.ocgsn import OCGSN


def _tetrahedron_batch(rbf_dim: int):
    z = torch.tensor([6, 1, 1, 1])
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    batch = torch.zeros(4, dtype=torch.long)
    return build_simplicial_batch(
        z=z,
        pos=pos,
        batch=batch,
        cutoff=5.0,
        max_dim=3,
        rbf_dim=rbf_dim,
    )


def test_precomputed_batch_rbf_dim_mismatch_raises_clear_error():
    sb = _tetrahedron_batch(rbf_dim=4)
    model = OCGSN(
        hidden_dim=16,
        num_layers=1,
        rbf_dim=5,
        cutoff=5.0,
        max_dim=3,
    )

    with pytest.raises(ValueError, match="rbf_dim|gamma1_perm|feature dimension"):
        model(batch_data=sb)


def test_precomputed_batch_matching_rbf_dim_runs():
    sb = _tetrahedron_batch(rbf_dim=4)
    model = OCGSN(
        hidden_dim=16,
        num_layers=1,
        rbf_dim=4,
        cutoff=5.0,
        max_dim=3,
    )

    y = model(batch_data=sb)
    assert y.shape == (1, 1)
    assert torch.isfinite(y).all()
