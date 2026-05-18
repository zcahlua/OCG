import dataclasses

import pytest
import torch

from oc_gsn.complex.lift import build_simplicial_batch
from oc_gsn.models.ocgsn import OCGSN


def test_precomputed_batch_malformed_gamma2_perm_shape_raises():
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
    sb = build_simplicial_batch(
        z=z,
        pos=pos,
        batch=batch,
        cutoff=5.0,
        max_dim=3,
        rbf_dim=4,
    )
    sb_bad = dataclasses.replace(
        sb,
        gamma2_perm=torch.empty((sb.num_triangles, 5, sb.gamma2_perm.shape[-1])),
    )
    model = OCGSN(
        hidden_dim=16,
        num_layers=1,
        rbf_dim=4,
        cutoff=5.0,
        max_dim=3,
    )

    with pytest.raises(ValueError, match="gamma2_perm"):
        model(batch_data=sb_bad)
