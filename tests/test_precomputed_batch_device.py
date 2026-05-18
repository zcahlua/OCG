import pytest
import torch

from oc_gsn.complex.lift import build_simplicial_batch
from oc_gsn.models.ocgsn import OCGSN


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_precomputed_batch_is_moved_to_model_device():
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

    model = OCGSN(
        hidden_dim=16,
        num_layers=1,
        rbf_dim=4,
        cutoff=5.0,
        max_dim=3,
    ).cuda()

    y, aux = model(batch_data=sb, return_aux=True)

    assert y.device.type == "cuda"
    assert aux["h0"].device.type == "cuda"
    assert aux["h1"].device.type == "cuda"
    assert aux["h2"].device.type == "cuda"
    assert aux["h3"].device.type == "cuda"
    assert aux["batch"].z.device.type == "cuda"
