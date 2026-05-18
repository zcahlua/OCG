import pytest
import torch

from oc_gsn.models.ocgsn import OCGSN


def test_model_forwards_simplex_cap():
    z = torch.tensor([6, 1])
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    batch = torch.zeros(2, dtype=torch.long)

    model = OCGSN(
        hidden_dim=16,
        num_layers=1,
        rbf_dim=4,
        cutoff=5.0,
        max_dim=1,
        max_num_simplices_per_graph=0,
    )

    with pytest.raises(RuntimeError, match="max_num_simplices_per_graph"):
        model(z=z, pos=pos, batch=batch)
