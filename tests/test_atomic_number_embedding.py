import pytest, torch
from oc_gsn import OCGSN


def test_atomic_number_100_and_type_requirement():
    z = torch.tensor([1, 6, 8, 100])
    pos = torch.randn(4, 3)
    y = OCGSN(
        max_atomic_number=100, hidden_dim=8, num_layers=1, rbf_dim=2, cutoff=10.0
    )(z=z, pos=pos)
    assert y.shape == (1, 1)
    with pytest.raises(ValueError):
        OCGSN(z_is_atomic_number=False)
