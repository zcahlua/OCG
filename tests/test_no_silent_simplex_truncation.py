import pytest, torch
from oc_gsn import build_simplicial_batch


def test_cap_raises():
    z = torch.ones(5, dtype=torch.long)
    pos = torch.randn(5, 3)
    with pytest.raises(RuntimeError):
        build_simplicial_batch(z, pos, cutoff=100.0, max_num_simplices_per_graph=1)
