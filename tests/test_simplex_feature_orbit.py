import torch
from oc_gsn.models.simplex_feature_orbit import SimplexFeatureOrbitEncoder


def test_simplex_feature_orbit_reorder_invariant():
    torch.manual_seed(1)
    enc = SimplexFeatureOrbitEncoder(11, 9, 16)
    x = torch.randn(7, 6, 11)
    p = torch.tensor([3, 0, 5, 1, 4, 2])
    assert (enc(x) - enc(x[:, p])).abs().max() < 1e-5
