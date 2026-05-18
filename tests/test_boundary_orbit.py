import torch
from oc_gsn.models.boundary_orbit import BoundaryOrbitEncoder


def test_boundary_orbit_full_reorder_invariant():
    torch.manual_seed(0)
    enc = BoundaryOrbitEncoder(3, 8, 2, 8, 16, pi_mode="full")
    h = torch.randn(5, 3, 8)
    inc = torch.randn(5, 3, 2)
    self_h = torch.randn(5, 8)
    y = enc(h, inc, self_h)
    p = torch.tensor([2, 0, 1])
    assert (y - enc(h[:, p], inc[:, p], self_h)).abs().max() < 1e-5
