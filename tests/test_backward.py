import torch

from oc_gsn import OCGSN


def test_backward_finite():
    z = torch.tensor([6, 1, 1, 1, 8])
    pos = torch.randn(5, 3, requires_grad=True)
    model = OCGSN(hidden_dim=12, num_layers=1, rbf_dim=3, cutoff=10.0)
    loss = model(z=z, pos=pos).sum()
    loss.backward()

    if pos.requires_grad:
        assert pos.grad is not None
        assert torch.isfinite(pos.grad).all()

    any_nonzero_grad = False
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
            if p.grad.abs().sum() > 0:
                any_nonzero_grad = True
    assert any_nonzero_grad
