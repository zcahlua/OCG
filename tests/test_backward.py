import torch
from oc_gsn import OCGSN

def test_backward_finite():
    z=torch.tensor([6,1,1,1,8]); pos=torch.randn(5,3,requires_grad=True)
    model=OCGSN(hidden_dim=12,num_layers=1,rbf_dim=3,cutoff=10.0)
    loss=model(z=z,pos=pos).sum(); loss.backward()
    assert pos.grad is not None and torch.isfinite(pos.grad).all()
    nonzero=False
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(); nonzero = nonzero or bool((p.grad.abs()>0).any())
    assert nonzero
