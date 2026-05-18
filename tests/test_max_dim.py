import pytest, torch
from oc_gsn import OCGSN, build_simplicial_batch

def test_max_dim_empty_shapes_and_forward():
    z=torch.tensor([6,1,1,1]); pos=torch.tensor([[0.,0,0],[1,0,0],[0,1,0],[0,0,1.]])
    for md in range(4):
        b=build_simplicial_batch(z,pos,cutoff=2.0,max_dim=md,rbf_dim=3)
        if md==0: assert b.k1.shape==(0,2) and b.k2.shape==(0,3) and b.k3.shape==(0,4)
        if md==1: assert b.k2.shape==(0,3) and b.k3.shape==(0,4)
        if md==2: assert b.k3.shape==(0,4)
        y=OCGSN(hidden_dim=8,num_layers=1,rbf_dim=3,cutoff=2.0,max_dim=md)(b)
        assert torch.isfinite(y).all()
    for bad in [-1,4]:
        with pytest.raises(ValueError): build_simplicial_batch(z,pos,max_dim=bad)
