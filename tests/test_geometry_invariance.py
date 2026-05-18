import torch
from oc_gsn import OCGSN

def test_e3_invariance_reflection_included():
    torch.manual_seed(2)
    z=torch.tensor([6,1,8,7,1]); pos=torch.tensor([[0.,0,0],[1.1,0.2,0],[0.1,1.3,0.2],[0.2,0.3,1.4],[1.0,1.1,0.9]])
    m=OCGSN(hidden_dim=12,num_layers=1,rbf_dim=3,cutoff=5.0)
    y=m(z=z,pos=pos); t=torch.tensor([3.,-2.,1.])
    A=torch.randn(3,3); Q,_=torch.linalg.qr(A); Q[:,0]*=-1
    assert torch.allclose(y,m(z=z,pos=pos+t),atol=1e-5)
    assert torch.allclose(y,m(z=z,pos=pos@Q.T+t),atol=1e-5)
