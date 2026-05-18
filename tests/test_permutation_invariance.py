import torch
from oc_gsn import OCGSN

def test_full_model_atom_permutation_invariance():
    torch.manual_seed(3)
    z=torch.tensor([6,1,8,7,1]); pos=torch.tensor([[0.,0,0],[1.1,0.2,0],[0.1,1.3,0.2],[0.2,0.3,1.4],[1.0,1.1,0.9]])
    m=OCGSN(hidden_dim=12,num_layers=1,rbf_dim=3,cutoff=5.0,pi_mode='full')
    p=torch.tensor([2,4,0,3,1])
    assert torch.allclose(m(z=z,pos=pos),m(z=z[p],pos=pos[p]),atol=1e-5)
