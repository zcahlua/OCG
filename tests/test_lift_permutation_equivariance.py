import torch
from oc_gsn import build_simplicial_batch

def sets(x): return {tuple(r) for r in x.tolist()}
def mapped(k, inv): return {tuple(sorted(int(inv[i]) for i in row)) for row in k.tolist()}

def test_lift_permutation_equivariance():
    z=torch.tensor([6,1,8,7,1]); pos=torch.tensor([[0.,0,0],[1.1,0.2,0],[0.1,1.3,0.2],[0.2,0.3,1.4],[1.0,1.1,0.9]])
    B=build_simplicial_batch(z,pos,cutoff=3.0,rbf_dim=3)
    p=torch.tensor([2,4,0,3,1]); inv=torch.empty_like(p); inv[p]=torch.arange(len(p))
    BP=build_simplicial_batch(z[p],pos[p],cutoff=3.0,rbf_dim=3)
    assert sets(B.k1)==mapped(BP.k1,p) and sets(B.k2)==mapped(BP.k2,p) and sets(B.k3)==mapped(BP.k3,p)
