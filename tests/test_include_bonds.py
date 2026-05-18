import torch
from oc_gsn import build_simplicial_batch

def test_include_bonds_controls_forcing_and_indicator():
    z=torch.tensor([1,1]); pos=torch.tensor([[0.,0,0],[10.,0,0]]); bonds=torch.tensor([[0],[1]])
    b=build_simplicial_batch(z,pos,bond_edge_index=bonds,cutoff=1.0,include_bonds=True,rbf_dim=2)
    assert b.k1.tolist()==[[0,1]]
    b=build_simplicial_batch(z,pos,bond_edge_index=bonds,cutoff=1.0,include_bonds=False,rbf_dim=2)
    assert b.k1.shape[0]==0
    pos2=torch.tensor([[0.,0,0],[0.5,0,0]])
    b=build_simplicial_batch(z,pos2,bond_edge_index=bonds,cutoff=1.0,include_bonds=False,rbf_dim=2)
    assert b.k1.tolist()==[[0,1]] and torch.all(b.gamma1_perm[:,:, -1] == 1)
