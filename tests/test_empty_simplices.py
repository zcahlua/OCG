import torch
from oc_gsn import OCGSN

def run(z,pos,batch=None,cutoff=.6,max_dim=3):
    m=OCGSN(hidden_dim=8,num_layers=1,rbf_dim=2,cutoff=cutoff,max_dim=max_dim)
    y,aux=m(z=z,pos=pos,batch=batch,return_aux=True)
    assert torch.isfinite(y).all(); return aux

def test_empty_cases_and_per_graph_nulls():
    a=run(torch.tensor([1]),torch.zeros(1,3)); assert a['batch'].k1.shape==(0,2)
    b=run(torch.tensor([1,1]),torch.tensor([[0.,0,0],[1.,0,0]])); assert b['batch'].k1.shape[0]==0
    c=run(torch.tensor([1,1,1]),torch.tensor([[0.,0,0],[.5,0,0],[1.,0,0]])); assert c['batch'].k2.shape[0]==0
    z=torch.tensor([1,1,1,1]); pos=torch.tensor([[0.,0,0],[.5,0,0],[.25,.4,0],[3.,0,0]])
    d=run(z,pos,cutoff=.7); assert d['batch'].k2.shape[0] >= 1 and d['batch'].k3.shape[0]==0
    z2=torch.tensor([1,1,1]); pos2=torch.tensor([[0.,0,0],[.5,0,0],[2.,0,0]]); batch=torch.tensor([0,0,1])
    run(z2,pos2,batch=batch,cutoff=1.0)
