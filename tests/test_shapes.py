import torch
from oc_gsn import OCGSN

def test_tetrahedron_shapes():
    z=torch.tensor([6,1,1,1])
    pos=torch.tensor([[0.,0,0],[1,0,0],[0,1,0],[0,0,1.]])
    model=OCGSN(hidden_dim=16,num_layers=1,rbf_dim=4,cutoff=2.0)
    y,aux=model(z=z,pos=pos,batch=torch.zeros(4,dtype=torch.long),return_aux=True)
    assert y.shape==(1,1)
    assert aux['simplex_counts'][0]==4 and aux['simplex_counts'][1]>0 and aux['simplex_counts'][2]>0 and aux['simplex_counts'][3]>0
    for h in [aux['h0'],aux['h1'],aux['h2'],aux['h3']]: assert h.shape[1]==16
