"""Smoke test for OC-GSN without pytest."""

import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from oc_gsn import OCGSN

z = torch.tensor([6, 1, 1, 1, 8, 1, 1], dtype=torch.long)
pos = torch.tensor(
    [
        [0.0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1.0],
        [4.0, 0, 0],
        [5.0, 0, 0],
        [4.0, 1, 0],
    ],
    dtype=torch.float32,
)
batch = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.long)
model = OCGSN(hidden_dim=16, num_layers=1, rbf_dim=4, cutoff=2.0)
y, aux = model(z=z, pos=pos, batch=batch, return_aux=True)
y.sum().backward()
params = sum(p.numel() for p in model.parameters())
with torch.no_grad():
    yt = model(z=z, pos=pos + torch.tensor([2.0, -1.0, 0.5]), batch=batch)
    A = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    yr = model(z=z, pos=pos @ A.T + torch.tensor([1.0, 2.0, 3.0]), batch=batch)
    p = torch.tensor([2, 0, 3, 1, 6, 4, 5])
    inv_batch = batch[p]
    yp = model(z=z[p], pos=pos[p], batch=inv_batch)
print("output shape:", tuple(y.shape))
print("parameter count:", params)
print("simplex counts:", aux["simplex_counts"])
print("max translation-invariance difference:", float((y.detach() - yt).abs().max()))
print(
    "max rotation/reflection-invariance difference:",
    float((y.detach() - yr).abs().max()),
)
print("max permutation-invariance difference:", float((y.detach() - yp).abs().max()))
