"""One Gauss-Seidel OC-GSN message-passing layer."""
from __future__ import annotations
from torch import nn, Tensor
import torch
from oc_gsn.complex.batch import SimplicialBatch
from .boundary_orbit import BoundaryOrbitEncoder
from .coface import CofaceAggregator
from .mlp import MLP

class OCGSNLayer(nn.Module):
    """Directed sweep: 1->0, 2->1<-0, 3->2<-1, 2->3."""
    def __init__(self, hidden_dim: int, pi_mode: str = "full", orbit_pool: str = "mean") -> None:
        super().__init__(); h=hidden_dim
        self.down10=CofaceAggregator(h,1,h,h); self.down21=CofaceAggregator(h,1,h,h); self.down32=CofaceAggregator(h,1,h,h)
        self.up01=BoundaryOrbitEncoder(2,h,1,h,h,pi_mode,orbit_pool); self.up12=BoundaryOrbitEncoder(3,h,1,h,h,pi_mode,orbit_pool); self.up23=BoundaryOrbitEncoder(4,h,1,h,h,pi_mode,orbit_pool)
        self.update0=MLP(2*h,h,h); self.update1=MLP(3*h,h,h); self.update2=MLP(3*h,h,h); self.update3=MLP(2*h,h,h)
        self.norm0=nn.LayerNorm(h); self.norm1=nn.LayerNorm(h); self.norm2=nn.LayerNorm(h); self.norm3=nn.LayerNorm(h)
    def forward(self, h0: Tensor, h1: Tensor, h2: Tensor, h3: Tensor, batch: SimplicialBatch) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Apply one layer to all simplex states."""
        m10 = self.down10(h1[batch.node_cofaces_edge_source], h0[batch.node_cofaces_edge_target], batch.r10, batch.node_cofaces_edge_target, h0.shape[0])
        h0n = self.norm0(h0 + self.update0(torch.cat([h0, m10], -1)))
        if h1.shape[0]:
            m01 = self.up01(h0n[batch.edge_to_nodes], batch.a10, h1)
            m21 = self.down21(h2[batch.edge_cofaces_tri_source], h1[batch.edge_cofaces_tri_target], batch.r21, batch.edge_cofaces_tri_target, h1.shape[0])
            h1n = self.norm1(h1 + self.update1(torch.cat([h1, m01, m21], -1)))
        else: h1n = h1
        if h2.shape[0]:
            m12 = self.up12(h1n[batch.tri_to_edges], batch.a21, h2)
            m32 = self.down32(h3[batch.tri_cofaces_tet_source], h2[batch.tri_cofaces_tet_target], batch.r32, batch.tri_cofaces_tet_target, h2.shape[0])
            h2n = self.norm2(h2 + self.update2(torch.cat([h2, m12, m32], -1)))
        else: h2n = h2
        if h3.shape[0]:
            m23 = self.up23(h2n[batch.tet_to_tris], batch.a32, h3)
            h3n = self.norm3(h3 + self.update3(torch.cat([h3, m23], -1)))
        else: h3n = h3
        return h0n, h1n, h2n, h3n
