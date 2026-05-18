"""Dataclass container for a lifted batched simplicial complex."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional
import torch
from torch import Tensor


@dataclass
class SimplicialBatch:
    """All tensors needed by OC-GSN for K0, K1, K2 and K3."""

    z: Tensor
    pos: Tensor
    batch0: Tensor
    bond_edge_index: Optional[Tensor]
    k1: Tensor
    k2: Tensor
    k3: Tensor
    batch1: Tensor
    batch2: Tensor
    batch3: Tensor
    edge_to_nodes: Tensor
    tri_to_edges: Tensor
    tet_to_tris: Tensor
    edge_perm_nodes: Tensor
    tri_perm_nodes: Tensor
    tet_perm_nodes: Tensor
    node_cofaces_edge_target: Tensor
    node_cofaces_edge_source: Tensor
    edge_cofaces_tri_target: Tensor
    edge_cofaces_tri_source: Tensor
    tri_cofaces_tet_target: Tensor
    tri_cofaces_tet_source: Tensor
    gamma0: Tensor
    gamma1_perm: Tensor
    gamma2_perm: Tensor
    gamma3_perm: Tensor
    a10: Tensor
    a21: Tensor
    a32: Tensor
    r10: Tensor
    r21: Tensor
    r32: Tensor
    num_graphs: int

    @property
    def num_nodes(self) -> int:
        return int(self.z.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.k1.shape[0])

    @property
    def num_triangles(self) -> int:
        return int(self.k2.shape[0])

    @property
    def num_tetrahedra(self) -> int:
        return int(self.k3.shape[0])

    def to(self, device: torch.device | str) -> "SimplicialBatch":
        """Return a copy with every tensor moved to ``device``."""
        kwargs = {}
        for f in fields(self):
            value = getattr(self, f.name)
            kwargs[f.name] = value.to(device) if isinstance(value, Tensor) else value
        return SimplicialBatch(**kwargs)
