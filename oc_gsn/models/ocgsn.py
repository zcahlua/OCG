"""Top-level Order-Controlled Geometric Simplicial Network."""
from __future__ import annotations
from typing import Optional, Any
import torch
from torch import nn, Tensor
from oc_gsn.complex.batch import SimplicialBatch
from oc_gsn.complex.lift import build_simplicial_batch
from .mlp import MLP
from .simplex_feature_orbit import SimplexFeatureOrbitEncoder
from .layer import OCGSNLayer
from .readout import OCGSNReadout

class OCGSN(nn.Module):
    """Scalar E(3)-invariant simplicial network over K0/K1/K2/K3."""
    def __init__(self, max_atomic_number: int = 100, num_atom_types: Optional[int] = None, hidden_dim: int = 128, num_layers: int = 4, rbf_dim: int = 32, cutoff: float = 5.0, max_neighbors: Optional[int] = None, include_bonds: bool = True, max_dim: int = 3, pi_mode: str = "full", orbit_pool: str = "mean", readout_pool: str = "sum", target_dim: int = 1, use_chirality: bool = False, z_is_atomic_number: bool = True) -> None:
        super().__init__()
        if use_chirality: raise NotImplementedError("Permutation-consistent chirality is not implemented in v1.")
        if not z_is_atomic_number and num_atom_types is None: raise ValueError("num_atom_types must be provided when z_is_atomic_number=False")
        self.hidden_dim=hidden_dim; self.rbf_dim=rbf_dim; self.cutoff=cutoff; self.max_neighbors=max_neighbors; self.include_bonds=include_bonds; self.max_dim=max_dim; self.use_chirality=use_chirality
        self.atom_embedding = nn.Embedding(max_atomic_number + 1, hidden_dim) if z_is_atomic_number else nn.Embedding(int(num_atom_types), hidden_dim)
        self.gamma0_mlp = MLP(2, hidden_dim, hidden_dim)
        self.edge_init = SimplexFeatureOrbitEncoder(rbf_dim + 2 + 2*hidden_dim, hidden_dim, hidden_dim, pool=orbit_pool)
        self.tri_init = SimplexFeatureOrbitEncoder(3*rbf_dim + 10 + 3*hidden_dim, hidden_dim, hidden_dim, pool=orbit_pool)
        self.tet_init = SimplexFeatureOrbitEncoder(6*rbf_dim + 17 + 4*hidden_dim, hidden_dim, hidden_dim, pool=orbit_pool)
        self.layers = nn.ModuleList([OCGSNLayer(hidden_dim, pi_mode, orbit_pool) for _ in range(num_layers)])
        self.readout = OCGSNReadout(hidden_dim, target_dim, readout_pool)
    def initialize_states(self, batch: SimplicialBatch) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Create h0/h1/h2/h3, using orbit-pooled ordered atom embeddings for higher simplices."""
        emb = self.atom_embedding(batch.z)
        h0 = emb + self.gamma0_mlp(batch.gamma0.to(emb.dtype))
        def init(gamma: Tensor, perm_nodes: Tensor, enc: SimplexFeatureOrbitEncoder) -> Tensor:
            if gamma.shape[0] == 0: return emb.new_empty((0, self.hidden_dim))
            atoms = emb[perm_nodes].reshape(gamma.shape[0], gamma.shape[1], -1)
            return enc(torch.cat([gamma.to(emb.dtype), atoms], dim=-1))
        return h0, init(batch.gamma1_perm, batch.edge_perm_nodes, self.edge_init), init(batch.gamma2_perm, batch.tri_perm_nodes, self.tri_init), init(batch.gamma3_perm, batch.tet_perm_nodes, self.tet_init)
    def forward(self, batch_data: Optional[SimplicialBatch] = None, *, z: Optional[Tensor] = None, pos: Optional[Tensor] = None, batch: Optional[Tensor] = None, bond_edge_index: Optional[Tensor] = None, return_aux: bool = False) -> Tensor | tuple[Tensor, dict[str, Any]]:
        """Run from either a prebuilt SimplicialBatch or raw molecular tensors."""
        sb = batch_data
        if sb is None:
            if z is None or pos is None: raise ValueError("Provide either batch_data or z and pos")
            sb = build_simplicial_batch(z, pos, batch=batch, bond_edge_index=bond_edge_index, cutoff=self.cutoff, max_neighbors=self.max_neighbors, include_bonds=self.include_bonds, max_dim=self.max_dim, rbf_dim=self.rbf_dim, use_chirality=self.use_chirality)
        h0,h1,h2,h3 = self.initialize_states(sb)
        for layer in self.layers: h0,h1,h2,h3 = layer(h0,h1,h2,h3,sb)
        y = self.readout(h0,h1,h2,h3,sb.batch0,sb.batch1,sb.batch2,sb.batch3,sb.num_graphs)
        if return_aux:
            return y, {"batch": sb, "h0": h0, "h1": h1, "h2": h2, "h3": h3, "simplex_counts": (sb.num_nodes, sb.num_edges, sb.num_triangles, sb.num_tetrahedra), "gamma_shapes": (tuple(sb.gamma0.shape), tuple(sb.gamma1_perm.shape), tuple(sb.gamma2_perm.shape), tuple(sb.gamma3_perm.shape))}
        return y
