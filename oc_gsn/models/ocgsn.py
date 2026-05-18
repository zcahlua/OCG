"""Top-level Order-Controlled Geometric Simplicial Network."""
from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor, nn

from oc_gsn.complex.batch import SimplicialBatch
from oc_gsn.complex.lift import build_simplicial_batch

from .layer import OCGSNLayer
from .mlp import MLP
from .readout import OCGSNReadout
from .simplex_feature_orbit import SimplexFeatureOrbitEncoder


class OCGSN(nn.Module):
    """Scalar E(3)-invariant simplicial network over K0/K1/K2/K3."""

    def __init__(
        self,
        max_atomic_number: int = 100,
        num_atom_types: Optional[int] = None,
        hidden_dim: int = 128,
        num_layers: int = 4,
        rbf_dim: int = 32,
        cutoff: float = 5.0,
        max_neighbors: Optional[int] = None,
        include_bonds: bool = True,
        max_dim: int = 3,
        pi_mode: str = "full",
        orbit_pool: str = "mean",
        readout_pool: str = "sum",
        target_dim: int = 1,
        use_chirality: bool = False,
        z_is_atomic_number: bool = True,
        max_num_simplices_per_graph: Optional[int] = None,
        tie_tol: float = 1e-8,
    ) -> None:
        super().__init__()
        if max_dim not in {0, 1, 2, 3}:
            raise ValueError("max_dim must be one of {0,1,2,3}")
        if use_chirality:
            raise NotImplementedError(
                "Permutation-consistent chirality is not implemented in v1."
            )
        if not z_is_atomic_number and num_atom_types is None:
            raise ValueError(
                "num_atom_types must be provided when z_is_atomic_number=False"
            )

        self.max_dim = max_dim
        self.rbf_dim = rbf_dim
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.include_bonds = include_bonds
        self.use_chirality = use_chirality
        self.hidden_dim = hidden_dim
        self.max_num_simplices_per_graph = max_num_simplices_per_graph
        self.tie_tol = tie_tol

        if z_is_atomic_number:
            self.atom_embedding = nn.Embedding(max_atomic_number + 1, hidden_dim)
        else:
            self.atom_embedding = nn.Embedding(int(num_atom_types), hidden_dim)

        self.gamma0_mlp = MLP(2, hidden_dim, hidden_dim)
        self.edge_init = SimplexFeatureOrbitEncoder(
            rbf_dim + 2 + 2 * hidden_dim,
            hidden_dim,
            hidden_dim,
            pool=orbit_pool,
        )
        self.tri_init = SimplexFeatureOrbitEncoder(
            3 * rbf_dim + 10 + 3 * hidden_dim,
            hidden_dim,
            hidden_dim,
            pool=orbit_pool,
        )
        self.tet_init = SimplexFeatureOrbitEncoder(
            6 * rbf_dim + 17 + 4 * hidden_dim,
            hidden_dim,
            hidden_dim,
            pool=orbit_pool,
        )
        self.layers = nn.ModuleList(
            [OCGSNLayer(hidden_dim, pi_mode, orbit_pool) for _ in range(num_layers)]
        )
        self.readout = OCGSNReadout(hidden_dim, target_dim, readout_pool)

    @property
    def expected_gamma_dims(self) -> tuple[int, int, int, int]:
        """Expected raw geometric feature sizes for gamma0/gamma1/gamma2/gamma3."""
        return (
            2,
            self.rbf_dim + 2,
            3 * self.rbf_dim + 10,
            6 * self.rbf_dim + 17,
        )

    def _raise_feature_dim_error(
        self,
        name: str,
        actual_shape: torch.Size,
        expected_shape: tuple[int, ...],
    ) -> None:
        raise ValueError(
            "SimplicialBatch feature dimension mismatch: "
            f"{name} has shape {tuple(actual_shape)}, "
            f"but model rbf_dim={self.rbf_dim} expects shape {expected_shape}. "
            "Rebuild the batch with the same rbf_dim and feature conventions as "
            "the model."
        )

    def _validate_batch_feature_dims(self, sb: SimplicialBatch) -> None:
        """Validate a precomputed or freshly built batch before state initialization."""
        expected_gamma0_dim, expected_g1, expected_g2, expected_g3 = (
            self.expected_gamma_dims
        )
        checks = [
            ("gamma0", sb.gamma0.shape, (sb.num_nodes, expected_gamma0_dim)),
            ("gamma1_perm", sb.gamma1_perm.shape, (sb.num_edges, 2, expected_g1)),
            (
                "gamma2_perm",
                sb.gamma2_perm.shape,
                (sb.num_triangles, 6, expected_g2),
            ),
            (
                "gamma3_perm",
                sb.gamma3_perm.shape,
                (sb.num_tetrahedra, 24, expected_g3),
            ),
            (
                "edge_perm_nodes",
                sb.edge_perm_nodes.shape,
                (sb.num_edges, 2, 2),
            ),
            (
                "tri_perm_nodes",
                sb.tri_perm_nodes.shape,
                (sb.num_triangles, 6, 3),
            ),
            (
                "tet_perm_nodes",
                sb.tet_perm_nodes.shape,
                (sb.num_tetrahedra, 24, 4),
            ),
            ("k1", sb.k1.shape, (sb.num_edges, 2)),
            ("k2", sb.k2.shape, (sb.num_triangles, 3)),
            ("k3", sb.k3.shape, (sb.num_tetrahedra, 4)),
        ]
        for name, actual_shape, expected_shape in checks:
            if tuple(actual_shape) != expected_shape:
                self._raise_feature_dim_error(name, actual_shape, expected_shape)

    def initialize_states(
        self,
        batch: SimplicialBatch,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Create h0/h1/h2/h3 from atom embeddings and simplex orbit features."""
        emb = self.atom_embedding(batch.z)
        h0 = emb + self.gamma0_mlp(batch.gamma0.to(emb.dtype))

        def init(
            gamma: Tensor,
            perm_nodes: Tensor,
            enc: SimplexFeatureOrbitEncoder,
        ) -> Tensor:
            if gamma.shape[0] == 0:
                return emb.new_empty((0, self.hidden_dim))
            atoms = emb[perm_nodes].reshape(gamma.shape[0], gamma.shape[1], -1)
            return enc(torch.cat([gamma.to(emb.dtype), atoms], dim=-1))

        return (
            h0,
            init(batch.gamma1_perm, batch.edge_perm_nodes, self.edge_init),
            init(batch.gamma2_perm, batch.tri_perm_nodes, self.tri_init),
            init(batch.gamma3_perm, batch.tet_perm_nodes, self.tet_init),
        )

    def forward(
        self,
        batch_data: Optional[SimplicialBatch] = None,
        *,
        z: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        bond_edge_index: Optional[Tensor] = None,
        return_aux: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        """Run from either a prebuilt SimplicialBatch or raw molecular tensors."""
        device = next(self.parameters()).device
        if batch_data is not None:
            sb = batch_data.to(device)
        else:
            if z is None or pos is None:
                raise ValueError("Provide either batch_data or z and pos")
            z = z.to(device=device, dtype=torch.long)
            pos = pos.to(device=device)
            batch = batch.to(device=device, dtype=torch.long) if batch is not None else None
            bond_edge_index = (
                bond_edge_index.to(device=device, dtype=torch.long)
                if bond_edge_index is not None
                else None
            )
            sb = build_simplicial_batch(
                z=z,
                pos=pos,
                batch=batch,
                bond_edge_index=bond_edge_index,
                cutoff=self.cutoff,
                max_neighbors=self.max_neighbors,
                include_bonds=self.include_bonds,
                max_dim=self.max_dim,
                rbf_dim=self.rbf_dim,
                use_chirality=self.use_chirality,
                max_num_simplices_per_graph=self.max_num_simplices_per_graph,
                tie_tol=self.tie_tol,
            )

        self._validate_batch_feature_dims(sb)
        h0, h1, h2, h3 = self.initialize_states(sb)
        for layer in self.layers:
            h0, h1, h2, h3 = layer(h0, h1, h2, h3, sb)
        y = self.readout(
            h0,
            h1,
            h2,
            h3,
            sb.batch0,
            sb.batch1,
            sb.batch2,
            sb.batch3,
            sb.num_graphs,
        )
        if return_aux:
            return y, {
                "batch": sb,
                "h0": h0,
                "h1": h1,
                "h2": h2,
                "h3": h3,
                "simplex_counts": (
                    sb.num_nodes,
                    sb.num_edges,
                    sb.num_triangles,
                    sb.num_tetrahedra,
                ),
                "gamma_shapes": (
                    tuple(sb.gamma0.shape),
                    tuple(sb.gamma1_perm.shape),
                    tuple(sb.gamma2_perm.shape),
                    tuple(sb.gamma3_perm.shape),
                ),
            }
        return y
