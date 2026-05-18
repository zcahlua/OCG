"""Geometric simplicial lift for OC-GSN."""

from __future__ import annotations

import itertools
from typing import Optional
import torch
from torch import Tensor
from .batch import SimplicialBatch
from .incidence import build_incidence
from .features import build_features

_ERR = "Number of simplices exceeds max_num_simplices_per_graph; increase the cap or reduce cutoff/max_neighbors."


def _canonical_bonds(
    bond_edge_index: Optional[Tensor], num_nodes: int, device: torch.device
) -> Optional[Tensor]:
    if bond_edge_index is None:
        return None
    pairs = {
        tuple(sorted(map(int, p)))
        for p in bond_edge_index.detach().cpu().t().tolist()
        if int(p[0]) != int(p[1])
    }
    pairs = sorted(
        (i, j) for i, j in pairs if 0 <= i < num_nodes and 0 <= j < num_nodes
    )
    return (
        torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous()
        if pairs
        else torch.empty((2, 0), dtype=torch.long, device=device)
    )


def _check_cap(count: int, cap: Optional[int]) -> None:
    if cap is not None and count > cap:
        raise RuntimeError(_ERR)


def build_simplicial_batch(
    z: Tensor,
    pos: Tensor,
    batch: Optional[Tensor] = None,
    bond_edge_index: Optional[Tensor] = None,
    cutoff: float = 5.0,
    max_neighbors: Optional[int] = None,
    include_bonds: bool = True,
    max_dim: int = 3,
    rbf_dim: int = 32,
    use_chirality: bool = False,
    max_num_simplices_per_graph: Optional[int] = None,
    tie_tol: float = 1e-8,
) -> SimplicialBatch:
    """Construct a downward-closed K0/K1/K2/K3 batch from atoms and coordinates."""
    if use_chirality:
        raise NotImplementedError(
            "Permutation-consistent chirality is not implemented in v1."
        )
    if max_dim not in {0, 1, 2, 3}:
        raise ValueError("max_dim must be one of {0,1,2,3}")
    device = pos.device
    z = z.to(device=device, dtype=torch.long)
    batch0 = (
        torch.zeros(z.shape[0], dtype=torch.long, device=device)
        if batch is None
        else batch.to(device=device, dtype=torch.long)
    )
    num_graphs = int(batch0.max().item()) + 1 if batch0.numel() else 0
    bond_edge_index = _canonical_bonds(
        bond_edge_index.to(device) if bond_edge_index is not None else None,
        int(z.shape[0]),
        device,
    )
    bond_pairs = (
        set()
        if bond_edge_index is None
        else {tuple(map(int, p)) for p in bond_edge_index.t().tolist()}
    )

    all_e, all_t, all_q, b1, b2, b3 = [], [], [], [], [], []
    for g in range(num_graphs):
        nodes = torch.nonzero(batch0 == g, as_tuple=False).flatten().tolist()
        node_set = set(map(int, nodes))
        edges: set[tuple[int, int]] = set()
        if max_dim >= 1:
            if max_neighbors is None:
                for a, b in itertools.combinations(nodes, 2):
                    if torch.linalg.norm(pos[b] - pos[a]).item() <= cutoff:
                        edges.add(tuple(sorted((int(a), int(b)))))
            else:
                for a in nodes:
                    dists = []
                    for b in nodes:
                        if a != b:
                            dists.append(
                                (torch.linalg.norm(pos[b] - pos[a]).item(), int(b))
                            )
                    dists.sort(key=lambda x: x[0])
                    if dists and max_neighbors > 0:
                        dk = dists[min(max_neighbors, len(dists)) - 1][0]
                        for d, b in dists:
                            if d <= dk + tie_tol and d <= cutoff:
                                edges.add(tuple(sorted((int(a), b))))
            if include_bonds:
                for e in bond_pairs:
                    if e[0] in node_set and e[1] in node_set:
                        edges.add(e)
        edges_l = sorted(edges)
        _check_cap(len(edges_l), max_num_simplices_per_graph)
        all_e.extend(edges_l)
        b1.extend([g] * len(edges_l))
        edge_set = set(edges_l)
        tris: list[tuple[int, int, int]] = []
        if max_dim >= 2:
            for tri in itertools.combinations(nodes, 3):
                tri = tuple(sorted(map(int, tri)))
                if all(
                    tuple(sorted(p)) in edge_set for p in itertools.combinations(tri, 2)
                ):
                    tris.append(tri)
        _check_cap(len(tris), max_num_simplices_per_graph)
        all_t.extend(tris)
        b2.extend([g] * len(tris))
        tets: list[tuple[int, int, int, int]] = []
        if max_dim >= 3:
            for tet in itertools.combinations(nodes, 4):
                tet = tuple(sorted(map(int, tet)))
                if all(
                    tuple(sorted(p)) in edge_set for p in itertools.combinations(tet, 2)
                ):
                    tets.append(tet)
        _check_cap(len(tets), max_num_simplices_per_graph)
        all_q.extend(tets)
        b3.extend([g] * len(tets))

    def ten(rows, width):
        return (
            torch.tensor(rows, dtype=torch.long, device=device)
            if rows
            else torch.empty((0, width), dtype=torch.long, device=device)
        )

    k1, k2, k3 = ten(all_e, 2), ten(all_t, 3), ten(all_q, 4)
    inc = build_incidence(k1, k2, k3, int(z.shape[0]))
    feats = build_features(z, pos, k1, k2, k3, bond_edge_index, rbf_dim, cutoff)
    return SimplicialBatch(
        z=z,
        pos=pos,
        batch0=batch0,
        bond_edge_index=bond_edge_index,
        k1=k1,
        k2=k2,
        k3=k3,
        batch1=(
            torch.tensor(b1, dtype=torch.long, device=device)
            if b1
            else torch.empty((0,), dtype=torch.long, device=device)
        ),
        batch2=(
            torch.tensor(b2, dtype=torch.long, device=device)
            if b2
            else torch.empty((0,), dtype=torch.long, device=device)
        ),
        batch3=(
            torch.tensor(b3, dtype=torch.long, device=device)
            if b3
            else torch.empty((0,), dtype=torch.long, device=device)
        ),
        **inc,
        **feats,
        num_graphs=num_graphs,
    )
