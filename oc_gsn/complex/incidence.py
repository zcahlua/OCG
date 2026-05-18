"""Boundary and coface incidence construction for OC-GSN."""
from __future__ import annotations

from typing import Dict, Tuple
import torch
from torch import Tensor


def build_incidence(k1: Tensor, k2: Tensor, k3: Tensor, num_nodes: int) -> dict[str, Tensor]:
    """Build boundary indices, coface maps, and constant incidence features."""
    device = k1.device
    edge_map: Dict[Tuple[int, int], int] = {tuple(map(int, e.tolist())): idx for idx, e in enumerate(k1)}
    tri_map: Dict[Tuple[int, int, int], int] = {tuple(map(int, t.tolist())): idx for idx, t in enumerate(k2)}

    edge_to_nodes = k1.clone() if k1.numel() else torch.empty((0, 2), dtype=torch.long, device=device)

    tri_edges = []
    for tri in k2.tolist():
        i, j, k = map(int, tri)
        tri_edges.append([edge_map[(i, j)], edge_map[(i, k)], edge_map[(j, k)]])
    tri_to_edges = torch.tensor(tri_edges, dtype=torch.long, device=device) if tri_edges else torch.empty((0, 3), dtype=torch.long, device=device)

    tet_tris = []
    for tet in k3.tolist():
        i, j, k, l = map(int, tet)
        tet_tris.append([tri_map[(i, j, k)], tri_map[(i, j, l)], tri_map[(i, k, l)], tri_map[(j, k, l)]])
    tet_to_tris = torch.tensor(tet_tris, dtype=torch.long, device=device) if tet_tris else torch.empty((0, 4), dtype=torch.long, device=device)

    node_tgt, node_src = [], []
    for e_idx, (i, j) in enumerate(k1.tolist()):
        node_tgt.extend([int(i), int(j)]); node_src.extend([e_idx, e_idx])
    edge_tgt, edge_src = [], []
    for f_idx, edges in enumerate(tri_to_edges.tolist()):
        for e_idx in edges:
            edge_tgt.append(int(e_idx)); edge_src.append(f_idx)
    tri_tgt, tri_src = [], []
    for t_idx, tris in enumerate(tet_to_tris.tolist()):
        for f_idx in tris:
            tri_tgt.append(int(f_idx)); tri_src.append(t_idx)

    def lt(x: list[int]) -> Tensor:
        return torch.tensor(x, dtype=torch.long, device=device) if x else torch.empty((0,), dtype=torch.long, device=device)

    n_ie, n_et, n_tt = len(node_tgt), len(edge_tgt), len(tri_tgt)
    return {
        "edge_to_nodes": edge_to_nodes,
        "tri_to_edges": tri_to_edges,
        "tet_to_tris": tet_to_tris,
        "node_cofaces_edge_target": lt(node_tgt),
        "node_cofaces_edge_source": lt(node_src),
        "edge_cofaces_tri_target": lt(edge_tgt),
        "edge_cofaces_tri_source": lt(edge_src),
        "tri_cofaces_tet_target": lt(tri_tgt),
        "tri_cofaces_tet_source": lt(tri_src),
        "a10": torch.ones((k1.shape[0], 2, 1), dtype=torch.float32, device=device),
        "a21": torch.ones((k2.shape[0], 3, 1), dtype=torch.float32, device=device),
        "a32": torch.ones((k3.shape[0], 4, 1), dtype=torch.float32, device=device),
        "r10": torch.ones((n_ie, 1), dtype=torch.float32, device=device),
        "r21": torch.ones((n_et, 1), dtype=torch.float32, device=device),
        "r32": torch.ones((n_tt, 1), dtype=torch.float32, device=device),
    }
