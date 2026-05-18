"""E(3)-invariant geometric feature construction for lifted simplices."""
from __future__ import annotations

import itertools
import torch
from torch import Tensor

EPS = 1e-8


def rbf(d: Tensor, rbf_dim: int, cutoff: float) -> Tensor:
    centers = torch.linspace(0.0, cutoff, rbf_dim, dtype=d.dtype, device=d.device)
    delta = cutoff if rbf_dim <= 1 else centers[1] - centers[0]
    beta = 1.0 / (delta * delta + EPS)
    return torch.exp(-beta * (d.unsqueeze(-1) - centers) ** 2)


def pair_distance(pos: Tensor, i: int, j: int) -> Tensor:
    """Euclidean distance between two atoms."""
    return torch.linalg.norm(pos[j] - pos[i])


def angle_cos(pos: Tensor, center: int, a: int, b: int) -> Tensor:
    """Cosine of angle a-center-b."""
    va, vb = pos[a] - pos[center], pos[b] - pos[center]
    return (va * vb).sum() / (torch.linalg.norm(va) * torch.linalg.norm(vb) + EPS)


def triangle_area(pos: Tensor, i: int, j: int, k: int) -> Tensor:
    """Unsigned triangle area."""
    return 0.5 * torch.linalg.norm(torch.cross(pos[j] - pos[i], pos[k] - pos[i], dim=0))


def tetra_abs_volume(pos: Tensor, i: int, j: int, k: int, l: int) -> Tensor:
    """Absolute tetrahedron volume."""
    mat = torch.stack([pos[j] - pos[i], pos[k] - pos[i], pos[l] - pos[i]], dim=1)
    return torch.abs(torch.linalg.det(mat)) / 6.0


def _bond_set(bond_edge_index: Tensor | None) -> set[tuple[int, int]]:
    if bond_edge_index is None or bond_edge_index.numel() == 0:
        return set()
    pairs = bond_edge_index.t().tolist()
    return {tuple(sorted(map(int, p))) for p in pairs if int(p[0]) != int(p[1])}


def _bond_indicator(bonds: set[tuple[int, int]], i: int, j: int, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    return torch.tensor(float(tuple(sorted((i, j))) in bonds), dtype=dtype, device=device)


def ordered_edge_raw(pos: Tensor, bonds: set[tuple[int, int]], i: int, j: int, rbf_dim: int, cutoff: float) -> Tensor:
    d = pair_distance(pos, i, j)
    return torch.cat([rbf(d, rbf_dim, cutoff), d.reshape(1), _bond_indicator(bonds, i, j, dtype=pos.dtype, device=pos.device).reshape(1)])


def ordered_triangle_raw(pos: Tensor, bonds: set[tuple[int, int]], i: int, j: int, k: int, rbf_dim: int, cutoff: float) -> Tensor:
    dij, dik, djk = pair_distance(pos, i, j), pair_distance(pos, i, k), pair_distance(pos, j, k)
    vals = [rbf(dij, rbf_dim, cutoff), rbf(dik, rbf_dim, cutoff), rbf(djk, rbf_dim, cutoff),
            torch.stack([dij, dik, djk, angle_cos(pos, i, j, k), angle_cos(pos, j, i, k), angle_cos(pos, k, i, j), triangle_area(pos, i, j, k),
                         _bond_indicator(bonds, i, j, dtype=pos.dtype, device=pos.device), _bond_indicator(bonds, i, k, dtype=pos.dtype, device=pos.device), _bond_indicator(bonds, j, k, dtype=pos.dtype, device=pos.device)])]
    return torch.cat(vals)


def ordered_tetra_raw(pos: Tensor, bonds: set[tuple[int, int]], i: int, j: int, k: int, l: int, rbf_dim: int, cutoff: float) -> Tensor:
    pairs = [(i,j),(i,k),(i,l),(j,k),(j,l),(k,l)]
    ds = [pair_distance(pos, a, b) for a, b in pairs]
    areas = [triangle_area(pos, i, j, k), triangle_area(pos, i, j, l), triangle_area(pos, i, k, l), triangle_area(pos, j, k, l)]
    bonds_v = [_bond_indicator(bonds, a, b, dtype=pos.dtype, device=pos.device) for a, b in pairs]
    return torch.cat([*(rbf(d, rbf_dim, cutoff) for d in ds), torch.stack(ds + areas + [tetra_abs_volume(pos, i, j, k, l)] + bonds_v)])


def build_features(z: Tensor, pos: Tensor, k1: Tensor, k2: Tensor, k3: Tensor, bond_edge_index: Tensor | None, rbf_dim: int, cutoff: float) -> dict[str, Tensor]:
    """Build invariant atom features and permutation-expanded simplex raw features."""
    device, dtype, n = pos.device, pos.dtype, int(pos.shape[0])
    bonds = _bond_set(bond_edge_index)
    lift_deg = torch.zeros(n, dtype=dtype, device=device)
    bond_deg = torch.zeros(n, dtype=dtype, device=device)
    for i, j in k1.tolist():
        lift_deg[int(i)] += 1; lift_deg[int(j)] += 1
    for i, j in bonds:
        if 0 <= i < n and 0 <= j < n:
            bond_deg[i] += 1; bond_deg[j] += 1
    scale = float(max(1, n - 1))
    gamma0 = torch.stack([lift_deg / scale, bond_deg / scale], dim=-1)

    edge_perms = list(itertools.permutations(range(2)))
    tri_perms = list(itertools.permutations(range(3)))
    tet_perms = list(itertools.permutations(range(4)))

    epn, g1 = [], []
    for row in k1.tolist():
        nodes = list(map(int, row)); epn.append([[nodes[a] for a in p] for p in edge_perms]); g1.append([ordered_edge_raw(pos, bonds, nodes[p[0]], nodes[p[1]], rbf_dim, cutoff) for p in edge_perms])
    tpn, g2 = [], []
    for row in k2.tolist():
        nodes = list(map(int, row)); tpn.append([[nodes[a] for a in p] for p in tri_perms]); g2.append([ordered_triangle_raw(pos, bonds, nodes[p[0]], nodes[p[1]], nodes[p[2]], rbf_dim, cutoff) for p in tri_perms])
    qpn, g3 = [], []
    for row in k3.tolist():
        nodes = list(map(int, row)); qpn.append([[nodes[a] for a in p] for p in tet_perms]); g3.append([ordered_tetra_raw(pos, bonds, nodes[p[0]], nodes[p[1]], nodes[p[2]], nodes[p[3]], rbf_dim, cutoff) for p in tet_perms])

    g1dim, g2dim, g3dim = rbf_dim + 2, 3 * rbf_dim + 10, 6 * rbf_dim + 17
    return {
        "gamma0": gamma0,
        "edge_perm_nodes": torch.tensor(epn, dtype=torch.long, device=device) if epn else torch.empty((0, 2, 2), dtype=torch.long, device=device),
        "tri_perm_nodes": torch.tensor(tpn, dtype=torch.long, device=device) if tpn else torch.empty((0, 6, 3), dtype=torch.long, device=device),
        "tet_perm_nodes": torch.tensor(qpn, dtype=torch.long, device=device) if qpn else torch.empty((0, 24, 4), dtype=torch.long, device=device),
        "gamma1_perm": torch.stack([torch.stack(x) for x in g1]) if g1 else torch.empty((0, 2, g1dim), dtype=dtype, device=device),
        "gamma2_perm": torch.stack([torch.stack(x) for x in g2]) if g2 else torch.empty((0, 6, g2dim), dtype=dtype, device=device),
        "gamma3_perm": torch.stack([torch.stack(x) for x in g3]) if g3 else torch.empty((0, 24, g3dim), dtype=dtype, device=device),
    }
