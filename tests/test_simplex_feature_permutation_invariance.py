import torch
from oc_gsn import OCGSN, build_simplicial_batch


def dmap(k, inv):
    return {
        tuple(sorted(int(inv[i]) for i in row)): idx
        for idx, row in enumerate(k.tolist())
    }


def test_initial_simplex_features_relabel_invariant():
    torch.manual_seed(0)
    z = torch.tensor([6, 1, 8, 7, 1])
    pos = torch.tensor(
        [[0.0, 0, 0], [1.1, 0.2, 0], [0.1, 1.3, 0.2], [0.2, 0.3, 1.4], [1.0, 1.1, 0.9]]
    )
    p = torch.tensor([2, 4, 0, 3, 1])
    inv = torch.empty_like(p)
    inv[p] = torch.arange(len(p))
    B = build_simplicial_batch(z, pos, cutoff=3.0, rbf_dim=3)
    BP = build_simplicial_batch(z[p], pos[p], cutoff=3.0, rbf_dim=3)
    m = OCGSN(hidden_dim=10, num_layers=0, rbf_dim=3, cutoff=3.0)
    _, h1, h2, h3 = m.initialize_states(B)
    _, g1, g2, g3 = m.initialize_states(BP)
    for K, H, G in [(B.k1, h1, g1), (B.k2, h2, g2), (B.k3, h3, g3)]:
        mp = dmap(BP.k1 if K.shape[1] == 2 else BP.k2 if K.shape[1] == 3 else BP.k3, p)
        for idx, row in enumerate(K.tolist()):
            assert torch.allclose(H[idx], G[mp[tuple(row)]], atol=1e-5)
