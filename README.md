# OC-GSN: Order-Controlled Geometric Simplicial Network

OC-GSN is a self-contained PyTorch research prototype for scalar molecular property prediction on a finite lifted simplicial complex `K=(K0,K1,K2,K3)`.

## Simplices

- `K0`: atom states.
- `K1`: lifted pair/edge states.
- `K2`: triangle/angle states.
- `K3`: tetrahedron/four-body local geometry states.

## Geometric lift

For each graph independently, `K1` is the union of radius/kNN geometric edges and bonds when `include_bonds=True`; when `include_bonds=False`, bonds are not forced into `K1`. `K2` is every 3-clique in `K1`, and `K3` is every 4-clique in `K1`, so the complex is downward closed. `max_dim` exactly controls whether only `K0`, up to `K1`, up to `K2`, or all dimensions are built. Enumeration is Python-first and correctness-oriented; simplex caps raise rather than truncate.

## Message equations

For a simplex `sigma`,

```text
partial_ord(sigma) = (tau_1, ..., tau_{k+1})

eta_k(sigma; pi)
  = MLP_{k-1->k}(
      (h_{tau_{pi(1)}} || a_{sigma,tau_{pi(1)}})
      || ...
      || (h_{tau_{pi(k+1)}} || a_{sigma,tau_{pi(k+1)}})
      || h_sigma)

m^{k-1 -> k}_sigma = Gamma_k { eta_k(sigma; pi) | pi in Pi_k }
```

Regimes: `Pi_k={id}` is order-aware; `Pi_k=S_{k+1}` is boundary-order-invariant; intermediate subsets are future partially order-aware regimes.

Downward coface aggregation is

```text
m^{k+1 -> k}_sigma
  = AGG_k { phi_{k+1->k}(h_rho, h_sigma, r_{rho,sigma}) | rho in cof(sigma) }
AGG_k(empty set) = b_down_k
```

Each layer applies the directed sweep `1 -> 0`, `2 -> 1 <- 0`, `3 -> 2 <- 1`, `2 -> 3`. Readout pools each simplex dimension per graph and predicts

```text
r_k = POOL_k { h_sigma^L | sigma in K_k }
h_K = MLP_K(r_0 || r_1 || r_2 || r_3)
y_hat = Head(h_K)
```

## Simplex feature orbit pooling

Full boundary-orbit pooling alone does not make the model atom-permutation-invariant if initial simplex features are tied to sorted node order. OC-GSN v1 therefore orbit-pools ordered geometric features and ordered atom embeddings during initialization:

```text
h_sigma^0 = Pool_{pi in S_{k+1}} phi_k(ordered_geom(pi(sigma)) || ordered_atom_embeddings(pi(sigma)))
```

for `k=1,2,3`. Default geometry uses distances, angle cosines, triangle areas, and unsigned tetrahedron volumes, giving scalar translation-, rotation-, and reflection-invariant features. Default incidence features are constants and contain no raw indices, orientation signs, simplex IDs, or lexicographic ranks.

## Limitations

This is not full 4-WL, not the local 4-WL tuple-patch model, and not a high-degree equivariant tensor model. Chirality is not implemented in v1 and `use_chirality=True` raises `NotImplementedError`. There is no RDKit integration and no QM9/MD17 training loop. Existing molecular models can be baselines or future optional stems, but are not the core.

## Run

```bash
pip install -r requirements.txt
pytest
python scripts/smoke_test.py
```

## Future extensions

Cache `SimplicialBatch` in dataset transforms, add QM9/MD17 loaders, add scalar-vector geometric stems, add force prediction via `F_i=-grad_{x_i}E`, add a permutation-consistent chirality pseudoscalar, and optimize clique enumeration.
