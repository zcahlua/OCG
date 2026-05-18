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

## Precomputed SimplicialBatch compatibility

`OCGSN.forward` can accept either raw molecular tensors (`z`, `pos`, optional `batch` and `bond_edge_index`) or a precomputed `SimplicialBatch`. If passing a precomputed `SimplicialBatch`, it must be built with feature conventions compatible with the model configuration, especially `rbf_dim` and `use_chirality`. The model validates `gamma0`, `gamma1_perm`, `gamma2_perm`, `gamma3_perm`, and permutation-node tensor shapes before initialization. If dimensions mismatch, the model raises `ValueError` rather than failing later with a low-level `Linear`/matmul error.

`OCGSN` exposes `max_num_simplices_per_graph` and forwards it to the lift. If a graph exceeds the cap, the model raises `RuntimeError` rather than silently truncating simplices, because silent truncation can break permutation invariance. The `tie_tol` constructor argument is also forwarded to kNN/radius tie handling during raw-tensor lifting.

## Differentiability of the lift

The lifted topology construction is discrete and non-differentiable. Gradients with respect to atomic positions flow through the geometric features computed on the selected simplices, not through edge/simplex selection itself. The selected `K1`/`K2`/`K3` topology is treated as fixed for the forward/backward pass.

## Limitations

This is not full 4-WL, not the local 4-WL tuple-patch model, and not a high-degree equivariant tensor model. Chirality is not implemented in v1 and `use_chirality=True` raises `NotImplementedError`. There is no RDKit integration, no MD17 training loop, and no force-training support. The included QM9 script is a lightweight benchmark layer around the existing model rather than production training infrastructure. Existing molecular models can be baselines or future optional stems, but are not the core.

## Run

```bash
pip install -r requirements.txt
pytest
python scripts/smoke_test.py
```

## Future extensions

Future work may add MD17 loaders, scalar-vector geometric stems, force prediction via `F_i=-grad_{x_i}E`, a permutation-consistent chirality pseudoscalar, optimized clique enumeration, and optional cached `SimplicialBatch` transforms.

## Lightweight QM9 training

This repository includes an extremely lightweight first QM9 benchmark script for the existing OC-GSN model. It is intended to verify real-dataset loading, deterministic split handling, target normalization, checkpointing, and MAE/MSE/RMSE reporting; it is not a full experimental framework and does not claim SOTA.

Install the optional QM9 dependency separately:

```bash
pip install -r requirements-qm9.txt
```

The script uses PyTorch Geometric QM9 directly. It does not add RDKit, does not regenerate QM9 conformers, does not train forces, and does not add MD17 or force datasets. Force datasets such as MD17 should be added later if force-training support is developed separately.

Quick smoke run:

```bash
python scripts/train_qm9.py --target 0 --epochs 1 --batch-size 8 --max-dim 2 --max-neighbors 8 --limit-train-batches 2 --limit-val-batches 1 --limit-test-batches 1
```

Split files are cached in the selected output directory with metadata for dataset length, train size, validation size, and seed. If those settings change, delete the cached split or use a different output directory so stale splits cannot be reused silently.

Quick max-dimension ablation smoke run:

```bash
python scripts/train_qm9.py \
    --target 4 \
    --epochs 1 \
    --batch-size 8 \
    --max-neighbors 8 \
    --ablation-max-dims true \
    --limit-train-batches 2 \
    --limit-val-batches 1 \
    --limit-test-batches 1
```

Longer ablation example:

```bash
python scripts/train_qm9.py \
    --target 4 \
    --epochs 20 \
    --batch-size 16 \
    --max-neighbors 8 \
    --ablation-max-dims true
```

The default QM9 settings use `max_dim=2` and `max_neighbors=8` to keep the first benchmark lightweight. Use `--ablation-max-dims` to test `max_dim=3` and include tetrahedra.

Runtime depends on device, cutoff, max_neighbors, max_dim, batch size, and the cost of online simplicial complex construction. The default settings are intended to be lightweight, but no fixed runtime is guaranteed.

The QM9 target is normalized using the training split only, training uses normalized MSE, and evaluation reports MAE/MSE/RMSE in original target units. Splits are deterministic for a seed and are shared across `--ablation-max-dims` runs, while each max-dimension run uses a fresh model and optimizer.

Topology construction is discrete and non-differentiable. Gradients flow through selected geometric features, not through edge/simplex selection.
