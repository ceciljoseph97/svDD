# Unsupervised Hyperbolic Multi‑Sphere SVDD (MNIST)

This folder contains the **unsupervised** trainer:

- `train_hyp_mnist_multi.py`

It implements a **soft-assignment, multi-sphere SVDD** objective in **hyperbolic space** (Poincaré ball) on top of an AE backbone.

---

## Core idea

Instead of hard membership:

> “sample \(x\) belongs to class \(k\)”

we use soft membership:

> “sample \(x\) belongs to all clusters with probability \(p(k\mid x)\)”

where \(p(k\mid x) = \mathrm{softmax}(-d^2(x,k)/T)\).

---

## What is “unsupervised” here (and what is not)

### Unsupervised (optimization objective)
In **Stage‑2** (SVDD stage), the training loop ignores labels:

- batches are `for x_scaled, _ in tr_loader`
- the loss uses **only**: reconstruction, distances to centers, soft assignments, radii, center separation/overlap penalties

### Label usage still present (pipeline “leak”)
Even though the Stage‑2 **loss** is unsupervised, the pipeline still uses digit labels in two places:

- **Preprocessing**: `preprocess_batch_by_digit_minmax(x_raw, digit)` uses the true digit to scale inputs.
- **Center initialization**: `init_centers_h(model, tr_loader_eval, ...)` computes `c_h[k]` using samples with label `k`.

### Supervised only for reporting
Evaluation uses labels to report:

- per‑digit AUC (legacy)
- Hungarian‑matched clustering accuracy
- NMI / ARI

If you want *strictly label‑free* runs, you must replace digit‑conditioned preprocessing and label‑based center initialization.

---

## Model components

- **Backbone**: `MNIST_LeNet_SVDDIAE(rep_dim=...)` (encoder+decoder)
- **Projection heads**: one linear head per cluster \(k\), mapping `rep -> z_k`
- **Hyperbolic embedding**: each `z_k` is mapped to the Poincaré ball (via `expmap0` inside `HyperbolicMultiSphereSVDD.project_all_h`)
- **Centers**: `c_h` in hyperbolic space, one per cluster
- **Radii**: `R[k]` per cluster (soft‑boundary objective)

---

## Hyperbolic geometry (what is computed)

### Poincaré ball projection (exp map at origin)

The projection heads output a Euclidean tangent vector \(v \in \mathbb{R}^{d}\).  
It is mapped to the Poincaré ball of curvature \(c>0\) by:

\[
\mathrm{expmap}_0(v)= \tanh(\sqrt{c}\,\|v\|)\,\frac{v}{\sqrt{c}\,\|v\|},
\]

then clipped to remain inside the open ball \(\|x\| < 1/\sqrt{c}\) (numerical projection).

This is implemented in `hySpCheck/hyperbolic_ops.py` as `expmap0()` + `proj_ball()`.

### Hyperbolic distance used in training

Distances are geodesic distances in the Poincaré ball:

\[
d_{\mathbb{B}}(x,y)=\frac{1}{\sqrt{c}}\operatorname{arcosh}\!\left(1 + \frac{2c\|x-y\|^2}{(1-c\|x\|^2)(1-c\|y\|^2)}\right).
\]

The trainer uses **squared** distance:
\[
d^2(x,y) = d_{\mathbb{B}}(x,y)^2
\]

via `hyp_distance(..., c=curvature) ** 2`.

---

## Stage‑2 training algorithm (current implementation)

For each epoch:

1. **Forward**
   - `rep, recon = model(x_scaled)`
   - `z_all_h = model.project_all_h(rep)`  → shape `(B, K, z_dim)`
2. **Distances to all centers**
   - `dist_sq_all = dist_sq_to_all_centers(z_all_h, c_h, curvature)` → `(B, K)`
3. **Soft assignments**
   - `p = softmax( -dist_sq_all / temperature )` → `(B, K)`
4. **Loss terms**
   - **Reconstruction**:
     - `rec = recon_mse_loss(recon, x_scaled)`
   - **Weighted SVDD (soft-boundary or one-class)**:
     - soft‑boundary per‑cluster term:
       \[
       \ell_{ik} = R_k^2 + \frac{1}{\nu}\max(0, d_{ik}^2 - R_k^2)
       \]
     - weighted:
       \[
       L_{svdd} = \mathbb{E}_i \sum_k p_{ik}\,\ell_{ik}
       \]
   - **Separation (anti-ambiguity)**:
     - let \(d_1 \le d_2\) be the two smallest distances for a sample
     - `sep = mean( relu(d1 - d2 + sep_margin) )`
   - **Overlap penalty (center spheres shouldn’t overlap)**:
     - penalize violations of: \(d(c_a,c_b) \ge R_a + R_b + margin\)
   - **Entropy regularization (anti-collapse)**:
     - entropy per sample: \(H(p_i)= -\sum_k p_{ik}\log(p_{ik})\)
     - implemented as `entropy_loss = -mean(H(p))` so **adding** it with positive weight pushes higher entropy
5. **Total loss**
   - `loss = rec + lambda_svdd*svdd + lambda_sep*sep + lambda_overlap*ov + lambda_entropy*entropy_loss`
6. **Radius update (after warmup)**
   - recompute `dist_sq_all` on the full training set (eval loader)
   - hard-assign by nearest center: `assign = argmin_k dist_sq_all[:,k]`
   - update each radius using a quantile: `new_R[k] = quantile(d_k, 1-nu)`
   - **stabilization**:
     - EMA smoothing: `ema_R = radius_ema*R + (1-radius_ema)*new_R`
     - max step clipping per epoch: `|ΔR| <= radius_max_step`
7. **Optional: cluster sorting / maintenance (feature-flagged)**
   - triggers after warmup when radii update triggers
   - logs: `[SORT] epoch=... under=... over=... reseeded=... merged=...`
   - does:
     - **divide**: reseed tiny clusters using farthest points from oversized clusters
     - **aggregate**: merge remaining tiny clusters to nearest populated center
8. **Early stopping (feature-flagged)**
   - selection metric: Hungarian accuracy if available, else macro AUC
   - stop when no improvement for `early_stop_patience` eval checks

---

## Current unsupervised features

- **Soft assignments**: `p = softmax(-dist_sq_all / temperature)`
- **Weighted SVDD**: expectation of SVDD penalty over all clusters using `p`
- **Separation loss**: margin between closest and 2nd‑closest centers
- **Entropy regularization**: prevents single-cluster collapse
- **Overlap penalty**: pushes sphere centers apart in hyperbolic distance
- **Unsupervised radius update**: radii updated from nearest-center assignments (no labels)
- **Radius stabilization**:
  - EMA smoothing (`radius_ema`)
  - max per‑epoch step (`radius_max_step`)
- **Cluster sorting (optional)**:
  - split oversized clusters / merge tiny ones (flagged)
  - debug logs emitted when enabled
- **Deterministic seeding (optional)**:
  - `--seed`, `--deterministic`
  - auto sets `CUBLAS_WORKSPACE_CONFIG` when deterministic is enabled
- **Unsupervised evaluation metrics**:
  - Hungarian‑matched accuracy
  - NMI, ARI
  - still prints legacy macro AUC

---

## Hyperparameters (CLI reference)

### Data / run control
- `--mnist_processed_dir`: dataset root containing `class_0 ... class_9` png folders
- `--xp_path`: output run directory
- `--device`: `cuda` or `cpu`
- `--digits`: `all` or comma list
- `--train_fraction`: hash-based split fraction
- `--batch_size`
- `--n_jobs_dataloader`
- `--max_train_samples`, `--max_test_samples`

### Model size / geometry
- `--rep_dim`: encoder representation dim
- `--z_dim`: embedding dim per head
- `--curvature`: hyperbolic curvature \(c\) (Poincaré ball)

### Optim
- `--ae_lr`, `--ae_n_epochs`
- `--lr`, `--svdd_n_epochs`
- `--weight_decay`

### Objectives / weights
- `--objective`: `soft-boundary` or `one-class`
- `--nu`: soft-boundary parameter (also used for radius quantile)
- `--lambda_svdd`: SVDD weight
- `--lambda_sep`: separation weight
- `--sep_margin`: separation margin
- `--lambda_entropy`: entropy weight
- `--temperature`: assignment temperature \(T\)

### Overlap penalty
- `--lambda_overlap`
- `--margin_overlap`

### Radius update stabilization
- `--warm_up_n_epochs`: start radius update when `epoch > warm_up_n_epochs`
- `--radius_ema`: EMA smoothing factor for radii
- `--radius_max_step`: per-epoch max change for radii

### Evaluation / plotting
- `--eval_every`
- `--plot_embeddings`
- `--plot_out`

### Determinism
- `--seed`
- `--deterministic`

### Early stopping
- `--early_stop_patience` (0 disables)
- `--early_stop_min_delta`

### Cluster sorting (optional)
- `--enable_cluster_sorting`
- `--sort_every`
- `--sort_min_frac`
- `--sort_max_frac`

---

## Practical notes

- If you set `warm_up_n_epochs` too high and enable early stopping, training can stop **before** radius update / sorting ever runs. For sorting tests, either lower warmup or disable early stop.
- Macro AUC in this script assumes a fixed cluster index ↔ digit index; in unsupervised clustering that mapping is not guaranteed. Prefer Hungarian/NMI/ARI for clustering quality.

---

## How AUC is calculated (current code)

The evaluation computes **one-vs-rest AUC per cluster index**, then macro-averages.

For each test sample \(x_i\) with true digit label \(y_i \in \{0,\dots,9\}\):

1. Compute distances to all clusters \(d^2_{ik}\).
2. Convert to “anomaly score” per cluster \(k\):
   - **soft-boundary**: \(s_{ik}=d^2_{ik}-R_k^2\)
   - **one-class**: \(s_{ik}=d^2_{ik}\)
3. For each \(k\), define binary targets:
   \[
   t_i^{(k)} = \mathbb{1}[y_i \neq k]
   \]
   i.e. digit \(k\) is the “inlier” class for that AUC, all other digits are outliers.
4. Compute `roc_auc_score(t^(k), s[:,k])` for each \(k\).
5. **Macro AUC** is the mean of valid per-\(k\) AUCs.

### Important caveat (permutation issue)

In unsupervised training, cluster index \(k\) is **not guaranteed** to correspond to digit \(k\).  
So macro AUC can be **underestimated** even when clustering is good (index mismatch).

That’s why the script also reports:
- Hungarian‑matched accuracy (index‑free)
- NMI / ARI (index‑free)

