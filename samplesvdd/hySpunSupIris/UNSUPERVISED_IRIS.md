# Iris: purely unsupervised hyperbolic multi-sphere SVDD

This folder implements the **same unsupervised objective** as `hySpCheckunSup/train_hyp_mnist_multi.py`, adapted to **Iris** (small \(n\), fast sanity checks), aligned with the **multi-sphere SVDD** idea: **one shared encoder**, **\(K\) hyperspheres** in hyperbolic space (Poincaré ball), trained jointly (extension of the Deep Multi-sphere SVDD paper setting to **all classes in one model**).

## What is purely unsupervised (training)

- **No label-conditioned preprocessing**: features are `StandardScaler` fit **only on the training split** (no per-class scaling).
- **No label-based center init**: centers `c_h[k]` are initialized from **unsupervised** statistics of `project_all_h(rep)` (per-head mean in the ball, then `proj_ball`), not from true species labels.
- **Stage-2 loss** uses only reconstruction + soft SVDD + separation + overlap + entropy (+ optional sorting), with batches `for x, _ in loader` (labels ignored in the loss).

Labels are used **only for evaluation metrics** (NMI, ARI, Hungarian accuracy, optional one-vs-rest AUC) and **optional** plot coloring.

## Hyperparameters

See `train_iris_hyp_unsup.py --help`. Defaults target \(K=3\) spheres (Iris species count); you can set `--n_clusters` for ablations.

### Example (full run + plots)

```bash
python train_iris_hyp_unsup.py --xp_path runs/iris_unsup_hyp --plot_embeddings --plot_tsne --seed 42
```

- Add `--plot_color_true` to color figures by **true** species (visualization only).
- Add `--stratify_split` if you want a stratified train/test split (uses labels **only** for splitting).

## Metrics: two macro-AUCs (unsupervised eval)

- **`best_macro_auc` (fixed index)** — For each sphere column \(k\), one-vs-rest AUC with target “not class \(k\)”, **assuming** cluster index \(k\) = true class \(k\). Misleading if the learned permutation of spheres ≠ identity (common when h_acc/NMI/ARI are good).
- **`best_macro_auc_hungarian_aligned`** — Same one-vs-rest definition, but the score column for true class \(\ell\) is the sphere **matched to \(\ell\)** by the Hungarian alignment on the confusion matrix. **Use this** to compare with `h_acc` / clustering quality.

`results.json` also stores `per_class_auc` (fixed) and `per_class_auc_hungarian_aligned`.

## Outputs

- `checkpoint_best.pth`, `checkpoint_latest.pth`, `results.json`
- `poincare_clusters.png` — PCA 2D of selected per-sample embeddings + centers + radii (approximate circles in PCA space)
- `poincare_inline_outline.png` — **\(K\) panels in one row**: unit disk outline per sphere index, center + radius, points (by default colored by **predicted** cluster)
- `tsne_clusters.png` — t-SNE of selected embeddings (optional / same style as prior Iris demos)

## Paper alignment (short)

- **Single model** maps inputs to a representation and **\(K\) class-specific heads** → embeddings; each class has a **hypersphere** (here in hyperbolic space via geodesic distance to center + radius), trained with SVDD-style objectives and multi-sphere interaction (overlap penalty), consistent with the multi-sphere SVDD family—here with **soft assignments** for unsupervised training.
