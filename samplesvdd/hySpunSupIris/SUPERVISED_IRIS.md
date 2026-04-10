# Supervised Iris — hyperbolic multi-sphere SVDD (`hySpCheck`-style)

Script: `train_iris_hyp_sup.py`

Mirrors `hySpCheck/train_hyp_mnist_multi.py`:

| Stage | Content |
|--------|---------|
| **AE** | Reconstruction only (`recon_mse_loss`); labels not used in the loss. |
| **Init** | `init_centers_h`: per-class mean of `project_self_h(rep, y)` on the **train** set (supervised). |
| **SVDD** | `z_h = project_self_h(rep, y)`, `dist_sq` to `c_y`; `svdd_loss_soft_boundary` or one-class. |
| **Exclusion** | `inter_class_exclusion_loss` on `dist_sq_to_all_centers` vs non-true spheres. |
| **Overlap** | `sphere_overlap_penalty` between centers in hyperbolic distance. |
| **Radii** | After `warm_up_n_epochs`, `update_radii` from per-class `dist_sq` lists (labeled). |

**Preprocessing:** `StandardScaler` fit on train only (tabular Iris; unlike MNIST there is no per-class min–max in this script).

**Eval:** one-vs-rest macro AUC per class index using **hyperbolic** `dist_sq_to_all_centers` scores (aligned with training geometry).

## Run

```bash
python train_iris_hyp_sup.py --xp_path runs/iris_sup_hyp --stratify_split --seed 42
```

Figures (same names as unsupervised) are written only if you pass **`--plot_embeddings`** and/or **`--plot_tsne`**:

- `poincare_clusters.png` — PCA disk + centers + radii  
- `per_class_inline_outline.png` — one Poincaré panel per sphere (points assigned to that sphere)  
- `tsne_clusters.png` — t-SNE of nearest-sphere embeddings  

Plots load **`checkpoint_best.pth`** if present (else final weights). Default point color = **true species**; use `--plot_color_pred` to color by **nearest-sphere** assignment.

Optional: `--skip_ae_pretrain --ae_stage1_checkpoint_path path/to/iris_ae.pth` (must match same `IrisMLPSVDDIAE` + `HyperbolicMultiSphereSVDD` shapes).
