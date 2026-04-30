# Multi-Sphere AllNormal (MNIST)

This folder should be used through one entrypoint:

- `hySpUnsup/train_hyp_mnist_unsup_unified.py`

## What each file means

- `train_hyp_mnist_unsup_unified.py`: canonical runner. Use this in all commands.
- `train_hyp_mnist_unsup_v3.py`: adaptive split/prune implementation used by unified.
- `train_hyp_mnist_unsup_v2.py`: base training/eval/export pipeline used by v3.
- `train_hyp_mnist_unsup.py`: shared helper functions used by v2.

If you only care about running experiments, use unified and ignore the others.

## Core options (tight explanation)

- `--mnist_processed_dir`: dataset root (`class_0 ... class_9` folders).
- `--xp_path`: output run directory.
- `--device`: `cuda` or `cpu`.
- `--geometry`: use `euclidean` unless you explicitly want hyperbolic.
- `--normal_digits`: normal class set used for training.
- `--digits`: evaluation/test digits (`all` for strict OCC reporting).
- `--strict_paper_reporting`: enables binary normal-vs-rest reporting fields.
- `--ae_n_epochs`: stage-1 AE epochs.
- `--svdd_n_epochs`: stage-2 SVDD epochs.
- `--n_spheres`: initial sphere/head count.
- `--nu`: controls radius quantile and prune pressure.

### Split/prune schedule options

- `--schedule_split_start_epoch`: first epoch where split can run.
- `--schedule_split_every_epochs`: split cadence.
- `--schedule_wait_after_split_epochs`: extra cooldown before prune after split.
- `--schedule_prune_start_epoch`: first epoch where prune can run.
- `--schedule_prune_every_epochs`: prune cadence.
- `--post_split_stability_epochs`: protect split/source clusters from immediate prune.
- `--min_active_clusters`: hard floor to prevent collapse.
- `--inline_split_min_members`: minimum cluster size to be split.
- `--inline_split_max_per_epoch`: split budget per split epoch.
- `--silhouette_prune_threshold`: prune if silhouette below this.
- `--silhouette_split_threshold`: split trigger on low silhouette.
- `--silhouette_split_negative_fraction`: split trigger on high negative-silhouette ratio.

### Export options

- `--tsne_views all`: exports all t-SNE views (`hulls`, `union`, `paper_occ`).
- `--export_cluster_samples N`: save N samples per active cluster.
- `--export_hotspot_analysis`: save class/cluster density hotspot plots.
- `--export_cluster_neural_hotspots`: save saliency + activation hotspots.

## Sample run commands

Run from `samplesvdd/`.

### 1) Quick sanity (CPU, tiny)

```bash
python hySpUnsup/train_hyp_mnist_unsup_unified.py --mnist_processed_dir "/home/josepcec/Downloads/OASPL/svDD/CVAEChecked/Data/MNIST_processed" --xp_path "/home/josepcec/Downloads/OASPL/svDD/samplesvdd/hySpUnsup/runs/smoke" --device cpu --geometry euclidean --normal_digits 0 --digits all --strict_paper_reporting --ae_n_epochs 5 --svdd_n_epochs 5 --max_train_samples 512 --max_test_samples 512
```

### 2) Standard binary OCC (0 normal vs rest)

```bash
python hySpUnsup/train_hyp_mnist_unsup_unified.py --mnist_processed_dir "/home/josepcec/Downloads/OASPL/svDD/CVAEChecked/Data/MNIST_processed" --xp_path "/home/josepcec/Downloads/OASPL/svDD/samplesvdd/hySpUnsup/runs/norm0_binary" --device cuda --geometry euclidean --normal_digits 0 --digits all --strict_paper_reporting --ae_n_epochs 100 --svdd_n_epochs 25
```

### 3) Stable scheduled split/prune (recommended long run)

```bash
python hySpUnsup/train_hyp_mnist_unsup_unified.py --mnist_processed_dir "/home/josepcec/Downloads/OASPL/svDD/CVAEChecked/Data/MNIST_processed" --xp_path "/home/josepcec/Downloads/OASPL/svDD/samplesvdd/hySpUnsup/runs/norm0_euclid_sched" --device cuda --geometry euclidean --normal_digits 0 --digits all --strict_paper_reporting --ae_n_epochs 250 --svdd_n_epochs 100 --warm_up_n_epochs 5 --n_spheres 100 --post_split_stability_epochs 5 --min_active_clusters 3 --schedule_split_start_epoch 10 --schedule_wait_after_split_epochs 5 --schedule_split_every_epochs 6 --schedule_prune_start_epoch 15 --schedule_prune_every_epochs 3 --silhouette_prune_threshold -0.10 --silhouette_split_threshold 0.15 --silhouette_split_negative_fraction 0.15 --inline_split_min_members 48 --inline_split_max_per_epoch 3
```

### 4) Export-heavy analysis run

```bash
python hySpUnsup/train_hyp_mnist_unsup_unified.py --mnist_processed_dir "/home/josepcec/Downloads/OASPL/svDD/CVAEChecked/Data/MNIST_processed" --xp_path "/home/josepcec/Downloads/OASPL/svDD/samplesvdd/hySpUnsup/runs/analysis_full" --device cuda --geometry euclidean --normal_digits 0 --digits all --strict_paper_reporting --ae_n_epochs 100 --svdd_n_epochs 25 --tsne_views all --export_cluster_samples 40 --export_hotspot_analysis --export_cluster_neural_hotspots
```

## Output structure

Inside `--xp_path`:

- `results.json`: full run metrics/history/config dump.
- `checkpoint_best.pth`, `checkpoint_latest.pth`, `checkpoint_v2.pth`.
- `tsne_unsup_v2_nearest_z*.png` (+ view-specific variants).
- Optional export folders:
  - `cluster_exports/`
  - `hotspot_analysis/`
  - `cluster_neural_hotspots/`
