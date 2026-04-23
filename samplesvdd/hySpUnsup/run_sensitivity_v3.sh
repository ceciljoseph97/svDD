#!/usr/bin/env bash
set -euo pipefail

DATA="/home/josepcec/Downloads/OASPL/svDD/CVAEChecked/Data/MNIST_processed"
AECKPT="samplesvdd/hySpUnsup/runs/my_v3_ae200_stage1/ae_stage1_ep200.pth"
SCRIPT="samplesvdd/hySpUnsup/train_hyp_mnist_unsup_v3.py"
RUNROOT_HYP="samplesvdd/hySpUnsup/runs/sensitivity_v3_2"
RUNROOT_EUC="samplesvdd/hySpUnsup/runs/sensitivity_v3_2_euclidean"
GEOMETRIES=("euclidean")

DEVICE="cuda"
SVDD_EPOCHS=100
AE_EPOCHS=200
EXPORT_SAMPLES=20
EXPORT_SPLIT="test"
EXPORT_SEED=42
NEURAL_MAX=20
TSNE_MAX_SAMPLES=6000
TSNE_VIEWS="all"

BASE_OBJECTIVE="union-soft"
BASE_NU="0.05"
BASE_LAMBDA_SVDD="2e-4"
BASE_LAMBDA_OVERLAP="1e-4"
BASE_MARGIN_OVERLAP="0.01"

BASE_N_SPHERES=30
BASE_LB=50
BASE_UB="0.2"
BASE_HYBRID_ITERS=4
BASE_CHAOS="0.15"
NORMAL_SETS=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "0,1" "0,1,3" "all")

CURRENT_RUNROOT="$RUNROOT_HYP"
CURRENT_GEOMETRY="hyperbolic"

mkdir -p "$RUNROOT_HYP" "$RUNROOT_EUC"

run_exp() {
  local xp_path="$1"
  local ae_ckpt="$2"
  shift
  shift
  local results_json="${xp_path}/results.json"

  if [[ -f "$results_json" ]]; then
    echo
    echo "============================================================"
    echo "Skipping (already exists): $xp_path"
    echo "Found: $results_json"
    echo "============================================================"
    return 0
  fi

  echo
  echo "============================================================"
  echo "Running: $xp_path"
  echo "============================================================"

  python "$SCRIPT" \
    --mnist_processed_dir "$DATA" \
    --xp_path "$xp_path" \
    --skip_ae_pretrain \
    --ae_stage1_checkpoint_path "$ae_ckpt" \
    --ae_stage1_fallback_checkpoint_path "$AECKPT" \
    --geometry "$CURRENT_GEOMETRY" \
    --device "$DEVICE" \
    --svdd_n_epochs "$SVDD_EPOCHS" \
    --auc_mode union \
    --export_cluster_samples "$EXPORT_SAMPLES" \
    --cluster_export_split "$EXPORT_SPLIT" \
    --cluster_export_seed "$EXPORT_SEED" \
    --export_hotspot_analysis \
    --export_cluster_neural_hotspots \
    --neural_hotspot_max_samples "$NEURAL_MAX" \
    --tsne_views "$TSNE_VIEWS" \
    --tsne_max_samples "$TSNE_MAX_SAMPLES" \
    --inline_split_inverse_distance_threshold 2.0 \
    --inline_split_min_members 128 \
    --inline_split_max_per_epoch 1 \
    --inline_split_every 1 \
    "$@"
}

run_ae_pretrain_case() {
  local normal="$1"
  local normal_root="$2"
  local ae_ckpt="$3"
  local ae_meta="${normal_root}/ae_pretrain/results_ae_only.json"
  if [[ -f "$ae_ckpt" || -f "$ae_meta" ]]; then
    echo "[AE] Skipping pretrain for normal=${normal}; checkpoint/meta already exists."
    return 0
  fi
  echo
  echo "============================================================"
  echo "Running AE pretrain for normal=${normal}"
  echo "Saving checkpoint: ${ae_ckpt}"
  echo "============================================================"
  python "$SCRIPT" \
    --mnist_processed_dir "$DATA" \
    --xp_path "${normal_root}/ae_pretrain" \
    --geometry "$CURRENT_GEOMETRY" \
    --device "$DEVICE" \
    --ae_n_epochs "$AE_EPOCHS" \
    --save_ae_stage1_checkpoint_path "$ae_ckpt" \
    --ae_only \
    --normal_digits "$normal" \
    --digits all
}

for GEOM in "${GEOMETRIES[@]}"; do
  CURRENT_GEOMETRY="$GEOM"
  if [[ "$GEOM" == "hyperbolic" ]]; then
    CURRENT_RUNROOT="$RUNROOT_HYP"
  else
    CURRENT_RUNROOT="$RUNROOT_EUC"
  fi

  echo "Starting normal-case runs under: $CURRENT_RUNROOT"
  echo "Geometry: $CURRENT_GEOMETRY | fixed baseline config (no sweeps)"

  for NORMAL in "${NORMAL_SETS[@]}"; do
    NORMAL_TAG="${NORMAL//,/}"
    if [[ "$NORMAL" == "all" ]]; then
      NORMAL_TAG="all"
    fi
    NORMAL_ROOT="$CURRENT_RUNROOT/norm_${NORMAL_TAG}"
    mkdir -p "$NORMAL_ROOT"
    echo ">>> Normal digits run set: ${NORMAL} (tag: ${NORMAL_TAG})"
    AE_CASE_CKPT="$NORMAL_ROOT/ae_stage1_ep${AE_EPOCHS}.pth"
    run_ae_pretrain_case "$NORMAL" "$NORMAL_ROOT" "$AE_CASE_CKPT"

    run_exp "$NORMAL_ROOT/base" "$AE_CASE_CKPT" \
    --n_spheres "$BASE_N_SPHERES" \
    --normal_digits "$NORMAL" \
    --digits all \
    --objective "$BASE_OBJECTIVE" \
    --nu "$BASE_NU" \
    --lambda_svdd "$BASE_LAMBDA_SVDD" \
    --lambda_overlap "$BASE_LAMBDA_OVERLAP" \
    --margin_overlap "$BASE_MARGIN_OVERLAP"
  done
done

echo
echo "All normal-case runs finished."
echo "Runs saved under: $RUNROOT_EUC"
