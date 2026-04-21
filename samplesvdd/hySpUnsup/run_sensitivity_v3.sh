#!/usr/bin/env bash
set -euo pipefail

DATA="/home/josepcec/Downloads/OASPL/svDD/CVAEChecked/Data/MNIST_processed"
AECKPT="samplesvdd/hySpUnsup/runs/my_v3_ae200_stage1/ae_stage1_ep200.pth"
SCRIPT="samplesvdd/hySpUnsup/train_hyp_mnist_unsup_v3.py"
RUNROOT_HYP="samplesvdd/hySpUnsup/runs/sensitivity_v3_2"
RUNROOT_EUC="samplesvdd/hySpUnsup/runs/sensitivity_v3_2_euclidean"
GEOMETRIES=("hyperbolic" "euclidean")

DEVICE="cuda"
SVDD_EPOCHS=20
EXPORT_SAMPLES=20
EXPORT_SPLIT="test"
EXPORT_SEED=42
NEURAL_MAX=50

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
NORMAL_SETS=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "0,1" "0,1,2" "0,1,2,3" "all")

CURRENT_RUNROOT="$RUNROOT_HYP"
CURRENT_GEOMETRY="hyperbolic"

mkdir -p "$RUNROOT_HYP" "$RUNROOT_EUC"

run_exp() {
  local xp_path="$1"
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
    --ae_stage1_checkpoint_path "$AECKPT" \
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
    "$@"
}

for GEOM in "${GEOMETRIES[@]}"; do
  CURRENT_GEOMETRY="$GEOM"
  if [[ "$GEOM" == "hyperbolic" ]]; then
    CURRENT_RUNROOT="$RUNROOT_HYP"
  else
    CURRENT_RUNROOT="$RUNROOT_EUC"
  fi

  echo "Starting sensitivity sweeps under: $CURRENT_RUNROOT"
  echo "Geometry: $CURRENT_GEOMETRY | fixed n_spheres=$BASE_N_SPHERES | inline+post pruning enabled"

  for NORMAL in "${NORMAL_SETS[@]}"; do
    NORMAL_TAG="${NORMAL//,/}"
    if [[ "$NORMAL" == "all" ]]; then
      NORMAL_TAG="all"
    fi
    NORMAL_ROOT="$CURRENT_RUNROOT/norm_${NORMAL_TAG}"
    mkdir -p "$NORMAL_ROOT"
    echo ">>> Normal digits sweep set: ${NORMAL} (tag: ${NORMAL_TAG})"

    # 1a. lambda_overlap sweep
    for LO in 0 1e-5 1e-4 1e-3 1e-2; do
      SAFE_LO="${LO//./p}"
      run_exp "$NORMAL_ROOT/overlap_lambda_${SAFE_LO}" \
      --n_spheres "$BASE_N_SPHERES" \
      --hybrid_rebalance \
      --normal_digits "$NORMAL" \
      --digits all \
      --min_cluster_members "$BASE_LB" \
      --max_cluster_fraction "$BASE_UB" \
      --lb_min_cluster_members "$BASE_LB" \
      --ub_max_cluster_fraction "$BASE_UB" \
      --hybrid_max_iters "$BASE_HYBRID_ITERS" \
      --hard_cap_reassign \
      --chaos_factor "$BASE_CHAOS" \
      --objective "$BASE_OBJECTIVE" \
      --nu "$BASE_NU" \
      --lambda_svdd "$BASE_LAMBDA_SVDD" \
      --lambda_overlap "$LO" \
      --margin_overlap "$BASE_MARGIN_OVERLAP"
    done

    # 1b. margin_overlap sweep
    for MO in 0.0 0.005 0.01 0.02 0.05; do
      SAFE_MO="${MO//./p}"
      run_exp "$NORMAL_ROOT/overlap_margin_${SAFE_MO}" \
      --n_spheres "$BASE_N_SPHERES" \
      --hybrid_rebalance \
      --normal_digits "$NORMAL" \
      --digits all \
      --min_cluster_members "$BASE_LB" \
      --max_cluster_fraction "$BASE_UB" \
      --lb_min_cluster_members "$BASE_LB" \
      --ub_max_cluster_fraction "$BASE_UB" \
      --hybrid_max_iters "$BASE_HYBRID_ITERS" \
      --hard_cap_reassign \
      --chaos_factor "$BASE_CHAOS" \
      --objective "$BASE_OBJECTIVE" \
      --nu "$BASE_NU" \
      --lambda_svdd "$BASE_LAMBDA_SVDD" \
      --lambda_overlap "$BASE_LAMBDA_OVERLAP" \
      --margin_overlap "$MO"
    done

    # 2. chaos_factor sweep
    for CF in 0.0 0.05 0.10 0.15 0.25 0.35; do
      SAFE_CF="${CF//./p}"
      run_exp "$NORMAL_ROOT/chaos_${SAFE_CF}" \
      --n_spheres "$BASE_N_SPHERES" \
      --hybrid_rebalance \
      --normal_digits "$NORMAL" \
      --digits all \
      --min_cluster_members "$BASE_LB" \
      --max_cluster_fraction "$BASE_UB" \
      --lb_min_cluster_members "$BASE_LB" \
      --ub_max_cluster_fraction "$BASE_UB" \
      --hybrid_max_iters "$BASE_HYBRID_ITERS" \
      --hard_cap_reassign \
      --chaos_factor "$CF" \
      --objective "$BASE_OBJECTIVE" \
      --nu "$BASE_NU" \
      --lambda_svdd "$BASE_LAMBDA_SVDD" \
      --lambda_overlap "$BASE_LAMBDA_OVERLAP" \
      --margin_overlap "$BASE_MARGIN_OVERLAP"
    done

    # 3a. LB sweep
    for LB in 20 50 100 150; do
      run_exp "$NORMAL_ROOT/lb_${LB}" \
      --n_spheres "$BASE_N_SPHERES" \
      --hybrid_rebalance \
      --normal_digits "$NORMAL" \
      --digits all \
      --min_cluster_members "$LB" \
      --max_cluster_fraction "$BASE_UB" \
      --lb_min_cluster_members "$LB" \
      --ub_max_cluster_fraction "$BASE_UB" \
      --hybrid_max_iters "$BASE_HYBRID_ITERS" \
      --hard_cap_reassign \
      --chaos_factor "$BASE_CHAOS" \
      --objective "$BASE_OBJECTIVE" \
      --nu "$BASE_NU" \
      --lambda_svdd "$BASE_LAMBDA_SVDD" \
      --lambda_overlap "$BASE_LAMBDA_OVERLAP" \
      --margin_overlap "$BASE_MARGIN_OVERLAP"
    done

    # 3b. UB sweep
    for UB in 0.10 0.15 0.20 0.25 0.30; do
      SAFE_UB="${UB//./p}"
      run_exp "$NORMAL_ROOT/ub_${SAFE_UB}" \
      --n_spheres "$BASE_N_SPHERES" \
      --hybrid_rebalance \
      --normal_digits "$NORMAL" \
      --digits all \
      --min_cluster_members "$BASE_LB" \
      --max_cluster_fraction "$UB" \
      --lb_min_cluster_members "$BASE_LB" \
      --ub_max_cluster_fraction "$UB" \
      --hybrid_max_iters "$BASE_HYBRID_ITERS" \
      --hard_cap_reassign \
      --chaos_factor "$BASE_CHAOS" \
      --objective "$BASE_OBJECTIVE" \
      --nu "$BASE_NU" \
      --lambda_svdd "$BASE_LAMBDA_SVDD" \
      --lambda_overlap "$BASE_LAMBDA_OVERLAP" \
      --margin_overlap "$BASE_MARGIN_OVERLAP"
    done

    # 3c. LB x UB grid
    for LB in 20 50 100; do
      for UB in 0.10 0.20 0.30; do
        SAFE_UB="${UB//./p}"
        run_exp "$NORMAL_ROOT/grid_lb${LB}_ub${SAFE_UB}" \
        --n_spheres "$BASE_N_SPHERES" \
        --hybrid_rebalance \
        --normal_digits "$NORMAL" \
        --digits all \
        --min_cluster_members "$LB" \
        --max_cluster_fraction "$UB" \
        --lb_min_cluster_members "$LB" \
        --ub_max_cluster_fraction "$UB" \
        --hybrid_max_iters "$BASE_HYBRID_ITERS" \
        --hard_cap_reassign \
        --chaos_factor "$BASE_CHAOS" \
        --objective "$BASE_OBJECTIVE" \
        --nu "$BASE_NU" \
        --lambda_svdd "$BASE_LAMBDA_SVDD" \
        --lambda_overlap "$BASE_LAMBDA_OVERLAP" \
        --margin_overlap "$BASE_MARGIN_OVERLAP"
      done
    done
  done
done

echo
echo "All sweeps finished for hyperbolic + euclidean."
echo "Runs saved under: $RUNROOT_HYP and $RUNROOT_EUC"
