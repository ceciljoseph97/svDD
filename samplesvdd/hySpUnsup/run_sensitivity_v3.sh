#!/usr/bin/env bash
set -euo pipefail

DATA="/home/josepcec/Downloads/OASPL/svDD/CVAEChecked/Data/MNIST_processed"
AECKPT="samplesvdd/hySpUnsup/runs/my_v3_ae200_stage1/ae_stage1_ep200.pth"
SCRIPT="samplesvdd/hySpUnsup/train_hyp_mnist_unsup_v3.py"
REPLAY_SCRIPT="samplesvdd/hySpUnsup/replay_checkpoint_viz_eval.py"
RUNROOT="samplesvdd/hySpUnsup/runs/sensitivity_v3_2"

DEVICE="cuda"
SVDD_EPOCHS=20
EXPORT_SAMPLES=20
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

# W&B replay logging (set WANDB_USE=1 to enable)
WANDB_USE=1
WANDB_PROJECT="hyspunsup-sensitivity"
WANDB_ENTITY=""
WANDB_GROUP="sensitivity_v3_2"
WANDB_TAGS="sensitivity,v3,replay"
WANDB_MODE="online"

mkdir -p "$RUNROOT"

if [[ ! -f "$AECKPT" ]]; then
  echo "Missing AE checkpoint: $AECKPT"
  echo "Update AECKPT in this script or generate the checkpoint first."
  exit 1
fi
if [[ ! -f "$REPLAY_SCRIPT" ]]; then
  echo "Missing replay script: $REPLAY_SCRIPT"
  echo "Create it first (replay_checkpoint_viz_eval.py)."
  exit 1
fi

run_exp() {
  local xp_path="$1"
  shift

  echo
  echo "============================================================"
  echo "Running: $xp_path"
  echo "============================================================"

  # Skip already-finished runs to make interrupted sweeps resumable.
  if [[ -f "$xp_path/results.json" ]]; then
    echo "Skipping (results.json exists): $xp_path"
    return 0
  fi

  python "$SCRIPT" \
    --mnist_processed_dir "$DATA" \
    --xp_path "$xp_path" \
    --skip_ae_pretrain \
    --ae_stage1_checkpoint_path "$AECKPT" \
    --device "$DEVICE" \
    --svdd_n_epochs "$SVDD_EPOCHS" \
    --auc_mode union \
    --skip_tsne \
    --export_cluster_samples 0 \
    "$@"
}

echo "Starting sensitivity sweeps under: $RUNROOT"

# 1. n_spheres sweep
for K in 5 10 15 20; do
  run_exp "$RUNROOT/nspheres_k${K}" \
    --n_spheres "$K" \
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
    --margin_overlap "$BASE_MARGIN_OVERLAP"
done

# 2a. lambda_overlap sweep
for LO in 0 1e-5 1e-4 1e-3 1e-2; do
  SAFE_LO="${LO//./p}"
  run_exp "$RUNROOT/overlap_lambda_${SAFE_LO}" \
    --n_spheres "$BASE_N_SPHERES" \
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

# 2b. margin_overlap sweep
for MO in 0.0 0.005 0.01 0.02 0.05; do
  SAFE_MO="${MO//./p}"
  run_exp "$RUNROOT/overlap_margin_${SAFE_MO}" \
    --n_spheres "$BASE_N_SPHERES" \
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

# 3. chaos_factor sweep
for CF in 0.0 0.05 0.10 0.15 0.25 0.35; do
  SAFE_CF="${CF//./p}"
  run_exp "$RUNROOT/chaos_${SAFE_CF}" \
    --n_spheres "$BASE_N_SPHERES" \
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

# 4a. LB sweep
for LB in 20 50 100 150; do
  run_exp "$RUNROOT/lb_${LB}" \
    --n_spheres "$BASE_N_SPHERES" \
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

# 4b. UB sweep
for UB in 0.10 0.15 0.20 0.25 0.30; do
  SAFE_UB="${UB//./p}"
  run_exp "$RUNROOT/ub_${SAFE_UB}" \
    --n_spheres "$BASE_N_SPHERES" \
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

# 4c. LB x UB grid
for LB in 20 50 100; do
  for UB in 0.10 0.20 0.30; do
    SAFE_UB="${UB//./p}"
    run_exp "$RUNROOT/grid_lb${LB}_ub${SAFE_UB}" \
      --n_spheres "$BASE_N_SPHERES" \
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

echo
echo "All sweeps finished (training only, no viz)."
echo "Runs saved under: $RUNROOT"
echo
echo "Starting final replay pass for viz/eval..."
REPLAY_ARGS=(
  --runs_root "$RUNROOT"
  --mnist_processed_dir "$DATA"
  --split both
  --device "$DEVICE"
  --auc_mode union
  --export_cluster_samples "$EXPORT_SAMPLES"
  --cluster_export_seed "$EXPORT_SEED"
  --export_hotspot_analysis
  --export_cluster_neural_hotspots
  --neural_hotspot_max_samples "$NEURAL_MAX"
)
if [[ "$WANDB_USE" == "1" ]]; then
  REPLAY_ARGS+=(
    --use_wandb
    --wandb_project "$WANDB_PROJECT"
    --wandb_group "$WANDB_GROUP"
    --wandb_tags "$WANDB_TAGS"
    --wandb_mode "$WANDB_MODE"
  )
  if [[ -n "$WANDB_ENTITY" ]]; then
    REPLAY_ARGS+=(--wandb_entity "$WANDB_ENTITY")
  fi
fi
python "$REPLAY_SCRIPT" \
  "${REPLAY_ARGS[@]}"
echo "Replay viz/eval finished."
