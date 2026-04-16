#!/usr/bin/env bash
set -euo pipefail

# Run from repo root:
#   bash samplesvdd/hySpCheck/run_excl_sensitivity_with_viz.sh
#
# Behavior:
# 1) Activates conda env lsAirsim (CUDA-capable env).
# 2) Runs ONE exclusion-sensitivity training + visualization as smoke test.
# 3) If smoke succeeds, runs the remaining exclusion-sensitivity trainings + visualizations.
# 4) Stores per-run train logs and extracts SVDD loss trend to CSV/PNG.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKDIR="${REPO_ROOT}/samplesvdd/hySpCheck"
cd "${WORKDIR}"

MNIST_DATA="${REPO_ROOT}/CVAEChecked/Data/MNIST_processed"
AE_STAGE1="${WORKDIR}/runs/ablate_mnist_full_t2_ae150/ae_stage1_ep150.pth"
DEVICE="${DEVICE:-cuda}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# Exclusion sweep from commands.txt
LAMBDAS=("0" "1e-4" "1e-3" "1e-2" "1e-1")
# First run as smoke/capability test before doing the rest.
SMOKE_LAMBDA="1e-4"

ensure_conda_env() {
  if ! command -v conda >/dev/null 2>&1; then
    # Try loading user shell init in non-interactive execution.
    [[ -f "${HOME}/.bashrc" ]] && source "${HOME}/.bashrc" || true
    [[ -f "${HOME}/.bash_profile" ]] && source "${HOME}/.bash_profile" || true
  fi

  if ! command -v conda >/dev/null 2>&1; then
    # Fallback to common conda install paths.
    local c
    for c in \
      "${HOME}/miniconda3/etc/profile.d/conda.sh" \
      "${HOME}/anaconda3/etc/profile.d/conda.sh" \
      "${HOME}/mambaforge/etc/profile.d/conda.sh" \
      "${HOME}/micromamba/etc/profile.d/conda.sh"; do
      if [[ -f "${c}" ]]; then
        # shellcheck disable=SC1090
        source "${c}"
        break
      fi
    done
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "[ERR] conda not found in PATH after bootstrap."
    exit 1
  fi

  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate lsAirsim
  echo "[INFO] Active conda env: ${CONDA_DEFAULT_ENV:-unknown}"
}

check_paths() {
  [[ -d "${MNIST_DATA}" ]] || { echo "[ERR] Missing MNIST dir: ${MNIST_DATA}"; exit 1; }
  [[ -f "${AE_STAGE1}" ]] || { echo "[ERR] Missing AE checkpoint: ${AE_STAGE1}"; exit 1; }
}

run_one() {
  local lambda_excl="$1"
  local tag="sens_mnist_excl_lambda${lambda_excl}"
  local run_dir="${WORKDIR}/runs/${tag}"
  local vis_out="${WORKDIR}/runs/suite_vis_excl/${tag}"
  local train_log="${run_dir}/train.log"
  local best_ckpt="${run_dir}/checkpoint_best.pth"

  mkdir -p "${run_dir}" "${vis_out}"

  echo "============================================================"
  echo "[RUN] lambda_excl=${lambda_excl}"
  echo "[RUN] xp_path=${run_dir}"
  echo "============================================================"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${best_ckpt}" ]]; then
    echo "[INFO] Reusing existing checkpoint: ${best_ckpt}"
  else
    python train_hyp_mnist_multi.py \
      --mnist_processed_dir "${MNIST_DATA}" \
      --skip_ae_pretrain \
      --ae_stage1_checkpoint_path "${AE_STAGE1}" \
      --xp_path "${run_dir}" \
      --svdd_n_epochs 100 \
      --lambda_excl "${lambda_excl}" \
      --margin_excl 0.1 \
      --lambda_overlap 1e-2 \
      --margin_overlap 0.05 \
      --device "${DEVICE}" 2>&1 | tee "${train_log}"
  fi

  if [[ ! -f "${best_ckpt}" ]]; then
    echo "[ERR] Missing best checkpoint after training: ${best_ckpt}"
    exit 1
  fi

  python visualize_hyp_mnist_suite.py \
    --mnist_processed_dir "${MNIST_DATA}" \
    --out_dir "${vis_out}" \
    --checkpoints "${tag}=${best_ckpt}" \
    --device "${DEVICE}" 2>&1 | tee "${vis_out}/visualize.log"

  # Extract epoch loss trend from training log.
  python - "${train_log}" "${run_dir}" <<'PY'
import csv
import os
import re
import sys

import matplotlib.pyplot as plt

log_path = sys.argv[1]
out_dir = sys.argv[2]

pat = re.compile(
    r"\[SVDD-H\]\s+(\d+)/(\d+)\s+loss=([0-9.eE+-]+)\s+rec=([0-9.eE+-]+)\s+svdd=([0-9.eE+-]+)\s+excl=([0-9.eE+-]+)\s+overlap=([0-9.eE+-]+)\s+R_mean=([0-9.eE+-]+)"
)

rows = []
with open(log_path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        m = pat.search(line)
        if m:
            rows.append(
                {
                    "epoch": int(m.group(1)),
                    "epoch_total": int(m.group(2)),
                    "loss": float(m.group(3)),
                    "recon": float(m.group(4)),
                    "svdd": float(m.group(5)),
                    "excl": float(m.group(6)),
                    "overlap": float(m.group(7)),
                    "R_mean": float(m.group(8)),
                }
            )

csv_path = os.path.join(out_dir, "loss_trend.csv")
if rows:
    with open(csv_path, "w", newline="", encoding="utf-8") as fw:
        wr = csv.DictWriter(
            fw, fieldnames=["epoch", "epoch_total", "loss", "recon", "svdd", "excl", "overlap", "R_mean"]
        )
        wr.writeheader()
        wr.writerows(rows)

    epochs = [r["epoch"] for r in rows]
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, [r["loss"] for r in rows], label="total_loss", linewidth=1.8)
    plt.plot(epochs, [r["recon"] for r in rows], label="recon", linewidth=1.1)
    plt.plot(epochs, [r["svdd"] for r in rows], label="svdd", linewidth=1.1)
    plt.plot(epochs, [r["excl"] for r in rows], label="excl", linewidth=1.1)
    plt.plot(epochs, [r["overlap"] for r in rows], label="overlap", linewidth=1.1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SVDD training loss trend")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_trend.png"), dpi=160)
    plt.close()
    print(f"[INFO] wrote {csv_path}")
else:
    print(f"[WARN] no SVDD loss lines parsed from {log_path}")
PY
}

main() {
  ensure_conda_env
  check_paths

  echo "[INFO] repo root: ${REPO_ROOT}"
  echo "[INFO] workdir: ${WORKDIR}"
  echo "[INFO] device: ${DEVICE}"
  nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

  # 1) smoke test run first
  run_one "${SMOKE_LAMBDA}"

  # 2) continue with the rest (excluding smoke lambda)
  for l in "${LAMBDAS[@]}"; do
    if [[ "${l}" == "${SMOKE_LAMBDA}" ]]; then
      continue
    fi
    run_one "${l}"
  done

  echo "[DONE] Exclusion sensitivity sweep + visualization complete."
}

main "$@"
