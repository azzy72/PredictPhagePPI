#!/bin/bash
# Hyperparameter sweep for FFNN_inner.py
# Each run writes to its own output dir so results don't clobber each other.

#SBATCH --job-name=TuneFFNN
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1
#SBATCH --gres=shard:1
#SBATCH --time=08:00:00
#SBATCH --output=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.out
#SBATCH --error=/home/projects/s215045/PredictPhagePPI/tmp/%j-%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s215045@student.dtu.dk

set -euo pipefail
 
ROOT_DIR=$(git rev-parse --show-toplevel)

# ---- Hyperparameter grid ----------------------------------------------------
KF_SPLITS=(3 6)
EPOCHS=(50 100)
LRS=(1e-3 5e-4 1e-4)
WEIGHT_DECAYS=(0 1e-4 1e-2)
 
# Flatten the 3-D grid into a 1-D list of "kf ep lr" triples.
COMBOS=()
for kf in "${KF_SPLITS[@]}"; do
  for ep in "${EPOCHS[@]}"; do
    for lr in "${LRS[@]}"; do
      for wd in "${WEIGHT_DECAYS[@]}"; do
        COMBOS+=("${kf} ${ep} ${lr} ${wd}")
      done
    done
  done
done
 
TOTAL=${#COMBOS[@]}
IDX=${SLURM_ARRAY_TASK_ID:?must be run as a job array}
 
if (( IDX >= TOTAL )); then
  echo "Array index ${IDX} >= ${TOTAL} combos, nothing to do."
  exit 0
fi
 
read -r KF EP LR WD <<< "${COMBOS[$IDX]}"
TAG="kf${KF}_ep${EP}_lr${LR}_wd${WD}"
OUT_DIR="$ROOT_DIR/nn_runs/FFNN_D1_on_D2_optim/${TAG}/"
mkdir -p "${OUT_DIR}" "$ROOT_DIR/nn_runs/FFNN_D1_on_D2_optim/sweep_logs"
 
echo "[${IDX}/${TOTAL}] ${TAG} on $(hostname)  $(date -Is)"
nvidia-smi -L || true
 
# ---- Environment ------------------------------------------------------------
# TODO: load whatever modules / activate whatever env your cluster uses, e.g.
# module load cuda/12.1 python/3.11
# source ~/venvs/ffnn/bin/activate
 
# ---- Run --------------------------------------------------------------------
python3 "${ROOT_DIR}/scripts/FFNN_inner2.py" \
  --nk 500 12 \
  --use_encoded \
  --train_d1_test_d2 \
  --cv \
  --force_presmat \
  --patience 30 \
  --kf_n_splits "${KF}" \
  --n_epochs "${EP}" \
  --learning_rate "${LR}" \
  --weight_decay "${WD}" \
  --logging \
  --out "${OUT_DIR}"

 