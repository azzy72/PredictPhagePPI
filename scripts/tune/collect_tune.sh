#!/usr/bin/env bash
# Walk every run folder under FFNN_D1_on_D2_optim/, parse its log_run*.txt,
# write a summary CSV + per-metric plots into a 'summary/' subdir.
#
# Reuses the metrics extraction + plotting logic from collect_iterres.py,
# but skips the gene-annotation / top_kmers path (these runs were launched
# without --perform_pfi, so no normalized_interaction.csv exists).

set -euo pipefail

BASE_DIR="${1:-FFNN_D1_on_D2_optim}"
OUT_DIR="${2:-${BASE_DIR}/summary}"

# Where this script lives — so we find the Python collector next to it.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COLLECTOR="${SCRIPT_DIR}/collect_metrics.py"

if [[ ! -d "${BASE_DIR}" ]]; then
  echo "Base dir '${BASE_DIR}' not found." >&2
  exit 1
fi
if [[ ! -f "${COLLECTOR}" ]]; then
  echo "Python collector not found at ${COLLECTOR}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "Collecting sweep results"
echo "  base_dir : ${BASE_DIR}"
echo "  out_dir  : ${OUT_DIR}"

python3 "${COLLECTOR}" \
  --base_dir "${BASE_DIR}" \
  --out_dir  "${OUT_DIR}"

echo
echo "Done. See:"
echo "  ${OUT_DIR}/sweep_summary.csv          (per-run metrics)"
echo "  ${OUT_DIR}/*_by_run.png               (bar plot per metric)"
echo "  ${OUT_DIR}/*_vs_tag_{kf,ep,lr}.png    (metric vs hyperparameter)"
echo "  ${OUT_DIR}/confusion_matrix_by_run.png"
echo "  ${OUT_DIR}/averaged_confusion_matrix.png"
echo "  ${OUT_DIR}/collect_metrics.log"