#!/usr/bin/env bash
# run_compare_sigs.sh  –  thin bash wrapper around compare_sigs.py
#
# Usage:
#   bash run_compare_sigs.sh                   # use default config.yaml
#   bash run_compare_sigs.sh --config my.yaml  # custom config
#   bash run_compare_sigs.sh --dry-run         # print commands, don't run
#   bash run_compare_sigs.sh --force           # ignore existing output files and overwrite them
#
# The script resolves its own directory so it can be called from anywhere.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

# ── Dependency check ──────────────────────────────────────────────────────────
for dep in sourmash "$PYTHON"; do
    if ! command -v "$dep" &>/dev/null; then
        echo "[ERROR] Required tool not found in PATH: $dep" >&2
        exit 1
    fi
done

if ! "$PYTHON" -c "import yaml" &>/dev/null; then
    echo "[ERROR] Python package 'pyyaml' is not installed." >&2
    echo "        Install with:  pip install pyyaml" >&2
    exit 1
fi

# ── Hand off to Python ────────────────────────────────────────────────────────
exec "$PYTHON" "$SCRIPT_DIR/compare_sigs.py" "$@"
