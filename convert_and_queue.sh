#!/bin/bash

# --- Help Function ---
show_help() {
cat << EOF
Usage: ./convert_and_queue.sh [NOTEBOOK_NAME] [QUEUE_SCRIPT_PATH]

Description:
    1. Extracts code from ./notebooks/[NOTEBOOK_NAME].ipynb
    2. Saves it as a .py script in ./scripts/
    3. Submits the job to SLURM using the specified queue script.

Arguments:
    NOTEBOOK_NAME        The filename of your notebook (WITHOUT .ipynb).
    QUEUE_SCRIPT_PATH    Relative path to the .sh file (e.g., scripts/queue_gpu.sh).

Example:
    ./convert_and_queue.sh nn_torch scripts/queue_job_gpu.sh
EOF
}

# 1. Handle Help and Missing Arguments
if [[ "$1" == "-h" || "$1" == "--help" || -z "$1" ]]; then
    show_help
    exit 0
fi

# 2. Configuration
NOTEBOOK_NAME=$1
QUEUE_SCRIPT=${2:-"scripts/queue_job_cpu.sh"} # Defaults to CPU if 2nd arg is empty
NOTEBOOK_PATH="./notebooks/${NOTEBOOK_NAME}.ipynb"
SCRIPT_DIR="./scripts"
OUTPUT_SCRIPT="${SCRIPT_DIR}/${NOTEBOOK_NAME}.py"

# 3. Validation
if [ ! -f "$NOTEBOOK_PATH" ]; then
    echo "❌ Error: Notebook not found at $NOTEBOOK_PATH"
    exit 1
fi

if [ ! -f "$QUEUE_SCRIPT" ]; then
    echo "❌ Error: Queue script not found at $QUEUE_SCRIPT"
    exit 1
fi

# Ensure log directory exists
mkdir -p ./logs

echo "🔄 Converting $NOTEBOOK_PATH to Python..."

# 4. Python-based Conversion
python3 -c "
import json, sys, re
try:
    with open('$NOTEBOOK_PATH', 'r') as f:
        data = json.load(f)
    with open('$OUTPUT_SCRIPT', 'w') as f:
        for cell in data['cells']:
            if cell['cell_type'] == 'code':
                for line in cell['source']:
                    # 1. Comment out Jupyter magics (%)
                    if line.strip().startswith('%'):
                        f.write(f'# {line}')
                    else:
                        # 2. Replace display(...) with print(...)
                        # This regex looks for 'display(' and replaces it with 'print('
                        clean_line = re.sub(r'\bdisplay\(', 'print(', line)
                        f.write(clean_line)
                f.write('\n\n')
except Exception as e:
    print(f'Conversion failed: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Error: Conversion failed."
    exit 1
fi

echo "✅ Script created: $OUTPUT_SCRIPT"

# 5. Submit to SLURM using the provided script
echo "🚀 Submitting to queue using $QUEUE_SCRIPT..."
sbatch "$QUEUE_SCRIPT" "$OUTPUT_SCRIPT"