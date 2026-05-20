#!/bin/bash
set -e
ENV_NAME="PredPPI"

conda create --name $ENV_NAME python=3.12 --no-default-packages -y
conda run -n $ENV_NAME pip install -r requirements.txt --no-cache-dir --isolated
echo "✅ Environment '$ENV_NAME' ready. Run: conda activate $ENV_NAME"
