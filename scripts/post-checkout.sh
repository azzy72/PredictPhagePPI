#!/bin/bash

# 1. Get the absolute path to the repository root
# This ensures the script works no matter where you run git commands from
REPO_ROOT=$(git rev-parse --show-toplevel)

echo "--- Git Hook: Refreshing Project Structure ---"

# 2. Create the required directories at the root
# -p flag ensures no error if they already exist
mkdir -p "$REPO_ROOT/data_prod"
mkdir -p "$REPO_ROOT/data_prod/encoded_sketches"
mkdir -p "$REPO_ROOT/data_prod/SM_sketches"
mkdir -p "$REPO_ROOT/tmp"
mkdir -p "$REPO_ROOT/fig"
mkdir -p "$REPO_ROOT/nn_runs"
mkdir -p "$REPO_ROOT/cnn_runs"
mkdir -p "$REPO_ROOT/raw_data"

# 3. Create or Update .gitignore
cat <<EOF > "$REPO_ROOT/.gitignore"
ignore/
mike/
KMC/
doc/
fig/
nn_*/
cnn_*/
proj1128/
proj3389/
data_prod/
phagehost_KU/
ncbi_dataset/
ncbi_dataset.zip
ncbi_phage_genomes/
scripts_ext/mike/
.Rproj.user
raw_data/
logs/
EOF

echo "✅ Directories 'data_prod/' and 'tmp/' are ready."
echo "----------------------------------------------"