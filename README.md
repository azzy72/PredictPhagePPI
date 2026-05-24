# PredictPhagePPI

Predicting phage–host interactions from downsampled bacterial and bacteriophage genomes using machine learning.

Given a phage genome and a bacterial genome, the model predicts whether the phage can infect that bacterium. Genome representation is achieved by downsampling each genome into a fixed-size MinHash sketch (k-mer set), either via [Sourmash](https://sourmash.readthedocs.io/) or a custom MurmurHash3 / xxHash implementation. The concatenated sketches form the input vector to a Feed-Forward Neural Network (FFNN).

---

## Project overview

```
PredictPhagePPI/
├── raw_data/               # Raw FASTA genomes + interaction tables
├── data_prod/              # Produced data: sketches, presence matrices, results
├── nn_runs/                # Per-run output from FFNN training
├── tmp/                    # Temporary Slurm task files
├── notebooks/              # Exploratory Jupyter notebooks
├── scripts/                # Core pipeline scripts
│   └── slurm_iec/          # Iterative cluster-exclusion Slurm pipeline
├── Makefile                # Environment setup helpers
└── README.md
```

---

## Data

### Raw data (`raw_data/`)

Two datasets are supported, referred to throughout the code as **data1** (default) and **data2**.

| Path | Description |
|------|-------------|
| `raw_data/phagehost_KU/bacteriaKU_cleaned.fasta` | Bacterial genomes (data1) |
| `raw_data/phagehost_KU/phage_cleaned.fasta` | Phage genomes (data1) |
| `raw_data/phagehost_KU/Hostrange_data_all_crisp_iso.xlsx` | Lab-tested phage–host interactions (EOP / binary) for data1 |
| `raw_data/phagehost_KU/data2_bacts.fasta` | Bacterial genomes (data2) |
| `raw_data/phagehost_KU/data2_phages.fasta` | Phage genomes (data2) |
| `raw_data/phagehost_KU/data2_EOP.xlsx` | EOP interaction table for data2 |

### Produced data (`data_prod/`)

| Path | Description |
|------|-------------|
| `SM_sketches/` | Sourmash/MurmurHash3 MinHash sketches (`BactMinhash_n<N>_k<K>/`, `PhageMinhash_n<N>_k<K>/`) |
| `SM_sketches/sim_matrices/` | Pairwise Jaccard similarity matrices + cluster assignment CSVs |
| `PresMat_*/` | Precomputed binary presence matrices used as NN input |
| `IterExclClus_*/` | Collected results from iterative cluster-exclusion runs |
| `FFNN_Test_PFI2_collected/` | Collected results with pairwise feature importance (PFI) analysis |
| `NN_files/` | Train/test split definitions and saved model files |

---

## Pipeline

### Step 1 — Genome downsampling

Converts raw FASTA files into MinHash sketches. Each genome is decomposed into k-mers of length *k*, and the *n* smallest hashes are retained.

```bash
# Sourmash-backed sketches (default)
python scripts/downsampling.py --nk 500 12 --method sourmash

# Custom MurmurHash3 sketches
python scripts/downsampling.py --nk 500 12 --method minhash --hash mmh3

# data2 dataset
python scripts/downsampling.py --nk 500 12 --method sourmash --data2

# Batch downsampling across many n/k combinations (via Slurm)
bash scripts/queue_batch_downsampling.sh
```

For bacterial genomes with multi-contig assemblies, use `bact_downsampling.py` (wraps `downsampling.py` with per-contig merging).

Key parameters:

| Flag | Description |
|------|-------------|
| `--nk N K` | Sketch size *N* and k-mer length *K* (applied to both bacteria and phages) |
| `--split_nk BN BK PN PK` | Independent sketch parameters for bacteria and phages |
| `--method` | `sourmash` (default), `minhash`, or `ohe` (one-hot encoding) |
| `--hash` | Hash function: `mmh3` (default), `xxhash`, or `ohe_custom` |
| `--data2` | Use the data2 dataset |

Output lands in `data_prod/SM_sketches/` (or `SM_sketches_data2/`).

### Step 2 — Cluster analysis & data composition

Computes pairwise Jaccard similarity between sketches and clusters genomes for use in the iterative-exclusion pipeline.

```bash
# Run full Sourmash compare + cluster pipeline
python scripts/compare_sigs.py --config scripts/config_compare_sigs.yaml

# Dry-run to preview commands
python scripts/compare_sigs.py --config scripts/config_compare_sigs.yaml --dry-run
```

Outputs similarity matrices, heatmap PNGs, and cluster assignment CSVs to `data_prod/SM_sketches/sim_matrices/`.

### Step 3 — FFNN training (`FFNN_inner.py`)

The core model script. Builds a presence matrix from the MinHash sketches, trains an FFNN, and optionally runs feature importance analysis.

```bash
# Basic run (n=500, k=12, Sourmash sketches)
python scripts/FFNN_inner.py --nk 500 12 --logging

# Cross-validation with SMOTE, saving the model
python scripts/FFNN_inner.py --nk 500 12 --cv --kf_n_splits 5 --smote --save_model --logging

# Exclude a bacterial cluster and test on unseen data
python scripts/FFNN_inner.py --nk 500 12 \
    --exclude_clusters \
    --exclude_bact_clusters StrainA StrainB \
    --test_on_excluded --logging

# Pairwise feature importance + gene annotation
python scripts/FFNN_inner.py --nk 500 12 --perform_pfi --perform_ga --logging

# data2 with EOP values
python scripts/FFNN_inner.py --nk 500 12 --data2 --logging
```

**Selected flags:**

| Flag | Description |
|------|-------------|
| `--nk N K` | Sketch parameters (n hashes, k-mer size) |
| `--data2` | Use the data2 EOP dataset |
| `--use_encoded` | Use custom-encoded sketches instead of Sourmash sketches |
| `--cv` / `--kf_n_splits` | K-Fold cross-validation |
| `--smote` | SMOTE class-imbalance oversampling |
| `--train_by_cluster` | Group data splits by bacterial cluster |
| `--exclude_clusters` | Hold out entire clusters for out-of-distribution testing |
| `--test_on_excluded` | Evaluate the held-out cluster as the test set |
| `--test_on_unseen` | Evaluate on completely unseen (zero overlap) data |
| `--perform_pfi` | Pairwise feature importance analysis |
| `--perform_ga` | Gene annotation of top-ranked k-mers |
| `--save_model` | Persist trained model to disk |
| `--n_epochs` / `--learning_rate` / `--batch_size` | Hyperparameters |

Output goes to `nn_runs/<run_dir>/`.

### Step 4 — Iterative cluster-exclusion pipeline (`slurm_iec/`)

Runs the full leave-one-cluster-out experiment: every (bacteria-cluster, phage-cluster) pair is excluded in turn, the model is trained on the remainder, and performance on the held-out pair is recorded. Results are aggregated by `collect_iterres.py`.

```bash
# Submit the full array + postprocess job to Slurm
bash scripts/slurm_iec/submit_iter_excl.sh 500 12 SM_sketches

# Local (non-Slurm) run for testing
bash scripts/slurm_iec/submit_iter_local.sh 500 12 SM_sketches
```

`submit_iter_excl.sh` generates a task map, submits `ffnn_train_task.sh` as a Slurm array, then chains `ffnn_postprocess.sh` with `--dependency=afterok` to run `collect_iterres.py` once all tasks finish.

### Step 5 — Result collection & analysis (`collect_iterres.py`)

Aggregates per-run results from an iterative-exclusion experiment into summary CSVs and figures.

```bash
python scripts/collect_iterres.py \
    --run_dir nn_runs/IterExcl_SM_sketches_n500_k12 \
    --out data_prod/IterExclClus_SM_sketches_n500_k12
```

Produces: `all_runs_summary.csv`, accuracy/F1/balanced-accuracy plots by n/k, confusion matrices, k-mer distribution plots, top gene annotations, and a full run log.

---

## Environment setup

```bash
# Create conda environment and install dependencies
make setup

# Or step by step:
make env          # create conda env 'PredPPI' (Python 3.11)
make requirements # scan and freeze requirements.txt

conda activate PredPPI
pip install -e .  # install package in editable mode (run from repo root)
```

---

## Notebooks (`notebooks/`)

Exploratory Jupyter notebooks for each stage of the project:

| Notebook | Description |
|----------|-------------|
| `downsampling.ipynb` | MinHash sketch exploration |
| `bact_eda.ipynb` / `phage_eda.ipynb` | Genome-level EDA for bacteria and phages |
| `phylogeny_and_minhash.ipynb` | Phylogenetic structure vs. sketch similarity |
| `preprocces_data.ipynb` | Interaction matrix preprocessing |
| `random_forest.ipynb` | Random Forest baseline |
| `k_nearest_neighbor.ipynb` | k-NN baseline |
| `nn_torch.ipynb` | Early FFNN prototyping |
| `FFNN_outer_dev.ipynb` | Outer-loop development (model selection) |
| `gene_investigation.ipynb` | Gene-level feature analysis |
| `cnn.ipynb` / `rnn.ipynb` | CNN and RNN experiments |
| `playground.ipynb` | Scratch space |

---

## Key scripts reference

| Script | Role |
|--------|------|
| `downsampling.py` | Genome → MinHash sketch (Sourmash / mmh3 / xxhash) |
| `bact_downsampling.py` | Bacterial-specific downsampling (multi-contig support) |
| `batch_downsampling.py` | Batch over multiple n/k configurations |
| `compare_sigs.py` | Pairwise Jaccard similarity + clustering |
| `FFNN_inner.py` | **Core FFNN training & evaluation** |
| `collect_iterres.py` | **Aggregate iterative-exclusion results** |
| `decompositions.py` | k-mer decomposition, `KmerCodec` (4-bit encoding), `Decompose` class |
| `io_operations.py` | Presence matrix I/O, host-range loading |
| `manipulations.py` | Feature construction, host-range binarisation |
| `analysis.py` | Feature importance, gene analysis, plotting |
| `data_generators.py` | PyTorch dataset utilities |
| `nn_torch.py` / `torchmlp.py` / `simpleffnn.py` | Network architecture definitions |
| `paths.py` | Centralised path constants |
| `utils.py` | Taxonomy / strain ID helpers |
| `slurm_iec/submit_iter_excl.sh` | Launch full iterative-exclusion array |
| `slurm_iec/ffnn_train_task.sh` | Single Slurm array task (one cluster pair) |
| `slurm_iec/ffnn_postprocess.sh` | Post-processing job (runs after array completes) |
| `slurm_iec/extract_accuracy.sh` | Extract per-run accuracy files |

---

## Dependencies

Core libraries: `torch`, `scikit-learn`, `imbalanced-learn`, `sourmash`, `mmh3`, `xxhash`, `biopython`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `networkx`, `tqdm`, `pyyaml`, `joblib`.

See `requirements.txt` (auto-generated by `make requirements`) for the full pinned list.

---

## Notes

- KU library preparation uses the Hackflex protocol: https://github.com/GaioTransposon/Hackflex
- Key protein classes investigated in gene annotation: depolymerases ([ref](https://link.springer.com/article/10.1186/s12929-023-00928-0)), anti-defence systems (CRISPR-related), modifying enzymes, integrases.
