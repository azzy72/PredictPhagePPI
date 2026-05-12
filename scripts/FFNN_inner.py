#!/usr/bin/python3

import os
import sys
import re
import argparse
import random
import logging
import pickle
import traceback
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from time import time, sleep
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split, GroupShuffleSplit
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, balanced_accuracy_score
from scipy.special import expit
from imblearn.over_sampling import SMOTE

# Custom imports
from io_operations import presence_matrix, obtain_idx_to_entity_mapping, call_hostrange_df, color_sheet_from_matrix
from paths import raw_data_path, data_prod_path, path_to_nn_runs
from manipulations import calc_PFI, hostrange_df_to_dict, binarize_host_range, clean_bact_names
from analysis import f1_analysis, plot_entity_counts, plot_bipartite_network, regain_kmers, plot_interaction_pairs, FeatureImportance, GeneAnalysis, GeneAnalysisNCBI
from utils import strain_id_tax_lookup
import json

def parse_arguments():
    parser = argparse.ArgumentParser(description="FFNN Training Script")

    # Parameters: Mutual exclusivity for n/k vs specific bn/bk/pn/pk
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--nk", nargs=2, type=int, metavar=('N', 'K'), default=(500, 12),
                        help="Unified n and k values (e.g., -nk 500 12)")
    group.add_argument("--split_nk", nargs=4, type=int, metavar=('BN', 'BK', 'PN', 'PK'),
                        help="Split values for Bact (n, k) and Phage (n, k)")

    # Data Source
    parser.add_argument("--use_encoded", action="store_true", help="Use encoded_sketches instead of SM_sketches")
    parser.add_argument("--data2", action="store_true", help="Use the second dataset with EOP values instead of binary interactions")
    parser.add_argument("--bits_encoded", type=str, default="4", help="(Optional) specify which type of bit encoding using in encoded_sketches (e.g. 4 for phage_encode4bit_n400_k12)")
    parser.add_argument("--out", type=str, help="custom directory to write to in nn_runs/")
    parser.add_argument("--sbatch_id", type=str, help="(Optional) sbatch job ID to include in output directory name for easier tracking")

    # Flags
    parser.add_argument("--logging", action="store_true", help="Enable logging and saving")
    parser.add_argument("--cv", action="store_true", help="Enable cross-validation")
    parser.add_argument("--kf_n_splits", type=int, default=4, help="Number of folds for K-Fold CV")
    parser.add_argument("--smote", action="store_true", help="Apply SMOTE oversampling")
    parser.add_argument("--train_by_cluster", action="store_true", help="Split data by bacterial clusters")
    parser.add_argument("--no_randomize", action="store_false", dest="randomize", help="Disable entity randomization")
    parser.add_argument("--no_shuffle", action="store_false", dest="shuffle", help="Disable feature shuffling")
    parser.add_argument("--entity_order", choices=["bact_first", "phage_first"], default="bact_first", help="Choose order of input vector; bact first then phage is the default.")
    parser.add_argument("--perform_fi", action="store_true", help="Perform feature importance analysis")
    parser.add_argument("--perform_pfi", action="store_true", help="Perform pairwise feature importance analysis")
    parser.add_argument("--perform_ga", action="store_true", help="Perform gene analysis on top features")
    parser.add_argument("--run_ga_on_pfi", action="store_true", help="Run gene analysis on top features from pairwise feature importance analysis, rather than the standard feature importance analysis")
    parser.add_argument("--reconstruct_gene_annotation", action="store_true", help="Reconstruct gene annotations from GeneAnalysisNCBI, rather than loading them from a file from a previous run.")
    parser.add_argument("--no_val", action="store_false", dest="use_val", help="Disable validation set in favor of larger training set (not recommended, but can be used for final training after hyperparameter tuning)")
    parser.add_argument("--save_model", action="store_true", help="Save the trained model to the output directory for future use")
    parser.add_argument("--subset_pfi", type=int, help="Number of top features to include in pairwise feature importance analysis (set to 0 or a negative number to include all features)")
    parser.add_argument("--force_pfi_recalculation", action="store_true", help="Force recalculation of pairwise feature importance even if the output file already exists. This can be useful if you have changed the subset_pfi parameter or made changes to the code that calculates feature importance and want to ensure that the pairwise feature importance is recalculated with the new logic.")

    # Exclusions
    parser.add_argument("--exclude_noninteractions", action="store_true", help="Exclude non-interacting pairs")
    parser.add_argument("--exclude_pairs", action="store_true", help="Exclude specified pairs of bacteria and phages, requires --exclude_bacts and --exclude_phages")
    parser.add_argument("--exclude_bacts", nargs='+', default=["J26_21_reoriented"], help="List of bacteria to exclude")
    parser.add_argument("--exclude_phages", nargs='+', default=["Abuela"], help="List of phages to exclude")
    
    parser.add_argument("--exclude_clusters", action="store_true", help="Exclude all pairs involving bacteria in the specified clusters, requires --exclude_bact_clusters and --exclude_phage_clusters")
    parser.add_argument("--exclude_bact_clusters", nargs='+', default=[], help="Array of bacterial strains to exclude in a cluster like manner")
    parser.add_argument("--exclude_phage_clusters", nargs='+', default=[], help="Array of phage strains to exclude in a cluster like manner")
    parser.add_argument("--cluster_by_genus", action="store_true", help="Cluster by genus instead of a pre-defined cluster file. This is a more extreme exclusion strategy that may be useful to test the model's ability to generalize to completely unseen genera.")
    
    parser.add_argument("--test_on_excluded", action="store_true", help="Test the model on the excluded pairs/clusters and not a test split from the main dataset")


    # Hyperparameters
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--val_split", type=float, default=0.2)

    parser.add_argument("--top_kmers_num", type=int, default=50, help="Number of top k-mers to retrieve from feature importance (pfi)")

    args = parser.parse_args()

    # --- VALIDATION LOGIC FOR ARGUMENTS ---
    # Requirement: kf_n_splits must be > 1 if --cv is on
    if args.cv and args.kf_n_splits <= 1:
        parser.error("--kf_n_splits must be greater than 1 when --cv is enabled.")

    # Requirement: exclude_pairs cannot be used with exclude_clusters
    if args.exclude_pairs and args.exclude_clusters:
        parser.error("--exclude_pairs cannot be used with --exclude_clusters because they represent two different exclusion strategies that would conflict with each other.")

    # Requirement: exclude_clusters requires exclude_bact_clusters and exclude_phage_clusters
    if args.exclude_clusters and (len(args.exclude_bact_clusters) < 1 or len(args.exclude_phage_clusters) < 1):
        parser.error("--exclude_clusters requires both --exclude_bact_clusters and --exclude_phage_clusters lists.")

    # Requirement: exclude_noninteractions requires exclude_bacts and exclude_phages
    if args.exclude_pairs and (len(args.exclude_bacts) < 1 or len(args.exclude_phages) < 1):
        parser.error("--exclude_pairs requires both --exclude_bacts and --exclude_phages lists.")

    # Requirement: test_on_excluded requires exclude_noninteractions
    if args.test_on_excluded and not any([args.exclude_pairs, args.exclude_clusters]):
        parser.error("--test_on_excluded requires at least one: --exclude_pairs or --exclude_clusters")
    
    # Warning: exclude_noninteractions with exclude_pairs means that both the entities in --exclude_bacts and --exclude_phages will be excluded from the training set, as well as all entities that doesn't have a positive interaction in hostrange 
    # This may result in a very small training set if those entities are involved in many interactions. This is not an error, but should be used with caution.
    if args.exclude_pairs and args.exclude_noninteractions:
        print("WARNING: Using --exclude_pairs with --exclude_noninteractions will exclude all pairs involving the specified bacteria and phages, as well as all non-interacting pairs. This may result in a very small training set if those entities are involved in many interactions. Please use with caution.", file=sys.stderr)

    # Requirement: can't have perform_ga with not args.use_encoded
    if args.perform_ga and not args.use_encoded:
        parser.error("--perform_ga requires --use_encoded as kmers can't be decoded from MinHash sketches.")
    
    # Warning: Using --no_val will ignore val_split
    if not args.use_val and args.val_split:
        print("WARNING: Using --no_val will ignore val_split.", file=sys.stderr)
    
    # Warning: Using --test_on_excluded will ignore test_split
    if args.test_on_excluded and args.test_split:
        print("WARNING: Using --test_on_excluded will ignore test_split.", file=sys.stderr)
    
    # Requirement: if --no_val is used, then --cv cannot be used because cross-validation requires a validation set for epoch-wise evaluation
    if not args.use_val and args.cv:
        parser.error("--no_val cannot be used with --cv because cross-validation requires a validation set for epoch-wise evaluation.") 

    # Requirement: ignore --force_pfi_recalculation if --perform_pfi is not set, since it doesn't make sense to force recalculation of pairwise feature importance if we're not performing feature importance analysis at all
    if args.force_pfi_recalculation and not args.perform_pfi:
        print("WARNING: --force_pfi_recalculation will be ignored because --perform_pfi is not set. It doesn't make sense to force recalculation of pairwise feature importance if we're not performing feature importance analysis at all.", file=sys.stderr)
        args.force_pfi_recalculation = False

    # Requirement: --perform_pfi should be True if --run_ga_on_pfi is True
    if args.run_ga_on_pfi and not args.perform_pfi:
        print("WARNING: --run_ga_on_pfi will be ignored because --perform_pfi is not set. It doesn't make sense to run gene analysis on pairwise feature importance if we're not performing feature importance analysis at all.", file=sys.stderr)
        args.run_ga_on_pfi = False

    # # Modification: automatically set test_on_excluded to True if exclude_pairs or exclude_clusters is used, since it doesn't make sense to have a test split from the main dataset if the excluded pairs/clusters are not in the test set
    # if (args.exclude_pairs or args.exclude_clusters) and not args.test_on_excluded:
    #     args.test_on_excluded = True
    #     print("INFO: --test_on_excluded has been automatically set to True because --exclude_pairs or --exclude_clusters is used. This means the model will be tested on the excluded pairs/clusters and not a test split from the main dataset.", file=sys.stderr)

    # Standardize args.exlude clusters arguments
    if args.exclude_clusters:
        # Modification: convert args.exclude_bact_clusters and args.exclude_phage_clusters from str to list if they are provided as comma-separated strings 
        # (this allows for more flexible input, e.g. --exclude_bact_clusters ["cluster1,cluster2,cluster3"] or --exclude_bact_clusters cluster1 cluster2 cluster3)
        if isinstance(args.exclude_bact_clusters, str):
            args.exclude_bact_clusters = re.sub(r'[\[\]]', '',args.exclude_bact_clusters)
            args.exclude_bact_clusters = [item.strip() for item in args.exclude_bact_clusters.split(',')]
        elif isinstance(args.exclude_bact_clusters, list) and len(args.exclude_bact_clusters) == 1 and ',' in args.exclude_bact_clusters[0]:
            args.exclude_bact_clusters = re.sub(r'[\[\]]', '',args.exclude_bact_clusters[0])
            args.exclude_bact_clusters = [item.strip() for item in args.exclude_bact_clusters.split(',')]
        
        if isinstance(args.exclude_phage_clusters, str):
            args.exclude_phage_clusters = re.sub(r'[\[\]]', '',args.exclude_phage_clusters)
            args.exclude_phage_clusters = [item.strip() for item in args.exclude_phage_clusters.split(',')]
        elif isinstance(args.exclude_phage_clusters, list) and len(args.exclude_phage_clusters) == 1 and ',' in args.exclude_phage_clusters[0]:
            args.exclude_phage_clusters = re.sub(r'[\[\]]', '',args.exclude_phage_clusters[0])
            args.exclude_phage_clusters = [item.strip() for item in args.exclude_phage_clusters.split(',')]

        # Make args.exclude_bact_clusters short names (e.g. J2_21) - phages stays the same
        args.exclude_bact_clusters = clean_bact_names(args.exclude_bact_clusters, data2=args.data2)

    return args

# This function maps model feature index 'idx' back to the encoded k-mer
class MLP(nn.Module):
    def __init__(self, input_dim, hidden1=256, hidden2=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden2, 1)
        )
    def forward(self, x):
        return self.net(x)


class helper:
    def __init__(self, parser_args=None):
        self.args = parser_args
        pass

    def model_idx_to_kmer(self, idx, num_features_per_entity, feature_indices, idx_to_minhash):
        # Determine if it's a Bact feature (idx < N) or Phage feature (idx >= N)
        original_col_idx = feature_indices[idx % num_features_per_entity]
        return idx_to_minhash[original_col_idx]
    
def main():
    args = parse_arguments()
    time_start = time()
    h = helper(args)
    ncbi_blast_res_df = None

    ### 1. Resolve N/K values ###
    if args.nk:
        n = bn = pn = args.nk[0]
        k = bk = pk = args.nk[1]
        presmat_suffix = f"n{n}_k{k}"
    else:
        bn, bk, pn, pk = args.split_nk
        n, k = bn, bk # Reference n/k for folder naming
        presmat_suffix = f"bn{bn}_bk{bk}_pn{pn}_pk{pk}"

    ### 2. Path Logic ###
    sourmash_used = not args.use_encoded
    prefix = "encoded_sketches" if args.use_encoded else "SM_sketches"
    if args.data2:
        prefix = f"{prefix}_data2"

    #use regex to find directories with n{bn}_k{bk} and n{pn}_k{pk} in their names, since dir prefix depends on method
    files_prefix_dirs = os.listdir(os.path.join(data_prod_path, prefix))
    
    phage_dir_pattern = re.compile(f".*n{pn}_k{pk}.*")
    input_phage_path = [file for file in files_prefix_dirs if phage_dir_pattern.match(file) and "phage" in file.lower()]
    if len(input_phage_path) == 0:
        raise FileNotFoundError(f"No directory found for phage minhash data with n={pn} and k={pk} in {os.path.join(data_prod_path, prefix)}")
    elif len(input_phage_path) > 1:
        raise ValueError(f"Multiple directories found for phage minhash data with n={pn} and k={pk} in {os.path.join(data_prod_path, prefix)}: {input_phage_path}")
    input_phage_path = f"{prefix}/{input_phage_path[0]}/"

    bact_dir_pattern = re.compile(f".*n{bn}_k{bk}.*")
    input_bact_path = [file for file in files_prefix_dirs if bact_dir_pattern.match(file) and "bact" in file.lower()]
    if len(input_bact_path) == 0:
        raise FileNotFoundError(f"No directory found for bacteria minhash data with n={bn} and k={bk} in {os.path.join(data_prod_path, prefix)}")
    elif len(input_bact_path) > 1:
        raise ValueError(f"Multiple directories found for bacteria minhash data with n={bn} and k={bk} in {os.path.join(data_prod_path, prefix)}: {input_bact_path}")
    input_bact_path = f"{prefix}/{input_bact_path[0]}/"

    presmat_path = f"{prefix}/PresMat_{presmat_suffix}/"
    print(f"Recognized data paths\ninput_phage_path:\t{input_phage_path}\ninput_bact_path:\t{input_bact_path}\npresmat_path:\t{presmat_path}")

    ### 3. Load Data ###
    bact_clusters = pd.read_csv(os.path.join(data_prod_path, "bact_clusters_with_genus.csv"), index_col=0)

    # Load Presence Matrix
    full_presmat_path = os.path.join(data_prod_path, presmat_path)
    if not os.path.exists(full_presmat_path):
        print("Reconstructing presence_matrix...")
        binary_matrix, entity_to_index, minhash_to_index, phage_minhash_data, bact_minhash_data = presence_matrix(
            phage_minhash_dir=os.path.join(data_prod_path, input_phage_path),
            bact_minhash_dir=os.path.join(data_prod_path, input_bact_path),
            k=[bk, pk], n=[bn, pn], reversecomp_data=False, TS=True, data2=args.data2)
        #Write files to full_presmat_path for faster loading in the future
        os.makedirs(full_presmat_path, exist_ok=True)
        with open(os.path.join(full_presmat_path, "binary_matrix.pkl"), "wb") as f: pickle.dump(binary_matrix, f)
        with open(os.path.join(full_presmat_path, "entity_to_index.pkl"), "wb") as f: pickle.dump(entity_to_index, f)
        with open(os.path.join(full_presmat_path, "phage_minhash_data.pkl"), "wb") as f: pickle.dump(phage_minhash_data, f)
        with open(os.path.join(full_presmat_path, "bact_minhash_data.pkl"), "wb") as f: pickle.dump(bact_minhash_data, f)
        with open(os.path.join(full_presmat_path, "minhash_to_index.pkl"), "wb") as f: pickle.dump(minhash_to_index, f)

    else:
        print("Loading presence matrix from pre-saved files...")
        with open(os.path.join(full_presmat_path, "binary_matrix.pkl"), "rb") as f: binary_matrix = pickle.load(f)
        with open(os.path.join(full_presmat_path, "entity_to_index.pkl"), "rb") as f: entity_to_index = pickle.load(f)
        with open(os.path.join(full_presmat_path, "phage_minhash_data.pkl"), "rb") as f: phage_minhash_data = pickle.load(f)
        with open(os.path.join(full_presmat_path, "bact_minhash_data.pkl"), "rb") as f: bact_minhash_data = pickle.load(f)
        with open(os.path.join(full_presmat_path, "minhash_to_index.pkl"), "rb") as f: minhash_to_index = pickle.load(f) 

    # Create an idx_to_entity to lookup the origin of index (phage or bacteria, and which one)
    idx_to_entity = obtain_idx_to_entity_mapping(
        phage_minhash_data=phage_minhash_data,
        bact_minhash_data=bact_minhash_data,
        minhash_to_index=minhash_to_index
    )

    #Create inverse mapping: entity_name to column_index
    entity_to_idx = {v: k for k, v in idx_to_entity.items()}

    #Load hash kmer lookup dict, if use_encoded and current n/k values
    if args.use_encoded:
        hash_kmer_dict_path = os.path.join(data_prod_path, f"{prefix}/hk_lookup_n{n}_k{k}.json")
        if os.path.exists(hash_kmer_dict_path):
            with open(hash_kmer_dict_path, "r") as f:
                hk_translation_dict = json.load(f)
                try:
                    hk_translation_dict = {int(k): v for k, v in hk_translation_dict.items()} # convert keys back to int after loading from json
                except Exception as e:
                    raise ValueError(f"Error converting hk_translation_dict keys to int: {e}")

        else:
            print(f"Hash k-mer lookup dictionary not found at {hash_kmer_dict_path}. Please ensure the file exists or run the script to generate it.")
            hk_translation_dict = None

    ### 4. Host Range Setup ###
    bact_lookup, host_range_df = call_hostrange_df(os.path.join(raw_data_path, "phagehost_KU/Hostrange_data_all_crisp_iso.xlsx"), data2=args.data2)
    host_range_data = binarize_host_range(hostrange_df_to_dict(host_range_df), continous=False)
    host_range_data = {bact.replace("_reoriented", ""): interactions for bact, interactions in host_range_data.items()} # if "_reoriented" is in the bacteria names in host_range_data, remove it to match the bacteria names in the presence matrix.

    ### 5. Logging Setup ###
    outdir, logfile = None, None
    if args.logging:
        run = 1
        tag = "smote" if args.smote else "standard"
        if not args.out:
            if args.sbatch_id:
                outdirname = f"{args.sbatch_id}_torch_mlp_n{n}_k{k}_{tag}"
            else:
                outdirname = f'torch_mlp_n{n}_k{k}_{tag}'
            outdir = os.path.join(path_to_nn_runs, f"{outdirname}_run{run}/")
            while os.path.exists(outdir):
                run += 1
                outdir = os.path.join(path_to_nn_runs, f"{outdirname}_run{run}/")
        else:
            outdir = os.path.join(path_to_nn_runs, f"{args.out}_run{run}/")
            while os.path.exists(outdir):
                run += 1
                outdir = os.path.join(path_to_nn_runs, f"{args.out}_run{run}/")
        os.makedirs(outdir, exist_ok=True)
        # Configure the logger
        logging.basicConfig(
            level=logging.INFO,
            filename=os.path.join(outdir, f'log_run{run}.txt'),
            filemode='w', # 'a' for append, 'w' for overwrite
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logging.info(f'Run started: {datetime.now()}\n')
        logging.info('Params:')
        for key, value in vars(args).items():
            logging.info(f'  {key}: {value}')
        logging.info('##################')
        print("Logging enabled. Output directory created at:", outdir)
    
    # Writing host range data logs
    if args.logging:
        sample_bact = next(iter(host_range_data))
        logging.info(f'Host range data loaded with {len(host_range_data)} bacteria.')
        logging.info(f'Sample host range entry for {sample_bact}: {host_range_data[sample_bact]}')
        if args.exclude_clusters:
            logging.info(f'Exclusion criteria: Clusters')
            logging.info(f'- Bacterial clusters to exclude: {args.exclude_bact_clusters}')
            logging.info(f'- Phage clusters to exclude: {args.exclude_phage_clusters}')
    
    ### 6. Feature Preparation ###
    X, y, X_excl, y_excl, rows_meta = [], [], [], [], []
    phage_names = list(phage_minhash_data.keys())
    bacteria_names = list(bact_minhash_data.keys())

    # Correcting exclude_bacts and exclude_phages if they're not in the same format as hostrange
    if args.exclude_clusters:
        args.exclude_bacts = clean_bact_names(args.exclude_bacts)

    if args.randomize:
        random.seed(42)
        random.shuffle(bacteria_names)
        random.shuffle(phage_names)

    if args.shuffle:
        feature_indices = list(range(binary_matrix.shape[1]))
        random.seed(42)
        random.shuffle(feature_indices)
        binary_matrix = binary_matrix[:, feature_indices]
    else:
        feature_indices = list(range(binary_matrix.shape[1]))
    
    # Save phage_names and bacteria_names to output directory for reference to order of appearance; randomization has taken place above, this order needs to be conserved in subsequent analysis.
    if args.logging:
        with open(os.path.join(outdir, "bacteria_names.txt"), "w") as f:
            for bact in bacteria_names:
                f.write(f"{bact}\n")
        with open(os.path.join(outdir, "phage_names.txt"), "w") as f:
            for phage in phage_names:
                f.write(f"{phage}\n")

    # Create inverse mapping: column_index -> kmer_encoded_int
    idx_to_minhash = {v: k for k, v in minhash_to_index.items()}

    # Convert exclusion lists to sets for O(1) membership testing inside the hot nested loop
    exclude_bacts_set = set(args.exclude_bacts) if args.exclude_pairs else set()
    exclude_phages_set = set(args.exclude_phages) if args.exclude_pairs else set()
    exclude_bact_clusters_set = set(args.exclude_bact_clusters) if args.exclude_clusters else set()
    exclude_phage_clusters_set = set(args.exclude_phage_clusters) if args.exclude_clusters else set()

    # Pre-resolve entity order to avoid repeated string comparison in the inner loop
    bact_first = (args.entity_order == "bact_first")

    if args.logging: feature_flag = False
    cidx = 0 #counter for idx of all features
    eidx = 0 #counter for idx of excluded features
    X_idx = []
    X_excl_idx = []
    X_excl_true_unseen_idx = [] #idx of the truly unseen pairs on excluded sets runs.
    #exclude_bact_characters = ["_reoriented", "_merged", "_KMC"]
    for bact in tqdm(bacteria_names, desc="Building dataset"):
        # Exclusion logic
        if args.exclude_noninteractions and not any(host_range_data.get(bact, {}).values()):
            continue
            
        for phage in phage_names:
            # if args.logging:
            #     print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Processing pair ({bact}, {phage})...', file=logfile)
            if phage not in host_range_data.get(bact, {}): 
                if args.logging:
                    logging.warning(f'No interaction data for pair ({bact}, {phage}). This pair will be skipped.')
                continue
            
            # Get features and metadata for this pair
            score = host_range_data[bact][phage]
            if bact_first:
                features = np.concatenate((binary_matrix[entity_to_index[bact]], 
                                       binary_matrix[entity_to_index[phage]]))
                rows_meta.append((bact, phage))
            else:
                features = np.concatenate((binary_matrix[entity_to_index[phage]], 
                                       binary_matrix[entity_to_index[bact]]))
                rows_meta.append((phage, bact))

            # Logging
            if args.logging and not feature_flag:
                logging.info(f'Sample feature vector for pair ({bact}, {phage}) with score: {score}')
                logging.info(f'- Number of features: {len(features.tolist())}')
                feature_flag = True

            # Decide what to do with this pair based on exclusion criteria
            if args.exclude_pairs and (bact in exclude_bacts_set or phage in exclude_phages_set):
                X_excl.append(features)
                y_excl.append(score)    
                X_excl_idx.append(cidx)
                if bact in exclude_bacts_set and phage in exclude_phages_set:
                    X_excl_true_unseen_idx.append(eidx)
                cidx += 1
                eidx += 1
                continue

            if args.exclude_clusters:
                if bact in exclude_bact_clusters_set or phage in exclude_phage_clusters_set:
                    X_excl.append(features)
                    y_excl.append(score)    
                    X_excl_idx.append(cidx)
                    if args.logging:
                        logging.info(f'Pair ({bact}, {phage}) added to exclusion set based on cluster criteria.')
                    if bact in exclude_bact_clusters_set and phage in exclude_phage_clusters_set:
                        X_excl_true_unseen_idx.append(eidx)
                        if args.logging:
                            logging.info(f'Pair ({bact}, {phage}) is truly unseen in test set because both entities are in the exclusion lists.')
                    eidx += 1
                    cidx += 1
                    continue
                # else:
                #     if args.logging:
                #         print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Pair ({bact}, {phage}) passed cluster exclusion criteria and will be included in main dataset.', file=logfile)
                #         print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} - {bact} in exclude_bact_clusters: {bact in args.exclude_bact_clusters}', file=logfile)
                #         print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} - {phage} in exclude_phage_clusters: {phage in args.exclude_phage_clusters}', file=logfile)
                #         print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} - bact dtype: {type(bact)}, exclude_bact_cluster inner dtype: {type(args.exclude_bact_clusters[0])}', file=logfile)
                #         print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} - phage dtype: {type(phage)}, exclude_phage_cluster inner dtype: {type(args.exclude_phage_clusters[0])}', file=logfile)
        
            # If it passes exclusion criteria, add to main dataset
            X.append(features)
            y.append(score)
            X_idx.append(cidx)
            cidx += 1

    X, y = np.array(X), np.array(y)
    X_excl, y_excl = np.array(X_excl), np.array(y_excl)
    if sum(y_excl) < 1:
        if args.logging:
            logging.error(f'No positive values in test\n{traceback.print_exc()}')
        print("No positive values in test")
        return
    
    if args.logging:
        logging.info(f'--- Finished building dataset ---')
        logging.info(f'Built dataset with {len(X)} pairs and excluded {len(X_excl)} pairs.')
        logging.info(f'There should be {len(X)} times {len(X[0])} total features for interacting pairs:')
        logging.info(f'{len(X)} x {len(X[0])} = {X.shape}')
        logging.info(f'cidx: {cidx}')
        if args.exclude_pairs or args.exclude_clusters:
            logging.info(f'Excluded {len(X_excl)} pairs based on --exclude_bacts and --exclude_phages lists.')
            logging.info(f'There should be {len(X_excl)} times {len(X_excl[0])} total non-unique features:')
            logging.info(f'{len(X_excl)} represent the excluded pairs.')
    ### 7. Splitting & Scaling ###
    train_idx, test_idx = X_idx, X_excl_idx
    if args.train_by_cluster:
        groups = bact_clusters.loc[[m[0] for m in rows_meta], 'Cluster'].values
        val_clusters = None
        test_clusters = None

        if args.test_on_excluded:
            X_train_f, y_train_f = X, y
            X_test, y_test = X_excl, y_excl
            X_test_unseen, y_test_unseen = X_excl[X_excl_true_unseen_idx], y_excl[X_excl_true_unseen_idx]

        else:
            # Split data into train and test
            gss = GroupShuffleSplit(n_splits=1, test_size=args.test_split, random_state=42)
            train_full_idx, test_idx = next(gss.split(X, y, groups=groups))

            X_train_f, X_test = X[train_full_idx], X[test_idx]
            y_train_f, y_test = y[train_full_idx], y[test_idx]
            test_clusters = groups[test_idx]
            groups = groups[train_full_idx] #Redefing groups in case dataset was split into test and train

        train_clusters = groups
        if not args.use_val:
            X_val, y_val = None, None
        elif not args.cv:
            # Split train into train and val - non-cross validation run requires a validation set for epoch-wise evaluation
            adj_val_ratio = args.val_split / (1 - args.test_split)
            gss_val = GroupShuffleSplit(n_splits=1, test_size=adj_val_ratio, random_state=42)
            train_idx, val_idx = next(gss_val.split(X_train_f, y_train_f, groups=groups))
            X_train_f, X_val = X_train_f[train_idx], X_train_f[val_idx]
            y_train_f, y_val = y_train_f[train_idx], y_train_f[val_idx]
            train_clusters = groups[train_idx]
            val_clusters = groups[val_idx]
        
        # Construct a bar graph on the distribution of bacterial clusters in train vs val vs test to confirm that the split is indeed by cluster and that the clusters are distributed in a way that makes sense (e.g. not all cluster 1 in train and all cluster 2 in test)
        split_cluster_counts = {
            "Train": pd.Series(train_clusters).value_counts()
        }
        if val_clusters is not None:
            split_cluster_counts["Val"] = pd.Series(val_clusters).value_counts()
        if test_clusters is not None:
            split_cluster_counts["Test"] = pd.Series(test_clusters).value_counts()

        cluster_dist_df = pd.DataFrame(split_cluster_counts).fillna(0).astype(int)
        if not cluster_dist_df.empty:
            try:
                cluster_dist_df = cluster_dist_df.sort_index(key=lambda x: pd.to_numeric(x, errors='coerce'))
            except Exception:
                cluster_dist_df = cluster_dist_df.sort_index()

            fig, ax = plt.subplots(figsize=(12, 6))
            cluster_dist_df.plot(kind='bar', ax=ax)
            ax.set_title('Bacterial Cluster Distribution Across Data Splits')
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Number of pairs')
            ax.legend(title='Split')
            plt.tight_layout()

            if args.logging:
                outname = 'cluster_distribution_train_val_test.png'
                plt.savefig(outdir+outname)
                logging.info(f'Cluster distribution figure saved as: {outdir+outname}')
            plt.close(fig)
        
    else:
        if args.test_on_excluded:
            X_train_f, y_train_f = X, y
            X_test, y_test = X_excl, y_excl
            X_test_unseen, y_test_unseen = X_excl[X_excl_true_unseen_idx], y_excl[X_excl_true_unseen_idx]

        else:
            train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=args.test_split, random_state=42, stratify=y)
            X_train_f, X_test, y_train_f, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
        
        # Split train into train and val
        if not args.cv:
            # Split train into train and val - non-cross validation run requires a validation set for epoch-wise evaluation
            train_idx, val_idx = train_test_split(train_idx, test_size=args.val_split/(1-args.test_split), random_state=42, stratify=y[train_idx])
            X_train_f, X_val = X[train_idx], X[val_idx]
            y_train_f, y_val = y[train_idx], y[val_idx]
        else: 
            if not args.use_val:
                X_val, y_val = None, None
    
    metadata_np = np.array(rows_meta, dtype=object)
    metadata_train_full, metadata_test = metadata_np[train_idx], metadata_np[test_idx]

    scaler = StandardScaler()
    X_train_f = scaler.fit_transform(X_train_f)
    X_test = scaler.transform(X_test)
    try:
        X_val = scaler.transform(X_val)
    except Exception as e:
        logging.warning(f"Error occurred while scaling validation data: {e}")

    if args.smote:
        sm = SMOTE(random_state=42)
        X_train_f, y_train_f = sm.fit_resample(X_train_f, y_train_f)

    ### 8. Training Logic ###
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training loop
    if args.cv:
        if args.logging: 
            logging.debug(f'Train + Val size: {X_train_f.shape[0]} samples, Test size: {X_test.shape[0]} samples')
            logging.debug(f'Fraction of positive interactions in train+val: {round(sum(y_train_f)/len(y_train_f)*100,2)}%')
            logging.debug(f'Fraction of positive interactions in test: {round(sum(y_test)/len(y_test)*100,2)}%')


        kf = KFold(n_splits=args.kf_n_splits, shuffle=True, random_state=42)
        fold = 1

        if args.logging: logging.info(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Starting cross-validation with {kf.get_n_splits()} folds...')

        for train_idx, val_idx in kf.split(X_train_f):
            print(f"Fold {fold}:")
            if args.logging: logging.debug(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Fold {fold}...')

            # Split data into training and validation sets for this fold
            X_train_fold, X_val_fold = X_train_f[train_idx], X_train_f[val_idx]
            y_train_fold, y_val_fold = y_train_f[train_idx], y_train_f[val_idx]

            # Convert to torch tensors
            X_train_t = torch.from_numpy(X_train_fold).float()
            X_val_t = torch.from_numpy(X_val_fold).float()
            y_train_t = torch.from_numpy(y_train_fold.reshape(-1, 1)).float()
            y_val_t = torch.from_numpy(y_val_fold.reshape(-1, 1)).float()

            # Create data loaders
            train_ds = TensorDataset(X_train_t, y_train_t)
            val_ds = TensorDataset(X_val_t, y_val_t)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

            # Initialize model, criterion, and optimizer
            model = MLP(input_dim=X_train_f.shape[1]).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

            # Training loop for this fold
            for epoch in range(1, args.n_epochs + 1):
                model.train()
                running_loss = 0.0
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * xb.size(0)
                epoch_loss = running_loss / len(train_loader.dataset)
                train_losses.append(epoch_loss)

                # Evaluate on validation set each epoch
                model.eval()
                with torch.no_grad():
                    total = 0
                    correct = 0
                    val_running_loss = 0.0
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        logits = model(xb)
                        loss = criterion(logits, yb)
                        val_running_loss += loss.item() * xb.size(0)
                        probs = torch.sigmoid(logits)
                        preds = (probs >= 0.5).float()
                        correct += (preds == yb).sum().item()
                        total += yb.numel()
                    val_loss = val_running_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else float('nan')
                    val_acc = correct / total if total > 0 else float('nan')
                    val_losses.append(val_loss)
                    val_accuracies.append(val_acc)

                print(f"Epoch {epoch:02d} - train_loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
                if args.logging: 
                    logging.debug(f'Epoch {epoch:02d} - train_loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}')

            fold += 1
        fold -= 1 # Adjust fold count after loop to reflect actual number of folds completed
    
    else:
        if args.logging: 
            logging.debug(f'Train size: {X_train_f.shape[0]} samples, Val size: {X_val.shape[0] if X_val is not None else 0} samples, Test size: {X_test.shape[0]} samples')
            logging.debug(f'Fraction of positive interactions in train: {round(sum(y_train_f)/len(y_train_f)*100,2)}%')
            if args.use_val:
                logging.debug(f'Fraction of positive interactions in val: {round(sum(y_val)/len(y_val)*100,2)}%')
            logging.debug(f'Fraction of positive interactions in test: {round(sum(y_test)/len(y_test)*100,2)}%')

        fold = 1 #used for n_epochs multiplier in later code
        # Convert to torch tensors
        X_train_t = torch.from_numpy(X_train_f).float()
        y_train_t = torch.from_numpy(y_train_f.reshape(-1, 1)).float()

        if args.use_val:
            X_val_t = torch.from_numpy(X_val).float()
            y_val_t = torch.from_numpy(y_val.reshape(-1, 1)).float()

        # Datasets / loaders
        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        if args.use_val:
            val_ds = TensorDataset(X_val_t, y_val_t)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        model = MLP(input_dim=X_train_f.shape[1]).to(device)
        criterion = nn.BCEWithLogitsLoss() #Loss function
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate) #Optimizes weights and biases

        # Training loop
        if args.logging: logging.info(f'Starting training with epochs: {args.n_epochs}...')
        for epoch in range(1, args.n_epochs + 1):
            model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)
            epoch_loss = running_loss / len(train_loader.dataset)
            train_losses.append(epoch_loss)

            if args.use_val:
                # Evaluate on validation set each epoch
                model.eval()
                with torch.no_grad():
                    total = 0
                    correct = 0
                    val_running_loss = 0.0
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        logits = model(xb)
                        loss = criterion(logits, yb)
                        val_running_loss += loss.item() * xb.size(0)
                        probs = torch.sigmoid(logits)
                        preds = (probs >= 0.5).float()
                        correct += (preds == yb).sum().item()
                        total += yb.numel()
                    val_loss = val_running_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else float('nan')
                    val_acc = correct / total if total > 0 else float('nan')
                    val_losses.append(val_loss)
                    val_accuracies.append(val_acc)

                print(f"Epoch {epoch:02d} - train_loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
                if args.logging: logging.debug(f'Epoch {epoch:02d} - train_loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}')
            else:
                print(f"Epoch {epoch:02d} - train_loss: {epoch_loss:.4f}")
                if args.logging: logging.debug(f'Epoch {epoch:02d} - train_loss: {epoch_loss:.4f}')
    
    # Appropriating test and excluded sets
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test.reshape(-1, 1)).float().to(device)
    test_ds = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    if args.test_on_excluded:
        X_test_unseen_t = torch.from_numpy(X_test_unseen).float().to(device)
        y_test_unseen_t = torch.from_numpy(y_test_unseen.reshape(-1, 1)).float().to(device)
    
    ### 9. Accuracy and training loss ###
    # Evaluation on test set: loss + accuracy --------
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t)
        test_loss = criterion(test_logits, y_test_t).item()
        test_probs = torch.sigmoid(test_logits)
        test_preds = (test_probs >= 0.5).float()
        test_acc = (test_preds == y_test_t).float().mean().item()
        #balanced accruacy calculation - cpu operation: move tensors back to the CPU before passing them to any scikit-learn function
        test_ba = balanced_accuracy_score(y_test_t.cpu().numpy(), test_preds.cpu().numpy())


    #print(f"\nFinal test loss: {test_loss:.4f}  test accuracy: {test_acc:.4f}")
    if args.logging: 
        if args.test_on_excluded:
            logging.info(f'\n{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Tested on excluded set')
        logging.info(f'\n{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Final test loss: {test_loss:.4f}  Standard test accuracy: {test_acc:.4f}  Standard test balanced accuracy: {test_ba:.4f}')
    print(f'Final test loss: {test_loss:.4f}  Standard test accuracy: {test_acc:.4f}  Standard test balanced accuracy: {test_ba:.4f}')
        
    
    # Plotting the losses 
    if args.use_val:
        fig,ax = plt.subplots(1,1, figsize=(9,5))

        ax.plot(range(args.n_epochs*fold), train_losses, label='Train loss', color='#FF8C00', linewidth=2)
        ax.plot(range(args.n_epochs*fold), val_losses, label='Val loss', color="#D88682", linewidth=2)
        ax.legend(loc='lower right')
        ax.set_ylabel('Loss')

        ax2 = ax.twinx()
        ax2.plot(range(args.n_epochs*fold), val_accuracies, label='Val accuracy', c='g', linestyle='--')
        ax2.set_ylabel('Accuracy')
        ax2.legend(loc='upper right')

        ax.set_xlabel('Epochs')
        fig.suptitle(f"Torch MLP Train/Val Loss & Val Accuracy for n{n}, k{k}. Test accuracy: {test_acc:.2f}, Test balanced accuracy: {test_ba:.2f}")

        outname = 'torchMLP_acc_loss.png'    
        if args.logging: 
            plt.savefig(outdir+outname)
            logging.info(f'Accuracy and train figure saved as: {outdir+outname}')
    else:
        if args.logging: logging.info(f'No validation set used, skipping loss and accuracy plotting.')

    # Evaluating on the truly unseen test set if applicable -------
    if args.test_on_excluded:
        with torch.no_grad():
            test_unseen_logits = model(X_test_unseen_t.to(device))
            test_unseen_loss = criterion(test_unseen_logits, y_test_unseen_t.to(device)).item()
            test_unseen_probs = torch.sigmoid(test_unseen_logits)
            test_unseen_preds = (test_unseen_probs >= 0.5).float()
            test_unseen_acc = (test_unseen_preds.to(device) == y_test_unseen_t).float().mean().item()
            test_unseen_ba = balanced_accuracy_score(y_test_unseen_t.cpu().numpy(), test_unseen_preds.cpu().numpy())

        if args.logging: 
            logging.info(f'Tested on truly unseen subset of excluded set')
            logging.info(f'Final truly unseen test loss: {test_unseen_loss:.4f}  truly unseen test accuracy: {test_unseen_acc:.4f}  truly unseen test balanced accuracy: {test_unseen_ba:.4f}')
        print(f'\nFinal truly unseen test loss: {test_unseen_loss:.4f}  truly unseen test accuracy: {test_unseen_acc:.4f}  truly unseen test balanced accuracy: {test_unseen_ba:.4f}')

    ### 10. Model Evaluations ###
    model.eval() # Set the model to evaluation mode
    all_logits = []
    all_labels = []

    with torch.no_grad(): # Disable gradient calculations for inference
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 1. Forward pass to get logits
            logits = model(inputs)

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate all results
    logits = np.concatenate(all_logits)
    true_labels = np.concatenate(all_labels)

    # Confusion matrix -----------
    probabilities = expit(logits).flatten()
    predicted_classes = (probabilities >= 0.5).astype(int)
    cm = confusion_matrix(true_labels, predicted_classes) # Calculate the confusion matrix
    # Interpretation:
    # cm[0, 0]: True Negatives (TN)
    # cm[0, 1]: False Positives (FP)
    # cm[1, 0]: False Negatives (FN)
    # cm[1, 1]: True Positives (TP)

    # Plotting the confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['True 0', 'True 1'])
    plt.title(f'Torch nn Confusion Matrix, n{n}, k{k}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')

    outname = 'torchMLP_confusion_matrix.png'
    if args.logging: 
        plt.savefig(outdir+outname)
        logging.info(f'Confusion matrix figure saved as: {outdir+outname}')

    # ROC Curve ------------------
    roc_auc = roc_auc_score(true_labels, probabilities)
    if args.logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} ROC AUC: {roc_auc:.4f}', file=logfile)
    fpr, tpr, thresholds = roc_curve(true_labels, probabilities)

    # 3. Plot the ROC Curve
    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--',
            label='Random Classifier') # Diagonal line for reference
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC Curve torch nn n{n}, k{k}')
    plt.legend(loc="lower right")
    plt.grid(True)

    outname = 'torchMLP_ROC.png'
    if args.logging: 
        plt.savefig(outdir+outname)
        logging.info(f'ROC curve figure saved as: {outdir+outname}')

    # Biparte --------------------
    J = tpr - fpr # Calculate Youden's J statistic

    # Find the index of the maximum J value
    optimal_idx = np.argmax(J)
    optimal_threshold = thresholds[optimal_idx]

    if args.logging:
        logging.info(f'\n--- Optimal Operating Point (Max Youdens J) ---')
        logging.info(f'Optimal Threshold: {optimal_threshold:.4f}')
        logging.info(f'Corresponding TPR: {tpr[optimal_idx]:.4f}')
        logging.info(f'Corresponding FPR: {fpr[optimal_idx]:.4f}')

    try:
        # 1. Load lookup data
        hostrange_pdf = pd.read_excel(raw_data_path+"phagehost_KU/Hostrange_data_all_crisp_iso.xlsx", sheet_name="sum_hostrange", header=1)
        id_lookup_bact = hostrange_pdf[["Seq ID", "Species"]].rename(columns={"Seq ID": "Bacterium_Name"})

        model.eval()
        with torch.no_grad():
            # Ensure X_test_t is your torch tensor for the test set
            test_logits = model(X_test_t.to(device))
            test_probs = torch.sigmoid(test_logits).cpu().numpy().flatten()

        # 2. Identify True Positives within the Test Set
        predicted_positive = (test_probs >= optimal_threshold)
        actual_positive = (y_test == 1) # y_test from your GSS split
        true_positive_mask = np.logical_and(predicted_positive, actual_positive)
        
        # Indices relative to the test set
        tp_test_indices = np.where(true_positive_mask)[0]

        # 3. DIRECT MAPPING (No re-splitting needed!)
        # Since metadata_test was already subset using test_idx in your GSS code:
        tp_pairs = metadata_test[tp_test_indices]
        
        tp_bacteria_names = tp_pairs[:, 0]
        tp_phage_names = tp_pairs[:, 1]
        tp_actual_scores = y_test[tp_test_indices]
        tp_predicted_probs = test_probs[tp_test_indices]

        # 4. Construct the Final DataFrame
        best_tp_df = pd.DataFrame({
            'Bacterium_Name': tp_bacteria_names,
            'Phage_Name': tp_phage_names,
            'Actual_Interaction_Score': tp_actual_scores,
            'Predicted_Probability': tp_predicted_probs
        })

        # Sort and Display
        best_tp_df = best_tp_df.sort_values(by='Predicted_Probability', ascending=False).reset_index(drop=True)

        # 5. Save and Plot
        if args.logging:
            logging.info(f'\n--- Top True Positive Entries (Grouped by strain) ---')
            logging.info(f'Total True Positives found in unseen test: {len(best_tp_df)}')
            for line in best_tp_df.head(10):
                logging.info(f'{line}')
            best_tp_df.to_csv(outdir+"best_predictions.csv", sep=";")
        
            plot_entity_counts(best_tp_df, 'Phage_Name', outdir = outdir, logging_on = args.logging)
            plot_entity_counts(best_tp_df, 'Bacterium_Name', outdir = outdir, logging_on = args.logging)
            plot_bipartite_network(best_tp_df, id_lookup_bact, outdir = outdir, logging_on = args.logging, limit=50, conf_threshold=0.5)

    except Exception as e:
        if args.logging:
            logging.error(f'Error during Biparte analysis: {e}\n{traceback.print_exc()}')

    # F1 Analysis -----------------
    probs = test_probs.flatten().cpu().numpy() if hasattr(test_probs, "cpu") else test_probs.flatten()
    y_true = y_test.flatten()  # already numpy
    if args.logging:
        f1_analysis(y_true, probs, logging=args.logging, outdir = outdir, logfile=logfile)

    ### Apply phage & bact to hostrange ###
    # Apply each phage & bacteria pair to the trained model and save predictions
    model.eval()
    thresh = globals().get('best_t', 0.5)  # use best_t if computed, otherwise fallback to 0.5

    # Build all valid pairs as a single matrix and run inference in one batched pass
    # This is far faster than calling scaler.transform(single_row) in a tight inner loop.
    pair_records = []
    all_pair_features = []
    for bact_name in tqdm(bacteria_names, desc="Building all-pairs matrix"):
        bact_idx = entity_to_index.get(bact_name)
        if bact_idx is None:
            continue
        for phage_name in phage_names:
            phage_idx = entity_to_index.get(phage_name)
            if phage_idx is None:
                continue
            combined = np.concatenate((binary_matrix[bact_idx], binary_matrix[phage_idx]))
            all_pair_features.append(combined)
            pair_records.append((bact_name, phage_name))

    if all_pair_features:
        all_pair_matrix = np.array(all_pair_features, dtype=np.float32)
        all_pair_scaled = scaler.transform(all_pair_matrix)
        all_pair_t = torch.from_numpy(all_pair_scaled).float()

        all_probs = []
        with torch.no_grad():
            pair_ds = TensorDataset(all_pair_t)
            pair_loader = DataLoader(pair_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
            for (xb,) in tqdm(pair_loader, desc="Running inference on all pairs"):
                logits = model(xb.to(device))
                probs_batch = torch.sigmoid(logits).cpu().numpy().flatten()
                all_probs.extend(probs_batch)

        all_probs = np.array(all_probs)
        all_preds = (all_probs >= thresh).astype(int)
        bact_names_col, phage_names_col = zip(*pair_records)
        results = [
            {"bacterium": b, "phage": p, "probability": prob, "prediction": pred}
            for b, p, prob, pred in zip(bact_names_col, phage_names_col, all_probs, all_preds)
        ]
    else:
        results = []

    # Save results to DataFrame + CSV
    pred_df = pd.DataFrame(results)
    if args.logging:
        outpath = os.path.join(outdir, "torchMLP_all_pairs_predictions.csv")
        pred_df.to_csv(outpath, index=False)

        print(f"Saved {len(pred_df)} predictions to {outpath}")
        logging.info(f'Saved {len(pred_df)} predictions to {outpath}')

    # Simply pred output matrix ---
    # create prediction matrix: rows=bacterium, cols=phage, values=prediction
    pred_matrix = pred_df.pivot_table(index='bacterium', columns='phage', values='prediction', aggfunc='max')

    # normalize column names (strip whitespace) then reorder columns to the requested phage order
    pred_matrix = pred_matrix.rename(columns=lambda x: x.strip())

    phage_order = [
        "Ymer","Taid","Poppous","Koroua","Abuela","Amona","Sabo","Mimer","Crus",
        "Gander","Guf","Hoejben","Magnum","Vims","Echoes","Galvinrad","Uther",
        "Rip","Rup","Slaad","Pantea","Rap","Zann"
    ]

    # keep only those desired that actually exist, then append any extra columns that were not listed
    cols_in_order = [c for c in phage_order if c in pred_matrix.columns]
    #rows_in_order = [c for c in pred_matrix.columns if c not in cols_in_order]
    final_cols = cols_in_order

    # ensure consistent ordering and include any missing rows/cols (fill missing pairs with 0)
    pred_matrix = pred_matrix.reindex(index=list(bacteria_names), columns=final_cols, fill_value=0)

    # save and show a quick preview
    if args.logging:
        logging.debug(f"Preview of ordered prediction matrix:\n{pred_matrix.head()}")
        outname = 'torchMLP_prediction_matrix_ordered.csv'
        pred_matrix.to_csv(outdir + outname)
        print(f"Saved ordered prediction matrix to {outdir + outname}")
        color_sheet_from_matrix(
            input_excel=raw_data_path + "phagehost_KU/Hostrange_data_all_crisp_iso.xlsx",
            sheet1_name="sum_hostrange",
            prediction_matrix_df=pred_matrix,
            excluded_bacteria=args.exclude_bact_clusters,
            excluded_phages=args.exclude_phage_clusters,
            output_excel=outdir + "hostrange_colored.xlsx", 
            TS=True
        )

    ### Feature Importance ###
    if args.perform_fi:
        fi = FeatureImportance(model, outdir, metadata_test, id_lookup_bact, host_range_data, 
                                raw_data_path, data_prod_path, TS = True, logging = args.logging, logfile=logfile)
        fi.compute_importance(X_test_t, target=0, delta=True)
        fi.plot_attributions()
        fi.plot_PCA(color_samples_by="bacteria")
        fi.plot_PCA(color_samples_by="phage")
        fi.plot_PCA(color_samples_by="interaction")

        # Do the attributions concur across samples?
        fi.plot_attributions_PCA_clusters() 

        # Regain kmer as string, given encoding
        try:
            fi.regain_kmers_fa(k=k, sourmash=sourmash_used, top_n=10, mapping_func=h.model_idx_to_kmer,
                                mapping_args=(binary_matrix.shape[1], feature_indices, idx_to_minhash))
            fi.plot_top_kmers(sourmash=sourmash_used, top_n=10)
        except Exception as e:
            print(f"Error during k-mer regaining: {e}")
            traceback.print_exc()

    ### Investigating Pairs ###
    if args.perform_pfi:
        hash_lookup = None
        pfi_failed = False
        out_pfi = f"pfi_{prefix}_n{n}_k{k}.txt"
        pfi_objects_dir = f"pfi_objects_{prefix}_n{n}_k{k}/"
        hash_lookup = "hash_lookup.csv"

        ### Subset host_range_data, phage_minhash_data, and bact_minhash_data to only include the strains present in the test set metadata
        phage_minhash_data_full = phage_minhash_data.copy()
        bact_minhash_data_full = bact_minhash_data.copy()
        if args.exclude_clusters:
            phage_minhash_data = {k: v for k, v in phage_minhash_data.items() if k in args.exclude_phage_clusters}
            bact_minhash_data = {k: v for k, v in bact_minhash_data.items() if k in args.exclude_bact_clusters}
        if args.logging: 
            print(f'Subsetted host range and minhash data to test set strains. Remaining bact strains: {len(host_range_data)}')
            print(f'Subsetted host range data: [{len(host_range_data)}x{len(next(iter(host_range_data.values())))}]')
            print(f'Remaining strains in phage minhash data: {list(phage_minhash_data.keys())}')
            print(f'Remaining strains in bacteria minhash data: {list(bact_minhash_data.keys())}')
            logging.info(f'Subsetted host range and minhash data to test set strains. Remaining strains: {len(host_range_data)}')
            logging.info(f'Remaining strains in host range data: {list(host_range_data.keys())}')
            logging.info(f'Remaining strains in phage minhash data: {list(phage_minhash_data.keys())}')
            logging.info(f'Remaining strains in bacteria minhash data: {list(bact_minhash_data.keys())}')

        ### Running PFI analysis and plotting results
        loaded_pfi = False # Flag for whehter the pfi results were loaded from memory or newly constructed
        run_match = re.search(r'run(\d+)', os.path.dirname(outdir.rstrip('/')))
        run_number = int(run_match.group(1)) if run_match else 0
        for i in range(1, run_number + 1):
            potential_parent = os.path.dirname(outdir.rstrip('/')).replace(f'run{run_number}', f'run{i}')
            potential_dir = f"{potential_parent}/{pfi_objects_dir}"
            logging.info(f'Checking for existing PFI results in potential directory: {potential_dir}')
            if os.path.isdir(potential_dir):
                logging.info(f'Found potential existing PFI results in {potential_dir}, attempting to load...')
                try:
                    interaction_pairs = joblib.load(potential_dir + "interaction_pairs.jbl")
                    occurence_pairs = joblib.load(potential_dir + "occurence_pairs.jbl")
                    interaction_freq_pairs = joblib.load(potential_dir + "interaction_freq_pairs.jbl")
                    occurence_freq_pairs = joblib.load(potential_dir + "occurence_freq_pairs.jbl")
                    expected_interactions = joblib.load(potential_dir + "expected_interactions.jbl")
                    hash_lookup = joblib.load(potential_dir + "hash_lookup.jbl")
                    loaded_pfi = True
                    if args.logging: logging.info(f'Successfully loaded existing PFI results from {outdir + pfi_objects_dir}')
                    break
                except Exception as e:
                    logging.error(f"Error loading existing PFI results: {e}")
                    interaction_pairs = None
        else:
            pfi_analyzer = calc_PFI(host_range_data=host_range_data, outdir=outdir, outname_pfi=out_pfi, pfi_objects_dir=pfi_objects_dir, logging=args.logging)
            interaction_pairs, occurence_pairs, interaction_freq_pairs, occurence_freq_pairs, expected_interactions, hash_lookup = pfi_analyzer.construct_interaction_pairs(phage_minhash_data=phage_minhash_data, bact_minhash_data=bact_minhash_data, subset=args.subset_pfi)
            if interaction_pairs is None:
                pfi_failed = True
                logging.error(f"PFI analysis failed during interaction pair construction - Check if test species interact.")
            elif args.logging: logging.info(f'Constructed interaction pairs and saved to {pfi_analyzer.outfile_pfi}')
        
        if hash_lookup is None and pfi_failed == False:
            try:
                hash_lookup = pd.read_csv(hash_lookup)
                if args.logging: logging.info(f'Successfully loaded hash lookup from {hash_lookup}')
            except Exception as e:
                logging.error(f"Error loading hash lookup: {e}")
        if args.logging and pfi_failed == False:
            logging.info(f'Total interacting pairs found: {len(interaction_pairs)}')
            logging.info(f'Total pairs with shared k-mers: {len(occurence_pairs)}')
            logging.info(f'Sample of interaction pairs:')
            max_c = 10
            for i, (pair, iscore) in enumerate(interaction_pairs.items()):
                logging.info(f'{pair}: Interaction Score = {iscore}')
                logging.info(f'{pair}: Occurrence Score = {occurence_pairs.get(pair, "N/A")}')
                if i >= max_c - 1:
                    break
                # for line in interaction_pairs[:10]:

        if args.use_encoded and hash_lookup is not None and pfi_failed == False: #can only regain string kmers from hash, if lookup dict has been made
            #plot_interaction_pairs(interaction_pairs, occurence_pairs, hash_lookup, logging=args.logging, outdir=outdir, bact_clusters=bact_clusters)
            plot_interaction_pairs(interaction_pairs, occurence_pairs, expected_interactions, hash_lookup, hk_translation_dict, sort_by_ratio=True, logging=args.logging, outdir=outdir, bact_clusters=bact_clusters)

            # Filter idx_to_minhash to only include the top X interaction pairs
            top_pairs = sorted(interaction_pairs.items(), key=lambda x: x[1], reverse=True)[:args.top_kmers_num] # Get top {args.top_kmers_num} pairs by interaction score
            top_minhashes = set()
            for (phage_hash, bact_hash), score in top_pairs:
                top_minhashes.add(phage_hash)
                top_minhashes.add(bact_hash)
            filtered_idx_to_minhash = {idx: mh for idx, mh in idx_to_minhash.items() if mh in top_minhashes}

            # Regain k-mers for the top interaction pairs
            pfi_top_kmers_df = None
            try:
                top_indices = [idx for idx, mh in filtered_idx_to_minhash.items()]
                regain_kmers_out = regain_kmers(k=k, n=n, prefix=prefix, sourmash=sourmash_used, top_n=args.top_kmers_num, 
                                                idx_to_minhash=filtered_idx_to_minhash, mapping_args=(binary_matrix.shape[1], feature_indices, idx_to_minhash), 
                                                logging=args.logging, logfile=logfile)
                pfi_top_idx, pfi_top_vals, pfi_top_kmers_decoded = regain_kmers_out
                if len(pfi_top_idx) == 0: 
                    pfi_failed = True
                    raise Exception("regain_kmers() failed")
                if args.logging: logging.info(f'Decoded k-mers for top interaction pairs: {list(pfi_top_kmers_decoded.values())}')
                pfi_top_kmers_df = pd.DataFrame({
                    "feature_index": list(pfi_top_kmers_decoded.keys()),
                    "entity": [idx_to_entity.get(idx, "unknown") for idx in pfi_top_kmers_decoded.keys()],
                    "organism": ["bacterium" if idx_to_entity.get(idx, "unknown") in bact_minhash_data_full.keys() else ("phage" if idx_to_entity.get(idx, "unknown") in phage_minhash_data_full.keys() else "unknown") for idx in pfi_top_kmers_decoded.keys()],
                    "decoded_kmer": list(pfi_top_kmers_decoded.values())
                })
                pfi_top_kmers_df.to_csv(outdir+"top_interaction_pair_kmers.csv", index=False)
                if args.logging: logging.info(f'Saved decoded k-mers for top interaction pairs to {outdir+"top_interaction_pair_kmers.csv"}')
            except Exception as e:
                logging.error(f"Error during k-mer regaining for top interaction pairs: {e}")
                pfi_failed = True
            
            #Plot the top k-mers for interaction pairs
            if pfi_top_kmers_df is not None and not pfi_top_kmers_df.empty:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x='feature_index', y='decoded_kmer', data=pfi_top_kmers_df.head(20))
                plt.title('Top 20 k-mers for Interaction Pairs on Feature Index')
                plt.xlabel('Feature Index')
                plt.xlim(0, binary_matrix.shape[1]) # Set x-axis limits to the range of feature indices
                plt.ylabel('Decoded k-mer')
                plt.xticks(rotation=45)
                outname = 'top_20_interaction_pair_kmers.png'
                if args.logging: 
                    plt.tight_layout()
                    plt.savefig(outdir + outname)
                    logging.info(f'Saved plot of top k-mers for interaction pairs to {outdir + outname}')
    
    ### Gene Annotation of Kmers ###
    # If PFI was performed and yielded results, use those top k-mers with annotations for GeneAnalysisNCBI. Otherwise, regain the top k-mers from the model feature importance and use those for GeneAnalysisNCBI.
    if args.perform_ga:
        if args.run_ga_on_pfi and pfi_top_kmers_df is not None and not pfi_top_kmers_df.empty and not pfi_failed:
            kmers_entity_df = pfi_top_kmers_df
            if args.logging: logging.info(f'Using top interaction pairs from pairwise feature importance analysis with entity annotations for GeneAnalysisNCBI.')
        else:
            try:
                indices, vals, all_kmers_decoded = regain_kmers(k=k, sourmash=sourmash_used, top_n=100, 
                    idx_to_minhash=idx_to_minhash,
                    mapping_args=(binary_matrix.shape[1], feature_indices, idx_to_minhash), 
                    logging=args.logging, logfile=logfile)
                if args.logging: logging.info(f'Decoded k-mers (subset): {[all_kmers_decoded[i] for i in range(5)]}')
                kmers_entity_df = pd.DataFrame([
                    {
                        "feature_index": idx,
                        "entity": idx_to_entity.get(idx, "unknown"),
                        "organism": "bacterium" if idx_to_entity.get(idx, "unknown") in bact_minhash_data_full.keys() else ("phage" if idx_to_entity.get(idx, "unknown") in phage_minhash_data_full.keys() else "unknown"),
                        "decoded_kmer": all_kmers_decoded[i] if i < len(all_kmers_decoded) else None
                    }
                    for i, idx in enumerate(indices)
                ])

            except Exception as e:
                logging.error(f"Error during k-mer regaining for annotation: {e}")

        try:
            GA = GeneAnalysisNCBI(logfile=logfile, logging=args.logging, outdir=outdir)
            phage_kmers_decoded_df = kmers_entity_df[kmers_entity_df["organism"] == "phage"]
            bact_kmers_decoded_df = kmers_entity_df[kmers_entity_df["organism"] == "bacterium"]

            logging.info("Phage_df - Started k-mer annotation with GeneAnalysisNCBI...")
            #ncbi_blast_res_df = GA.search_and_annotate_kmers(phage_kmers_decoded_df, organism="phage", summarise_by="function", tax_origin="txid38018[orgn]") # Phage first
            ncbi_blast_res_df = GA.search_and_annotate_kmers(phage_kmers_decoded_df, organism="phage", tax_origin="txid38018[orgn]") # Phage first
            if ncbi_blast_res_df is not None and not ncbi_blast_res_df.empty:
                number_of_blast_res_before = len(ncbi_blast_res_df)
                for genus in bact_clusters['genus'].unique():
                    subset_bact_kmers_decoded_df = bact_kmers_decoded_df[bact_kmers_decoded_df["entity"].isin(bact_clusters[bact_clusters["genus"] == genus].index)]
                    for i in tqdm(range(10), desc=f"Waiting 10 secs"): #sleep for 10 seconds to avoid overwhelming NCBI with back-to-back requests
                        sleep(1)
                    tax_str = f"txid{strain_id_tax_lookup.get(genus, 'unknown')}[orgn]" if strain_id_tax_lookup.get(genus, None) is not None else "txid2[orgn]"
                    logging.info(f"Bact_df - Started k-mer annotation with GeneAnalysisNCBI for genus {genus}, tax_origin={tax_str}...")
                    ncbi_bact_df = GA.search_and_annotate_kmers(subset_bact_kmers_decoded_df, organism="bact", tax_origin=tax_str, expect=10000)
                    ncbi_blast_res_df = pd.concat([ncbi_blast_res_df, ncbi_bact_df], ignore_index=True) # Bact second
                
                if len(ncbi_blast_res_df) > number_of_blast_res_before:
                    logging.info(f'GeneAnalysisNCBI completed and results saved to {outdir+"GA_kmers_blast_results.csv"}')
                else:
                    raise ValueError("No BLAST results were obtained for bacterial k-mers. Please check the input data and BLAST parameters.")
            else:
                raise ValueError("No BLAST results were obtained for phage k-mers. Please check the input data and BLAST parameters.")
            
            try: 
                if args.logging: 
                    logging.info(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Fraction of kmers annotated: {len(ncbi_blast_res_df)}/{len(kmers_entity_df)}')
                ncbi_blast_res_df = ncbi_blast_res_df.merge(kmers_entity_df[["feature_index", "entity", "organism", "decoded_kmer"]], left_on="Kmer", right_on="decoded_kmer", how="right")
                ncbi_blast_res_df.to_csv(outdir+"GA_kmers_blast_results.csv", index=False)
            except Exception as e:
                logging.error(f"Error during merging BLAST results with k-mer annotations: {e}")

        except Exception as e:
            logging.error(f"Error during GeneAnalysisNCBI: {e}")
    
    # Plotting the distribution of annotated Function and Genes of kmers
    if ncbi_blast_res_df is not None and not ncbi_blast_res_df.empty:
        GA.plot_annotated_kmer_statistics(ncbi_blast_res_df)


    ### Optional: Save the trained model for future use ###
    if args.save_model:
        model_save_path = outdir + "torchMLP_model.pth"
        torch.save(model.state_dict(), model_save_path)
        if args.logging: logging.info(f'Trained model saved to {model_save_path}')

    ### X. Closing & Terminating ###
    print(f"\nProcess completed in {time() - time_start:.2f} seconds.")
    logging.info(f"Process completed in {time() - time_start:.2f} seconds.")
    if logfile: logfile.close()

if __name__ == "__main__":
    main()