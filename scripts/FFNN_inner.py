#!/usr/bin/python3

import os
import sys
import argparse
import random
import pickle
import traceback
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from time import time
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split, GroupShuffleSplit
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from scipy.special import expit
from imblearn.over_sampling import SMOTE

# Custom imports
from io_operations import presence_matrix, call_hostrange_df, color_sheet_from_matrix
from paths import raw_data_path, data_prod_path, path_to_nn_runs
from manipulations import hostrange_df_to_dict, binarize_host_range
from analysis import (f1_analysis, plot_entity_counts, plot_bipartite_network, 
                      FeatureImportance)

def parse_arguments():
    parser = argparse.ArgumentParser(description="FFNN Training Script")

    # Parameters: Mutual exclusivity for n/k vs specific bn/bk/pn/pk
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--nk", nargs=2, type=int, metavar=('N', 'K'),
                        help="Unified n and k values (e.g., -nk 500 12)")
    group.add_argument("--split_nk", nargs=4, type=int, metavar=('BN', 'BK', 'PN', 'PK'),
                        help="Split values for Bact (n, k) and Phage (n, k)")

    # Data Source
    parser.add_argument("--use_encoded", action="store_true", 
                        help="Use encoded_sketches instead of SM_sketches")
    parser.add_argument("--bits_encoded", type=str, default="4", 
                        help="(Optional) specify which type of bit encoding using in encoded_sketches (e.g. 4 for phage_encode4bit_n400_k12)")
    parser.add_argument("--out", type=str, help="custom directory to write to in nn_runs/")
    parser.add_argument("--sbatch_id", type=str, help="(Optional) sbatch job ID to include in output directory name for easier tracking")

    # Flags
    parser.add_argument("--logging", action="store_true", help="Enable logging and saving")
    parser.add_argument("--cv", action="store_true", help="Enable cross-validation")
    parser.add_argument("--kf_n_splits", type=int, default=1, help="Number of folds for K-Fold CV")
    parser.add_argument("--smote", action="store_true", help="Apply SMOTE oversampling")
    parser.add_argument("--train_by_cluster", action="store_true", help="Split data by bacterial clusters")
    parser.add_argument("--no_randomize", action="store_false", dest="randomize", help="Disable entity randomization")
    parser.add_argument("--no_shuffle", action="store_false", dest="shuffle", help="Disable feature shuffling")
    parser.add_argument("--entity_order", choices=["bact_first", "phage_first"], default="bact_first", help="Choose order of input vector; bact first then phage is the default.")

    # Exclusions
    parser.add_argument("--exclude_noninteractions", action="store_true", help="Exclude non-interacting pairs")
    parser.add_argument("--exclude_bacts", nargs='+', default=["J26_21_reoriented"], help="List of bacteria to exclude")
    parser.add_argument("--exclude_phages", nargs='+', default=["Abuela"], help="List of phages to exclude")

    # Hyperparameters
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--val_split", type=float, default=0.2)

    args = parser.parse_args()

    # --- VALIDATION LOGIC FOR ARGUMENTS ---
    # Requirement: kf_n_splits must be > 1 if --cv is on
    if args.cv and args.kf_n_splits <= 1:
        parser.error("--kf_n_splits must be greater than 1 when --cv is enabled.")

    # Requirement: exclude_noninteractions requires exclude_bacts and exclude_phages
    if args.exclude_noninteractions and (len(args.exclude_bacts) != 1 or len(args.exclude_phages) != 1):
        parser.error("--exclude_noninteractions requires both --exclude_bacts and --exclude_phages lists.")

    return args

# This function maps model feature index 'idx' back to the encoded k-mer
def model_idx_to_kmer(idx, num_features_per_entity, feature_indices, idx_to_minhash):
    # Determine if it's a Bact feature (idx < N) or Phage feature (idx >= N)
    original_col_idx = feature_indices[idx % num_features_per_entity]
    return idx_to_minhash[original_col_idx]

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

def main():
    args = parse_arguments()
    time_start = time()

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

    if args.use_encoded:
        input_phage_path = f"{prefix}/phage_encode{args.bits_encoded}bit_n{pn}_k{pk}/"
        input_bact_path = f"{prefix}/bact_encode{args.bits_encoded}bit_n{bn}_k{bk}/"
    else:
        input_phage_path = f"{prefix}/PhageMinhash_n{pn}_k{pk}/"
        input_bact_path = f"{prefix}/BactMinhash_n{bn}_k{bk}/"
    presmat_path = f"{prefix}/PresMat_{presmat_suffix}/"
    print(f"Recognized data paths\ninput_phage_path:\t{input_phage_path}\ninput_bact_path:\t{input_bact_path}\npresmat_path:\t{presmat_path}")

    ### 3. Load Data ###
    bact_clusters = pd.read_csv(os.path.join(data_prod_path, "bact_clusters.csv"), index_col=0)
    
    # Load Presence Matrix
    full_presmat_path = os.path.join(data_prod_path, presmat_path)
    if not os.path.exists(full_presmat_path):
        print("Reconstructing presence_matrix...")
        binary_matrix, entity_to_index, minhash_to_index, phage_minhash_data, bact_minhash_data = presence_matrix(
            phage_minhash_dir=os.path.join(data_prod_path, input_phage_path),
            bact_minhash_dir=os.path.join(data_prod_path, input_bact_path),
            k=[bk, pk], n=[bn, pn], reversecomp_data=False, TS=True)
    else:
        with open(os.path.join(full_presmat_path, "binary_matrix.pkl"), "rb") as f: binary_matrix = pickle.load(f)
        with open(os.path.join(full_presmat_path, "entity_to_index.pkl"), "rb") as f: entity_to_index = pickle.load(f)
        with open(os.path.join(full_presmat_path, "phage_minhash_data.pkl"), "rb") as f: phage_minhash_data = pickle.load(f)
        with open(os.path.join(full_presmat_path, "bact_minhash_data.pkl"), "rb") as f: bact_minhash_data = pickle.load(f)
        with open(os.path.join(full_presmat_path, "minhash_to_index.pkl"), "rb") as f: minhash_to_index = pickle.load(f) 

    ### 4. Host Range Setup ###
    bact_lookup, host_range_df = call_hostrange_df(os.path.join(raw_data_path, "phagehost_KU/Hostrange_data_all_crisp_iso.xlsx"))
    host_range_data = binarize_host_range(hostrange_df_to_dict(host_range_df), continous=False)

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
        logfile = open(os.path.join(outdir, f'log_run{run}.txt'), 'w')
        logfile.write(f'Run started: {datetime.now()}\nParams: {vars(args)}\n')

    ### 6. Feature Preparation ###
    X, y, X_excl, y_excl, rows_meta = [], [], [], [], []
    phage_names = list(phage_minhash_data.keys())
    bacteria_names = list(bact_minhash_data.keys())

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
    
    # Create inverse mapping: column_index -> kmer_encoded_int
    idx_to_minhash = {v: k for k, v in minhash_to_index.items()}

    for bact in tqdm(bacteria_names, desc="Building dataset"):
        # Exclusion logic
        if args.exclude_noninteractions and not any(host_range_data.get(bact, {}).values()):
            continue
            
        for phage in phage_names:
            if phage not in host_range_data.get(bact, {}): continue

            score = host_range_data[bact][phage]
            if args.entity_order == "bact_first":
                features = np.concatenate((binary_matrix[entity_to_index[bact]], 
                                       binary_matrix[entity_to_index[phage]]))
            elif args.entity_order == "phage_first":
                features = np.concatenate((binary_matrix[entity_to_index[phage]], 
                                       binary_matrix[entity_to_index[bact]]))

            if args.exclude_noninteractions and (bact in args.exclude_bacts or phage in args.exclude_phages):
                X_excl.append(features)
                y_excl.append(score)    
                continue

            X.append(features)
            y.append(score)
            if args.entity_order == "bact_first":
                rows_meta.append((bact, phage))
            elif args.entity_order == "phage_first":
                rows_meta.append((phage, bact))
            else:
                raise ValueError("Invalid entity_order argument. Must be 'bact_first' or 'phage_first'.")

    X, y = np.array(X), np.array(y)
    X_excl, y_excl = np.array(X_excl), np.array(y_excl)
    
    ### 7. Splitting & Scaling ###
    if args.train_by_cluster:
        groups = bact_clusters.loc[[m[0] for m in rows_meta], 'Cluster'].values
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_split, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))

    else:
        train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=args.test_split, random_state=42, stratify=y)

    X_train_f, X_test, y_train_f, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    
    metadata_np = np.array(rows_meta, dtype=object)
    metadata_train_full, metadata_test = metadata_np[train_idx], metadata_np[test_idx]

    if args.logging: 
        print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Train size: {X_train_f.shape[0]} samples, Test size: {X_test.shape[0]} samples', file=logfile)
        print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Fraction of positive interactions in train: {round(sum(y_train_f)/len(y_train_f)*100,2)}%', file=logfile)
        print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Fraction of positive interactions in test: {round(sum(y_test)/len(y_test)*100,2)}%', file=logfile)

    scaler = StandardScaler()
    X_train_f = scaler.fit_transform(X_train_f)
    X_test = scaler.transform(X_test)

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
    kf = KFold(n_splits=args.kf_n_splits, shuffle=True, random_state=42)
    fold = 1

    if args.logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Starting cross-validation with {kf.get_n_splits()} folds...', file=logfile)

    for train_idx, val_idx in kf.split(X_train_f):
        print(f"Fold {fold}:")
        if args.logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Fold {fold}...', file=logfile)

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
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

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
                optimizer.zero_grad()
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
            if args.logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Epoch {epoch:02d} - train_loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}', file=logfile)

        fold += 1
    fold -= 1 # Adjust fold count after loop to reflect actual number of folds completed

    # Appropriating test and excluded sets
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test.reshape(-1, 1)).float().to(device)
    X_excluded_t = torch.from_numpy(X_excl).float().to(device) if X_excl.size > 0 else None 
    y_excluded_t = torch.from_numpy(y_excl.reshape(-1, 1)).float().to(device) if y_excl.size > 0 else None

    test_ds = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    #X_excluded_ds = TensorDataset(X_excluded_t, y_excluded_t) if len(X_excl) > 0 else None
    #X_excluded_loader = DataLoader(X_excluded_ds, batch_size=args.batch_size, shuffle=False) if X_excluded_ds is not None else None
    #y_excluded_ds = TensorDataset(y_excluded_t) if len(y_excl) > 0 else None
    #y_excluded_loader = DataLoader(y_excluded_ds, batch_size=args.batch_size, shuffle=False) if y_excluded_ds is not None else None

    ### 9. Accuracy and training loss ###
    # Final evaluation on test set: loss + accuracy
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t.to(device))
        test_loss = criterion(test_logits, y_test_t.to(device)).item()
        test_probs = torch.sigmoid(test_logits)
        test_preds = (test_probs >= 0.5).float()
        test_acc = (test_preds.to(device) == y_test_t).float().mean().item()

    #print(f"\nFinal test loss: {test_loss:.4f}  test accuracy: {test_acc:.4f}")
    if args.logging: print(f'\n{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Final test loss: {test_loss:.4f}  test accuracy: {test_acc:.4f}', file=logfile)
    
    # Plotting the losses 
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
    fig.suptitle(f"Torch MLP Train/Val Loss & Val Accuracy for n{n}, k{k}. Test accuracy: {test_acc:.2f}")

    outname = 'torchMLP_acc_loss.png'    
    if args.logging: plt.savefig(outdir+outname)
    if args.logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Accuracy and train figure saved as: {outdir+outname}', file=logfile)

    # Eval on excluded set (if toggled and data exists)
    if args.exclude_noninteractions and X_excluded_t is not None:
        model.eval()
        with torch.no_grad():
            excluded_logits = model(X_excluded_t)
            excluded_probs = torch.sigmoid(excluded_logits).cpu().numpy().flatten()
            excluded_preds = (excluded_probs >= 0.5).astype(int)

        print(f"\n--- Excluded Pairs Results ---")
        for i in range(len(excluded_preds)):
            # Note: You'll need to track the names of excluded pairs if you want to print them specifically here
            line = f"Excluded pair {i} - prediction: {excluded_preds[i]} (prob: {excluded_probs[i]:.4f}), actual: {y_excl[i]}"
            print(line)
            if args.logging: 
                print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} {line}', file=logfile)

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
        print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Confusion matrix figure saved as: {outdir+outname}', file=logfile)

    # ROC Curve ------------------
    roc_auc = roc_auc_score(true_labels, probabilities)
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
        print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} ROC curve figure saved as: {outdir+outname}', file=logfile)

    # Biparte --------------------
    J = tpr - fpr # Calculate Youden's J statistic

    # Find the index of the maximum J value
    optimal_idx = np.argmax(J)
    optimal_threshold = thresholds[optimal_idx]

    if args.logging:
        print(f'\n{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} --- Optimal Operating Point (Max Youdens J) ---', file=logfile)
        print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Optimal Threshold: {optimal_threshold:.4f}', file=logfile)
        print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Corresponding TPR: {tpr[optimal_idx]:.4f}', file=logfile)
        print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Corresponding FPR: {fpr[optimal_idx]:.4f}', file=logfile)

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
            print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} \n--- Top True Positive Entries (Grouped by Cluster) ---')
            print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Total True Positives found in unseen clusters: {len(best_tp_df)}')
            for line in best_tp_df.head(10):
                print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} {line}', file=logfile)
            best_tp_df.to_csv(outdir+"best_predictions.csv", sep=";")
        
            plot_entity_counts(best_tp_df, 'Phage_Name', outdir = outdir, logging = args.logging)
            plot_entity_counts(best_tp_df, 'Bacterium_Name', outdir = outdir, logging = args.logging)
            plot_bipartite_network(best_tp_df, id_lookup_bact, outdir = outdir, logging = args.logging, limit=50, conf_threshold=0.5)

    except Exception as e:
        if args.logging:
            print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Error during Biparte analysis: {e}\n{traceback.print_exc()}', file=logfile)

    # F1 Analysis -----------------
    probs = test_probs.flatten().cpu().numpy() if hasattr(test_probs, "cpu") else test_probs.flatten()
    y_true = y_test.flatten()  # already numpy
    if args.logging:
        f1_analysis(y_true, probs, logging=args.logging, outdir = outdir, logfile=logfile)

    ### Apply phage & bact to hostrange ###
    # Apply each phage & bacteria pair to the trained model and save predictions
    model.eval()
    results = []
    thresh = globals().get('best_t', 0.5)  # use best_t if computed, otherwise fallback to 0.5

    with torch.no_grad():
        for bact_name in tqdm(bacteria_names, desc="Bacteria names iterated"):
            for phage_name in phage_names:
                # skip pairs that don't exist in entity_to_index
                try:
                    bact_index = entity_to_index[bact_name]
                    phage_index = entity_to_index[phage_name]
                except KeyError:
                    continue

                # build combined feature vector like in training
                bact_features = binary_matrix[bact_index, :]
                phage_features = binary_matrix[phage_index, :]
                combined = np.concatenate((bact_features, phage_features)).astype(np.float32).reshape(1, -1)

                # scale and convert to tensor
                scaled = scaler.transform(combined)
                x_t = torch.from_numpy(scaled).float().to(device)

                # inference
                logits = model(x_t)
                prob = torch.sigmoid(logits).item()
                pred = int(prob >= thresh)

                results.append({
                    "bacterium": bact_name,
                    "phage": phage_name,
                    "probability": prob,
                    "prediction": pred
                })

    # Save results to DataFrame + CSV
    pred_df = pd.DataFrame(results)
    if args.logging:
        outpath = outdir + "torchMLP_all_pairs_predictions.csv"
        pred_df.to_csv(outpath, index=False)

        print(f"Saved {len(pred_df)} predictions to {outpath}")
        print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Saved {len(pred_df)} predictions to {outpath}', file=logfile)

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
        print(pred_matrix.head())
        outname = 'torchMLP_prediction_matrix_ordered.csv'
        pred_matrix.to_csv(outdir + outname)
        print(f"Saved ordered prediction matrix to {outdir + outname}")
        color_sheet_from_matrix(
            input_excel=raw_data_path + "phagehost_KU/Hostrange_data_all_crisp_iso.xlsx",
            sheet1_name="sum_hostrange",
            prediction_matrix_df=pred_matrix,
            output_excel=outdir + "hostrange_colored.xlsx", 
            TS=True
        )

    ### Feature Importance ###
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
        fi.regain_kmers(k=k, sourmash=sourmash_used, top_n=10, mapping_func=model_idx_to_kmer,
                            mapping_args=(binary_matrix.shape[1], feature_indices, idx_to_minhash))
        fi.plot_top_kmers(sourmash=sourmash_used, top_n=10)
    except Exception as e:
        print(f"Error during k-mer regaining: {e}")
        traceback.print_exc()

    ### X. Closing & Terminating ###
    print(f"Process completed in {time() - time_start:.2f} seconds.")
    if logfile: logfile.close()

if __name__ == "__main__":
    main()