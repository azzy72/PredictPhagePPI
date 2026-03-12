# Setting up and obtaining unique minhashes

import os, sys
from tqdm import tqdm
import pandas as pd
import numpy as np
from io_operations import presence_matrix
from paths import raw_data_path, data_prod_path, path_to_nn_runs
from time import time
from datetime import datetime
import random
import traceback
import pickle 

time_start = time()

# raw_data_path = "../raw_data/"
# data_prod_path = "../data_prod/"
bact_clusters = pd.read_csv(data_prod_path+"bact_clusters.csv", index_col=0)
bn = 500
pn = 500
bk = 12
pk = 12
n = 500
k = 500
logging = False
cross_validation_on = False
smote_on = False
randomize_entity_order = True #Bacteria & Phage Order Randomized
shuffle_feature_order = True #Shuffled input vector
train_by_cluster = False

# Exclusions
exclude_noninteractions = True
exclude_bacts = ["J26_21_reoriented"]
exclude_phages = ["Abuela"]

# Hyperparameters
n_epochs = 50
learning_rate = 1e-3
batch_size = 64 # adjust based on dataset size and memory
test_split_ratio = 0.2
val_split_ratio = 0.2

## Choose either sets:
input_phage_path = f"SM_sketches/PhageMinhash_n{pn}_k{pk}_rev/" 
input_bact_path = f"SM_sketches/BactMinhash_n{bn}_k{bk}_rev/" 
sourmash_used = True

# input_phage_path = f"encoded_sketches/phage_encode4bit_n{pn}_k{pk}/"
# input_bact_path = f"encoded_sketches/bact_encode4bit_n{bn}_k{bk}/"
# sourmash_used = False

### Load presence_matrix() results
presmat_path = f"SM_sketches/PresMat_bn{bn}_bk{bk}_pn{pn}_pk{pk}/"
if not os.path.exists(data_prod_path+presmat_path):
    try: 
        print("Reconstruct presence_matrix() results (sourmash dependent)")
        binary_matrix, entity_to_index, minhash_to_index, phage_minhash_data, bact_minhash_data = presence_matrix(
            phage_minhash_dir=data_prod_path+input_phage_path, 
            bact_minhash_dir=data_prod_path+input_bact_path,
            k=[bk, pk],
            n=[bn, pn],
            reversecomp_data=False, TS=True)
    except ValueError as e:
        raise ValueError("Unable to find or recreate presence_matrix results")
else:
    try:
        with open(data_prod_path+presmat_path+"binary_matrix.pkl", "rb") as binary_matrix_file:
            binary_matrix = pickle.load(binary_matrix_file)
        with open(data_prod_path+presmat_path+"entity_to_index.pkl", "rb") as entity_to_index_file:
            entity_to_index = pickle.load(entity_to_index_file)
        with open(data_prod_path+presmat_path+"minhash_to_index.pkl", "rb") as minhash_to_index_file:
            minhash_to_index= pickle.load(minhash_to_index_file)
        with open(data_prod_path+presmat_path+"phage_minhash_data.pkl", "rb") as phage_minhash_data_file:
            phage_minhash_data = pickle.load(phage_minhash_data_file)
        with open(data_prod_path+presmat_path+"bact_minhash_data.pkl", "rb") as bact_minhash_data_file:
            bact_minhash_data = pickle.load(bact_minhash_data_file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Unable to load presence matrix result files, exception:\n{e}")    

# This check produces a syntax error 
# for file in [binary_matrix, entity_to_index, minhash_to_index, phage_minhash_data, bact_minhash_data]:
#     if file == None:
#         raise FileNotFoundError(f"Unable to load presence matrix resul file: {file}")

from io_operations import call_hostrange_df
bact_lookup, host_range_df = call_hostrange_df(raw_data_path + "phagehost_KU/Hostrange_data_all_crisp_iso.xlsx")
print(bact_lookup)
print(host_range_df)

from manipulations import hostrange_df_to_dict, binarize_host_range

# Convert the host range data into a dictionary
host_range_data = hostrange_df_to_dict(host_range_df)
host_range_data = binarize_host_range(host_range_data, continous=False) #for classification model

#host_range_data["J10_21_reoriented"]

outdir = None
logfile = None
if logging:
    run = 1
    #path_to_nn_runs = "../nn_runs/"
    if smote_on:
        outdirname = f'torch_mlp_n{n}_k{k}_smote'
    else:
        outdirname = f'torch_mlp_n{n}_k{k}'

    outdir = path_to_nn_runs+outdirname+f'_run{run}/'

    while os.path.exists(outdir):
        run += 1
        outdir = path_to_nn_runs+outdirname+f'_run{run}/'

    os.makedirs(outdir, exist_ok=True)
    print("Output directory:", outdir)

    # Open logfile for run
        
    logfile = open(outdir+f'torchMLP_log_run{run}.txt', 'w')
    logfile.write(f'Torch MLP log for n={n}, k={k}\n')
    logfile.write(f'Data type used: {"SM sketches" if sourmash_used else "encoded sketches"}\n')
    logfile.write(f'Cross-validation (KFold): {"Yes" if cross_validation_on else "No"}\n')
    logfile.write(f'SMOTE applied: {"Yes" if smote_on else "No"}\n')
    logfile.write(f'Excluding non-interactions: {"Yes" if exclude_noninteractions else "No"}\n')
    logfile.write(f'Training by cluster: {"Yes" if train_by_cluster else "No"}\n')
    logfile.write(f'Randomizing order of entities: {"Yes" if randomize_entity_order else "No"}\n')
    logfile.write(f'Shuffling feature order: {"Yes" if shuffle_feature_order else "No"}\n')
    logfile.write(f'Excluding bacteria: {exclude_bacts if exclude_noninteractions else "None"}\n')
    logfile.write(f'Excluding phages: {exclude_phages if exclude_noninteractions else "None"}\n')
    logfile.write(f'Path to input phage data: {data_prod_path+input_phage_path}\n')
    logfile.write(f'Path to input bact data: {data_prod_path+input_bact_path}\n')
    logfile.write('-----------------------------------\n')

X = []
y = []
X_excluded = []
y_excluded = []
rows_metadata = [] # To keep track of which entities form the row

phage_names = phage_minhash_data.keys()
bacteria_names = bact_minhash_data.keys()

# Randomize order of bacteria and phage names to avoid any ordering bias
if randomize_entity_order:
    bacteria_names = list(bacteria_names)
    phage_names = list(phage_names)
    random.seed(42)
    random.shuffle(bacteria_names)
    random.shuffle(phage_names)

# Shuffle feature order via concurrent schema
if shuffle_feature_order:
    # Since the feature order is determined by the order of minhashes in the binary matrix, 
    # we can shuffle the columns of the binary matrix to ultimately reorder bact and phage features.
    if logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Shuffling feature order...', file=logfile)
    feature_indices = list(range(binary_matrix.shape[1]))
    #if logging: print(feature_indices, file=logfile)
    random.seed(42)
    random.shuffle(feature_indices)
    #if logging: print(feature_indices, file=logfile)
    binary_matrix = binary_matrix[:, feature_indices]

# Iterate through all valid phage-bacteria pairs (the required pairwise iteration)
for bact_name in tqdm(bacteria_names, desc="Bacteria names iterated"):
    if exclude_noninteractions:
        # Skip bacterias that have no recorded interactions with any phage in the host range data
        if not any(host_range_data.get(bact_name, {}).values()):
            print(f"Skipping non-interacting bacteria: {bact_name}")
            if logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Skipping non-interacting bacteria: {bact_name}', file=logfile)
            continue
    for phage_name in phage_names:    
        # Get the interaction score (target variable y)
        try:
            interaction_score = host_range_data[bact_name][phage_name]
        except KeyError:
            continue

        # Get the feature vectors (rows from the incidence matrix)
        bact_index = entity_to_index[bact_name]
        phage_index = entity_to_index[phage_name]

        bact_features = binary_matrix[bact_index, :]
        phage_features = binary_matrix[phage_index, :]

        # Concatenate: [Bacterium Features | Phage Features]
        combined_features = np.concatenate((bact_features, phage_features))

        # Skip pairs that are in the exclusion lists (added to log and stored separately for testing the effect of unforseen strains)
        if bact_name in exclude_bacts or phage_name in exclude_phages:
            print(f"Excluding pair: {bact_name} - {phage_name}")
            if logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Excluding pair: {bact_name} - {phage_name}', file=logfile)
            if bact_name in exclude_bacts and phage_name in exclude_phages:
                X_excluded.append(combined_features)
                y_excluded.append(interaction_score)
        else:
            X.append(combined_features)
            y.append(interaction_score)
            rows_metadata.append((bact_name, phage_name))
        #print(X)
        #print(y)
        

X = np.array(X)
y = np.array(y)
X_excluded = np.array(X_excluded)
y_excluded = np.array(y_excluded)

print("Unique values found in train y:", set(y))
if logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Unique values found in train y:', set(y), file=logfile)
print(f"Percent zeros in train y: {round(([sum(val == 0 for val in y)][0]/len(y))*100,2)}%")
if logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Percent zeros in train y: {round(([sum(val == 0 for val in y)][0]/len(y))*100,2)}%', file=logfile)

if len(X_excluded) > 0 and exclude_noninteractions:
    print("Unique values found in excluded y:", set(y_excluded))
    if logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Unique values found in excluded y:', set(y_excluded), file=logfile)
    print(f"Percent zeros in excluded y: {round(([sum(val == 0 for val in y_excluded)][0]/len(y_excluded))*100,2)}%")
    if logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Percent zeros in excluded y: {round(([sum(val == 0 for val in y_excluded)][0]/len(y_excluded))*100,2)}%', file=logfile)

# Check if we have enough data to proceed
if X.shape[0] < 2:
    print(f"Error: Not enough data points ({X.shape[0]} found) for train-test split.")
    if logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Error: Not enough data points ({X.shape[0]} found) for train-test split.', file=logfile)
    sys.exit(1)

if train_by_cluster:
    from sklearn.model_selection import GroupShuffleSplit
    
    groups = bact_clusters.loc[[meta[0] for meta in rows_metadata], 'Cluster'].values
    metadata_np = np.array(rows_metadata, dtype=object)

    # 2. First split: Train+Val vs Test
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_split_ratio, random_state=42)
    train_full_idx, test_idx = next(gss_test.split(X, y, groups=groups))

    X_train_full, X_test = X[train_full_idx], X[test_idx]
    y_train_full, y_test = y[train_full_idx], y[test_idx]
    groups_train_full = groups[train_full_idx]
    metadata_train_full, metadata_test = metadata_np[train_full_idx], metadata_np[test_idx]

    # 3. Second split: Train vs Val
    # Calculate adjusted ratio for the remaining training set
    adj_val_ratio = val_split_ratio / (1 - test_split_ratio)

    gss_val = GroupShuffleSplit(n_splits=1, test_size=adj_val_ratio, random_state=42)
    train_idx, val_idx = next(gss_val.split(X_train_full, y_train_full, groups=groups_train_full))

    X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
    y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]
    metadata_train, metadata_val = metadata_train_full[train_idx], metadata_train_full[val_idx]
    print(f"Train data clusters: {set(groups_train_full[train_idx])}\nVal data clusters: {set(groups_train_full[val_idx])}\nTest data clusters: {set(groups[test_idx])}\n")
    if logging: 
        print(f'\n{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Train data clusters: {set(groups_train_full[train_idx])}', file=logfile)
        print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Val data clusters: {set(groups_train_full[val_idx])}', file=logfile)
        print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Test data clusters: {set(groups[test_idx])}', file=logfile)
    print(f"Data split into train/val/test with test size {test_split_ratio*100}% and val size {val_split_ratio*100}% (estimated due to use of cluster splitting)")
    if logging: print(f'\n{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Data split into train/val/test with test size {test_split_ratio*100}% and val size {val_split_ratio*100}% (estimated due to use of cluster splitting)', file=logfile)

else:
    from sklearn.model_selection import train_test_split
    # Prepare train / val / test split (use stratify if possible)
    metadata_np = np.array(rows_metadata, dtype=object)
    strat = y if np.unique(y).size > 1 else None
    X_train_full, X_test, y_train_full, y_test, metadata_train_full, metadata_test = train_test_split(
        X, y, metadata_np, test_size=test_split_ratio, random_state=42, stratify=strat
    )
    # now split training part into train + val (use stratify on the training labels if possible)
    strat_train = y_train_full if np.unique(y_train_full).size > 1 else None
    X_train, X_val, y_train, y_val, metadata_train, metadata_val = train_test_split(
        X_train_full, y_train_full, metadata_train_full, test_size=val_split_ratio / (1 - test_split_ratio), random_state=42, stratify=strat_train
    )
    print(f"Data split into train/val/test with test size {test_split_ratio*100}% and val size {val_split_ratio*100}%")
    if logging: print(f'\n{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Data split into train/val/test with test size {test_split_ratio*100}% and val size {val_split_ratio*100}%', file=logfile)

print(f"Train size: {X_train.shape[0]} samples, Val size: {X_val.shape[0]} samples, Test size: {X_test.shape[0]} samples")
if logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Train size: {X_train.shape[0]} samples, Val size: {X_val.shape[0]} samples, Test size: {X_test.shape[0]} samples', file=logfile)
print(f"Fraction of positive interactions in train: {round(sum(y_train)/len(y_train)*100,2)}%\nFraction of positive interactions in val: {round(sum(y_val)/len(y_val)*100,2)}%\nFraction of positive interactions in test: {round(sum(y_test)/len(y_test)*100,2)}%")
if logging: 
    print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Fraction of positive interactions in train: {round(sum(y_train)/len(y_train)*100,2)}%', file=logfile)
    print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Fraction of positive interactions in val: {round(sum(y_val)/len(y_val)*100,2)}%', file=logfile)
    print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Fraction of positive interactions in test: {round(sum(y_test)/len(y_test)*100,2)}%', file=logfile)

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold

# Scale features (fit only on training set)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
X_excluded = scaler.transform(X_excluded) if len(X_excluded) > 0 else X_excluded
y_excluded = y_excluded.astype(int) if len(y_excluded) > 0 else y_excluded

if smote_on:
    from imblearn.over_sampling import SMOTE
    from collections import Counter
    smote = SMOTE(random_state=42)
    print(f"Before SMOTE, training set class distribution: {Counter(y_train)}")
    y_train = y_train.astype(int)  # Ensure y_train is of integer type for SMOTE
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE, training set class distribution: {Counter(y_train)}")

# Saving data splits (e.g. for k nearest neighbor analysis)
if logging:
    np.savez_compressed(outdir+f'torchMLP_splits_run{run}.npz', 
                        X_train=X_train, y_train=y_train, 
                        X_val=X_val, y_val=y_val, 
                        X_test=X_test, y_test=y_test)
    print(f"Data splits saved to {outdir+f'torchMLP_splits_run{run}.npz'}")
    print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Data splits saved to {outdir+f"torchMLP_splits_run{run}.npz"}', file=logfile)

### Simple MLP architecture for binary classification ###
# epoch stats
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

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
            nn.Linear(hidden2, 1)  # logits for BCEWithLogitsLoss
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if logging: print(f'\n{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Using device: {device}', file=logfile)

X_test_t = torch.from_numpy(X_test).float().to(device)
y_test_t = torch.from_numpy(y_test.reshape(-1, 1)).float().to(device)
X_excluded_t = torch.from_numpy(X_excluded).float().to(device) if len(X_excluded) > 0 else X_excluded
y_excluded_t = torch.from_numpy(y_excluded.reshape(-1, 1)).float().to(device) if len(y_excluded) > 0 else y_excluded

test_ds = TensorDataset(X_test_t, y_test_t)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
X_excluded_ds = TensorDataset(X_excluded_t, y_excluded_t) if len(X_excluded) > 0 else None
X_excluded_loader = DataLoader(X_excluded_ds, batch_size=batch_size, shuffle=False) if X_excluded_ds is not None else None
y_excluded_ds = TensorDataset(y_excluded_t) if len(y_excluded) > 0 else None
y_excluded_loader = DataLoader(y_excluded_ds, batch_size=batch_size, shuffle=False) if y_excluded_ds is not None else None

if not cross_validation_on:
    fold = 1 #used for n_epochs multiplier in later code
    # Convert to torch tensors
    X_train_t = torch.from_numpy(X_train).float()
    X_val_t = torch.from_numpy(X_val).float()
    y_train_t = torch.from_numpy(y_train.reshape(-1, 1)).float()
    y_val_t = torch.from_numpy(y_val.reshape(-1, 1)).float()

    # Datasets / loaders
    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MLP(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss() #Loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) #Optimizes weights and biases

    # Training loop
    if logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Starting training with epochs: {n_epochs}...', file=logfile)
    for epoch in range(1, n_epochs + 1):
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
        if logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Epoch {epoch:02d} - train_loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}', file=logfile)

if cross_validation_on:
    # Training loop
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1

    if logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Starting cross-validation with {kf.get_n_splits()} folds...', file=logfile)

    for train_idx, val_idx in kf.split(X_train):
        print(f"Fold {fold}:")
        if logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Fold {fold}...', file=logfile)

        # Split data into training and validation sets for this fold
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Convert to torch tensors
        X_train_t = torch.from_numpy(X_train_fold).float()
        X_val_t = torch.from_numpy(X_val_fold).float()
        y_train_t = torch.from_numpy(y_train_fold.reshape(-1, 1)).float()
        y_val_t = torch.from_numpy(y_val_fold.reshape(-1, 1)).float()

        # Create data loaders
        train_ds = TensorDataset(X_train_t, y_train_t)
        val_ds = TensorDataset(X_val_t, y_val_t)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Initialize model, criterion, and optimizer
        model = MLP(input_dim=X_train.shape[1]).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop for this fold
        for epoch in range(1, n_epochs + 1):
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
            if logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Epoch {epoch:02d} - train_loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}', file=logfile)

        fold += 1
    fold -= 1 # Adjust fold count after loop to reflect actual number of folds completed

# Final evaluation on test set: loss + accuracy
model.eval()
with torch.no_grad():
    test_logits = model(X_test_t.to(device))
    test_loss = criterion(test_logits, y_test_t.to(device)).item()
    test_probs = torch.sigmoid(test_logits)
    test_preds = (test_probs >= 0.5).float()
    test_acc = (test_preds.to(device) == y_test_t).float().mean().item()

print(f"\nFinal test loss: {test_loss:.4f}  test accuracy: {test_acc:.4f}")
if logging: print(f'\n{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Final test loss: {test_loss:.4f}  test accuracy: {test_acc:.4f}', file=logfile)

import matplotlib.pyplot as plt
# Plotting the losses 
fig,ax = plt.subplots(1,1, figsize=(9,5))

ax.plot(range(n_epochs*fold), train_losses, label='Train loss', color='#FF8C00', linewidth=2)
ax.plot(range(n_epochs*fold), val_losses, label='Val loss', color="#D88682", linewidth=2)
ax.legend(loc='lower right')
ax.set_ylabel('Loss')

ax2 = ax.twinx()
ax2.plot(range(n_epochs*fold), val_accuracies, label='Val accuracy', c='g', linestyle='--')
ax2.set_ylabel('Accuracy')
ax2.legend(loc='upper right')

ax.set_xlabel('Epochs')
fig.suptitle(f"Torch MLP Train/Val Loss & Val Accuracy for n{n}, k{k}. Test accuracy: {test_acc:.2f}")
fig.show()

outname = 'torchMLP_acc_loss.png'    
if logging: plt.savefig(outdir+outname)

if logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Accuracy and train figure saved as: {outdir+outname}', file=logfile)


if X_excluded_loader is not None and len(X_excluded) > 0:
    model.eval()
    with torch.no_grad():
        excluded_logits = model(X_excluded_t.to(device))
        #excluded_loss = criterion(excluded_logits, y_excluded_t.to(device)).item()
        excluded_probs = torch.sigmoid(excluded_logits)
        excluded_preds = (excluded_probs >= 0.5).float().item()
        print(excluded_preds)
        #excluded_acc = (excluded_preds.cpu() == y_excluded_t).float().mean().item()

    line = f"\nExcluded pair {exclude_bacts, exclude_phages} - prediction: {excluded_preds:.4f}"
    print(line)
    if logging: print(f'\n{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} {line}', file=logfile)



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

from scipy.special import expit # Equivalent to the Sigmoid function

# Convert logits to probabilities (since you used BCEWithLogitsLoss)
# Logits are the input to the sigmoid function to get probabilities.
probabilities = expit(logits).flatten()

# Convert probabilities to predicted classes (0 or 1)
# Use a threshold of 0.5 (standard for binary classification)
predicted_classes = (probabilities >= 0.5).astype(int)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the confusion matrix
cm = confusion_matrix(true_labels, predicted_classes)

# Plotting the confusion matrix (optional but recommended)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['True 0', 'True 1'])
plt.title(f'Torch nn Confusion Matrix, n{n}, k{k}')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')

outname = 'torchMLP_confusion_matrix.png'
if logging: plt.savefig(outdir+outname)
plt.show()

if logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Confusion matrix figure saved as: {outdir+outname}', file=logfile)
# Interpretation:
# cm[0, 0]: True Negatives (TN)
# cm[0, 1]: False Positives (FP)
# cm[1, 0]: False Negatives (FN)
# cm[1, 1]: True Positives (TP)

from sklearn.metrics import roc_curve, roc_auc_score

# 1. Calculate AUC
# Probabilities are needed here, not the final class prediction
roc_auc = roc_auc_score(true_labels, probabilities)

# 2. Calculate the ROC curve points
# Returns FPR, TPR, and thresholds
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
if logging: plt.savefig(outdir+outname)
plt.show()

if logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} ROC curve figure saved as: {outdir+outname}', file=logfile)

# Calculate Youden's J statistic
J = tpr - fpr

# Find the index of the maximum J value
optimal_idx = np.argmax(J)
optimal_threshold = thresholds[optimal_idx]

# Print the best operating point
print("\n--- Optimal Operating Point (Max Youden's J) ---")
print(f"Optimal Threshold: {optimal_threshold:.4f}")
print(f"Corresponding TPR: {tpr[optimal_idx]:.4f}")
print(f"Corresponding FPR: {fpr[optimal_idx]:.4f}")

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

    print(f"\n--- Top True Positive Entries (Grouped by Cluster) ---")
    print(f"Total True Positives found in unseen clusters: {len(best_tp_df)}")
    print(best_tp_df.head(10))

    # 5. Save and Plot
    if logging:
        best_tp_df.to_csv(outdir+"best_predictions.csv", sep=";")
    
    from analysis import plot_entity_counts, plot_bipartite_network
    plot_entity_counts(best_tp_df, 'Phage_Name', outdir = outdir, logging = logging)
    plot_entity_counts(best_tp_df, 'Bacterium_Name', outdir = outdir, logging = logging)
    plot_bipartite_network(best_tp_df, id_lookup_bact, outdir = outdir, logging = logging, limit=50, conf_threshold=0.5)

except Exception as e:
    print(f"Error during Biparte analysis: {e}")
    traceback.print_exc() # This helps see exactly which line failed

from analysis import f1_analysis

probs = test_probs.flatten().cpu().numpy() if hasattr(test_probs, "cpu") else test_probs.flatten()
y_true = y_test.flatten()  # already numpy

f1_analysis(y_true, probs, logging=logging, outdir = outdir, logfile=logfile)

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
if logging:
    outpath = outdir + "torchMLP_all_pairs_predictions.csv"
    pred_df.to_csv(outpath, index=False)

    print(f"Saved {len(pred_df)} predictions to {outpath}")
    print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Saved {len(pred_df)} predictions to {outpath}', file=logfile)


# create prediction matrix: rows=bacterium, cols=phage, values=prediction
pred_matrix = pred_df.pivot_table(index='bacterium', columns='phage', values='prediction', aggfunc='max')

# normalize column names (strip whitespace) then reorder columns to the requested phage order
pred_matrix = pred_matrix.rename(columns=lambda x: x.strip())

phage_order = [
    "Ymer","Taid","Poppous","Koroua","Abuela","Amona","Sabo","Mimer","Crus",
    "Gander","Guf","Hoejben","Magnum","Vims","Echoes","Galvinrad","Uther",
    "Rip","Rup","Slaad","Pantea","Rap","Zann"
]

bact_order = [

]

# keep only those desired that actually exist, then append any extra columns that were not listed
cols_in_order = [c for c in phage_order if c in pred_matrix.columns]
#rows_in_order = [c for c in pred_matrix.columns if c not in cols_in_order]
final_cols = cols_in_order

# ensure consistent ordering and include any missing rows/cols (fill missing pairs with 0)
pred_matrix = pred_matrix.reindex(index=list(bacteria_names), columns=final_cols, fill_value=0)

# save and show a quick preview
print(pred_matrix.head())
if logging:
    outname = 'torchMLP_prediction_matrix_ordered.csv'
    pred_matrix.to_csv(outdir + outname)
    print(f"Saved ordered prediction matrix to {outdir + outname}")


if logging:
    from io_operations import color_sheet_from_matrix
    color_sheet_from_matrix(
        input_excel=raw_data_path + "phagehost_KU/Hostrange_data_all_crisp_iso.xlsx",
        sheet1_name="sum_hostrange",
        prediction_matrix_df=pred_matrix,
        output_excel=outdir + "hostrange_colored.xlsx", 
        TS=True
    )

# if not sourmash_used:
#     # End run
#     print(f'\n{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} End of run.\nExecuted in {time()-time_start}s', file=logfile)
#     logfile.close()
#     sys.exit(0)

#if sourmash_used: # OHE fails in memory 
from analysis import FeatureImportance
fi = FeatureImportance(model, outdir, metadata_test, id_lookup_bact, host_range_data, raw_data_path, data_prod_path, TS = True, logging = logging)
fi.compute_importance(X_test_t, target=0, delta=True)
fi.plot_attributions()
fi.plot_PCA(color_samples_by="bacteria")
fi.plot_PCA(color_samples_by="phage")
fi.plot_PCA(color_samples_by="interaction")

# Do the attributions concur across samples?
fi.plot_attributions_PCA_clusters() 

# Regain kmer as string, given encoding
try:
    fi.regain_kmers(k=k, sourmash=sourmash_used, top_n=10)
    fi.plot_top_kmers(sourmash=sourmash_used, top_n=10)
except Exception as e:
    print(f"Error during k-mer regaining: {e}")
    traceback.print_exc()

if logging:
    print(f'\n{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} End of run.\n{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Executed in {time()-time_start}s', file=logfile)
    logfile.close()

