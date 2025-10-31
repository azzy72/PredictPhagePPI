# %% [markdown]
# # Neural Network for Kmers
# Training neural network not on minhashes, but om kmers. Perhaps too large for bacteria genomes, but should be feasible for phage genomes

# %% [markdown]
# ## Prepping data 
# 

# %%
import pandas as pd
from Bio import SeqIO
import numpy as np
import os, sys
from tqdm import tqdm

K = 12
raw_data_path = "../raw_data/"
data_prod_path = "../data_prod/"
scripts_path = "../scripts/"

sys.path.append(scripts_path)
print(sys.path)

# %% [markdown]
# #### Kmer phage genomes

# %%
from manipulations import fasta_to_kmerdf
all_phage_kmers = fasta_to_kmerdf(raw_data_path+"phagehost_KU/phage_cleaned.fasta", k=K)
display(all_phage_kmers)

# %% [markdown]
# #### Hostrange data

# %%
from manipulations import binarize_host_range, short_species_name

import pandas as pd

# Load the host range data from the Excel file
file_path = raw_data_path + "phagehost_KU/Hostrange_data_all_crisp_iso.xlsx"
sheet_name = "sum_hostrange"  # replace with your sheet name
host_range_df = pd.read_excel(
    file_path,
    sheet_name='sum_hostrange',
    header=1).drop(columns=["isolate ID", "Hostrange_analysis", "Phage"])

# Create a lookup dictionary for bacteria species based on Seq ID - dict
bact_lookup = host_range_df[["Seq ID", "Species"]].drop_duplicates(subset=['Seq ID']).set_index('Seq ID').to_dict()['Species']
print("Bacteria lookup dictionary created with", len(bact_lookup), "entries.")
print(bact_lookup)

# Make Seq ID to phage name mapping - pandas df
host_range_df = host_range_df.drop(columns=["Species"]).set_index('Seq ID').rename_axis('phage').reset_index()
display(host_range_df.head())

# Convert the host range data into a dictionary
host_range_data = {}
for index, row in host_range_df.iterrows():
    cleaned_index = row[1:].index.str.replace(" ", "")
    curr_bact_series = row[1:]
    curr_bact_series.index = cleaned_index
    host_range_data[row['phage']] = curr_bact_series.to_dict()

host_range_data["J10_21_reoriented"]

# %%
import shutil
from manipulations import hostrange_bact

parent_out_dir = data_prod_path + f"phage_kmers_{K}_txt/"
if not os.path.exists(parent_out_dir):
    os.makedirs(parent_out_dir)
else:
    shutil.rmtree(parent_out_dir)
    os.makedirs(parent_out_dir)

for selected_bact_species in set(bact_lookup.values()): 
    ### PREPPING HOST RANGE DATA ###
    # Select a specific bacteria species for host range analysis
    print(f"\nProcessing host range data for bacteria species: {selected_bact_species}")

    #obtain all the keys where the value is equal to selected_bact_species
    selected_seqIDs = [key for key, value in bact_lookup.items() if value == selected_bact_species]
    print("Seq IDs for selected species:", selected_seqIDs)

    # Acceptive approach: since all seqIDs for the same species should have similar host ranges, we combine their host range data.
    # if non-zero is found for any seqID, we set it to 1 in the final host range data.

    combined_host_range = hostrange_bact(host_range_data, selected_seqIDs, approach="acceptive")
    print("Combined host range data for selected species:", combined_host_range)

    ### LOADING PHAGE KMERS ###
    phage_data = {}
    print(f"\nLoading kmers from phages")
    for p_index in all_phage_kmers.index:
        phage_name = p_index.split("_")[-1]
        curr_phage_kmers = all_phage_kmers.T[p_index]

        if phage_name in combined_host_range:
            phage_data[phage_name] = curr_phage_kmers.index
        else:
            print(f"Warning: No matching phage name for {phage_name} in host range data. Skipping...")

    ### OUTPUTTING KMER TXT FILES ###
    out_dir = parent_out_dir + f"{str(selected_bact_species).replace(' ', '_')}/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for phage_name in phage_data.keys():
        print(f"Phage: {phage_name}, Number of kmers: {len(phage_data[phage_name])}")
        print(f"Writing {out_dir}+{phage_name}_{short_species_name(selected_bact_species)}...")
        with open(os.path.join(out_dir, f"{phage_name}.txt"), 'w') as f:
            for kmer in phage_data[phage_name]:
                f.write(f"{kmer}\t{combined_host_range[phage_name]}\n")

# %% [markdown]
# ### Make combined txt files per bacteria
# each file will contain which kmers has been in a phage that could infect the bacteria.

# %%
import random
### Concatenate all phage kmer txt files into a single file for easier NN training ###
all_phage_kmers_txt_path = data_prod_path + f"phage_kmers_{K}_combined/"
if not os.path.exists(all_phage_kmers_txt_path):
    os.makedirs(all_phage_kmers_txt_path)

for filename in os.listdir(parent_out_dir):
    if os.path.isdir(os.path.join(parent_out_dir, filename)):
        bact_folder = os.path.join(parent_out_dir, filename)
        combined_output_file = os.path.join(all_phage_kmers_txt_path, f"{filename}_combined.txt")
        with open(combined_output_file, 'w') as outfile:
            for phage_file in os.listdir(bact_folder):
                if phage_file.endswith('.txt'):
                    phage_filepath = os.path.join(bact_folder, phage_file)
                    with open(phage_filepath, 'r') as infile:
                        shutil.copyfileobj(infile, outfile)
        print(f"Combined kmers for {filename} into {combined_output_file}")
        
        print("Shuffling combined file for random distribution...")
        with open(combined_output_file, 'r') as f:
            lines = f.readlines()
        random.shuffle(lines)
        with open(combined_output_file, 'w') as f:
            f.writelines(lines)

# %% [markdown]
# ## Running NN

# %% [markdown]
# Inits

# %%
### Packages -----------------------------
import random, shutil, os, sys
import numpy as np
import pandas as pd
import math
import pickle
import sys
import pdb
import matplotlib.pyplot as plt

### Paths --------------------------------
raw_data_path = "../raw_data/"
data_prod_path = "../data_prod/"
NN_files_path = data_prod_path + "NN_files/"
if not os.path.exists(NN_files_path):
    os.makedirs(NN_files_path)
else:
    shutil.rmtree(NN_files_path)
    os.makedirs(NN_files_path)

### Custom variables ---------------------
K = 12 #kmer size; equal to 6 aa.
selected_bact_species = "Pectobacterium brasiliense" 
full_path = data_prod_path+f"phage_kmers_{K}_combined/{selected_bact_species.replace(' ', '_')}_combined.txt"
if not os.path.exists(full_path):
    raise FileNotFoundError("Combined kmer txt file for the selected bacteria species not found. Please run the data preparation steps first.")

# %% [markdown]
# ### Splitting data
# each folder in phage_minhash_K_txt is a bacteria name, specifying whether the sketches of each underlying phage txt file, can infect it.

# %%
def separate_test(path, train_val_ratio=0.9, test_ratio=0.1, TS = False) -> tuple:
    """
    Separate the input file into training/validation and test sets based on specified ratios.
    Write the separated data to respective train_val and test files.
    Returns the paths to the train_val and test files.
    """
    if abs(train_val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.")

    if type(path) == list:
        lines = []
        for input_file in path:
            if TS: print(f"Separating data from {input_file} into train/val and test sets...")
            try: 
                with open(input_file, 'r') as f:
                    lines.extend(f.readlines())
            except FileNotFoundError as e:
                print("File not found: ", e)
    else:
        if TS: print(f"Separating data from {path} into train/val and test sets...")
        with open(path, 'r') as f:
            lines = f.readlines()

    random.shuffle(lines)

    total_lines = len(lines)
    train_val_end = int(total_lines * train_val_ratio)

    train_data = lines[:train_val_end]
    test_data = lines[train_val_end:]
    
    if TS: 
        print(f"Total lines: {total_lines}, Train/Val lines: {len(train_data)}, Test lines: {len(test_data)}")
        print(f"Ratios - Train/Val: {len(train_data)/total_lines:.2f}, Test: {len(test_data)/total_lines:.2f}", end="\n\n")

    # Write the separated data to respective files to save memory
    train_val_path = NN_files_path + f"{selected_bact_species.replace(' ', '_')}_train_val.txt"
    with open(NN_files_path + f"{selected_bact_species.replace(' ', '_')}_train_val.txt", 'w') as f:
        f.writelines(train_data)
    
    test_path = NN_files_path + f"{selected_bact_species.replace(' ', '_')}_test.txt"
    with open(NN_files_path + f"{selected_bact_species.replace(' ', '_')}_test.txt", 'w') as f:
        f.writelines(test_data)
    
    return train_val_path, test_path

def lines_to_df(lines):
    rows = [l.strip().split('\t') for l in lines if l.strip()]
    df = pd.DataFrame(rows, columns=['kmer', 'label'])
    # set proper dtypes
    df['kmer'] = df['kmer'].astype(str)
    df['label'] = pd.to_numeric(df['label'], errors='coerce').astype(int)
    return df

def k_fold_split(train_val_file, k=5, seed=42, TS = False):
    """
    Perform k-fold cross-validation split on the input file.
    Returns a list of (train_data, val_data) tuples for each fold.
    Does not return test data.
    """    
    if TS: print("Initiating k_fold_split() -----")
    # Read all the shuffled train val data
    with open(train_val_file, 'r') as f:
        lines = f.readlines()

    # Compute fold sizes
    total = len(lines)
    fold_size = total // k
    folds = [lines[i*fold_size:(i+1)*fold_size] for i in range(k)]
    if TS: print(f"total lines {total}, fold_size {fold_size}")

    # Handle remainder lines (if not evenly divisible)
    remainder = lines[k*fold_size:]
    for i, line in enumerate(remainder):
        folds[i % k].append(line)

    # Generate (train, val) pairs
    all_folds = []
    for i in range(k):
        if TS: print(f"\nStarting split {i+1}/{k}...")
        val_lines = folds[i]
        train_lines = [line for j, f in enumerate(folds) if j != i for line in f]

        train_df = lines_to_df(train_lines)
        if TS: 
            print(f"train_df constructed with shape: {train_df.shape} & data types\n{train_df.dtypes}")
            print("snippet of train_df:")
            display(train_df.head())
            
        val_df = lines_to_df(val_lines)
        if TS: 
            print(f"val_df constructed with shape: {val_df.shape} & data types\n{val_df.dtypes}")
            print("snippet of val_df:")
            display(val_df.head())

        all_folds.append((train_df, val_df))
    
    return all_folds

def train_network(net, x_train, y_train, learning_rate):
    """
    Trains the network for a single epoch, running the forward and backward pass, and compute and return the loss.
    """
    # Forward pass
    z1, a1, z2, a2  = net.forward(x_train)
    # backward pass
    backward(net, x_train, y_train, z1, a1, z2, a2, learning_rate)
    loss = np.mean((a2 - y_train) ** 2)
    return loss
        
def eval_network(net, x_valid, y_valid):
    """
    Evaluates the network ; Note that we do not update weights (no backward pass)
    """
    z1, a1, z2, a2 = net.forward(x_valid)
    loss = np.mean((a2-y_valid)**2)
    return loss

# Model saving and loading functions
def save_ffnn_model(filepath, model):
    if not filepath.endswith('.pkl'):
        filepath = filepath+'.pkl'
    with open(filepath, 'wb') as f:
        dict_to_save = {'input_size': model.W1.shape[0], 'hidden_size':model.W1.shape[1], 'output_size':model.W2.shape[1],
                        'W1': model.W1, 'b1':model.b1, 'W2':model.W2, 'b2':model.b2}
        pickle.dump(dict_to_save, f)
        print(f'Saved FFNN model at {filepath}')


def load_ffnn_model(filepath, model=None):

    with open(filepath, 'rb') as f:
        loaded_dict = pickle.load(f)
    if model is None:
            model = SimpleFFNN(loaded_dict['input_size'], loaded_dict['hidden_size'], loaded_dict['output_size'])
    assert (model.W1.shape[0]==loaded_dict['input_size'] and model.W1.shape[1]==loaded_dict['hidden_size'] and model.W2.shape[1]==loaded_dict['output_size']), \
        f"Model and loaded weights size mismatch!. Provided model has weight of dimensions {model.W1.shape, model.W2.shape} ; Loaded weights have shape {loaded_dict['W1'].shape, loaded_dict['W2'].shape}"

    model.W1 = loaded_dict['W1']
    model.b1 = loaded_dict['b1']
    model.W2 = loaded_dict['W2']
    model.b2 = loaded_dict['b2']
    print(f"Model loaded successfully from {filepath}\nwith weights [ W1, W2 ] dimensions : {model.W1.shape, model.W2.shape}")
    return model

def plot_losses(train_losses, valid_losses, n_epochs, title=None):
    # Plotting the losses 
    fig,ax = plt.subplots(1,1, figsize=(9,5))
    ax.plot(range(n_epochs), train_losses, label='Train loss', c='b')
    ax.plot(range(n_epochs), valid_losses, label='Valid loss', c='m')
    ax.legend()
    if title is not None:
        fig.suptitle(title)
    fig.show()


# %%
train_val_path, test_path = separate_test(full_path, train_val_ratio=0.9, test_ratio=0.1, TS=True)
print(f"Training/Validation data saved to: {train_val_path}")
print(f"Test data saved to: {test_path}")

#Retun list of paths were files start with Pecto in data_prod_path+f"phage_minhash_{K}_combined/
# combined_path = data_prod_path+f"phage_minhash_{K}_combined/"
# pecto_files = [combined_path+path for path in os.listdir(combined_path) if path.startswith("Pecto")]
# print(pecto_files)
# train_val_path, test_path = separate_test(pecto_files, train_val_ratio=0.9, test_ratio=0.1, TS=True)
# print(f"Training/Validation data saved to: {train_val_path}")
# print(f"Test data saved to: {test_path}")

# %%
#Purely to define split, no writing
#k_fold_data = k_fold_split(train_val_path, k=5, seed=42)
#for fold_idx, (train_data, val_data) in enumerate(k_fold_data):
    #print(f"Fold {fold_idx+1}: Train data length: {len(train_data)}, Val data length: {len(val_data)}")

# %% [markdown]
# ### Defining NN class

# %%
# Weights initialization function.
def xavier_initialization_normal(input_dim, output_dim):
    shape = (input_dim, output_dim)
    stddev = np.sqrt(2 / (input_dim + output_dim))
    return np.random.normal(0, stddev, size=shape) * 0.1

def random_initialization_normal(input_dim, output_dim):
    return np.random.randn(input_dim, output_dim) * 0.1

# %%
class SimpleFFNN:
    def __init__(self, input_size, hidden_size, output_size, initialization_function=xavier_initialization_normal):
        # Initialize weights and biases with small random values
        # initialization_function(input_dim, output_dim) -> np.array of shape (input_dim, output_dim)
        self.W1 = initialization_function(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = initialization_function(hidden_size, output_size)
        self.b2 = np.zeros(output_size)
        print(f'Input -> Hidden Layer Weight Matrix Shape: {self.W1.shape}',
              f'First Layer Bias Weights Vector Shape: {self.b1.shape}',
              f'Hidden -> Output layer Weight Matrix Shape: {self.W2.shape}',
              f'Second Layer Bias Weights Vector Shape: {self.b2.shape}', sep="\n")
        
    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x): 
        """
        The normal version of sigmoid 1 / (1 + np.exp(-x)) is NOT numerically stable
        Here we split the case into two for positive and negative inputs
        because np.exp(-x) for something negative will quickly overflow if x is a large negative number
        """
        # This is equivalent to : 
        # if x>=0, then compute (1/(1+np.exp(-x)))
        # if x<0: compute (np.exp(x)/(1+np.exp(x))))
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), 
                        np.exp(x) / (1 + np.exp(x)))

    def forward(self, x):
        """
        x:
        zi denotes the output of a hidden layer i
        ai denotes the output of an activation function (non-linearity) at layer i
        (activations are relu, sigmoid, tanh, etc.)
        Use self.function to call a method. for example: self.relu(XX)
        """

        # First layer : Use a relu here for the activation 
        z1 = np.dot(x, self.W1) + self.b1 #XX
        a1 = self.relu(z1) #XX
        
        # Output layer : Use a sigmoid here for the activation
        z2 = np.dot(a1, self.W2) + self.b2 #XX
        a2 = self.sigmoid(z2) #XX
        
        # Return all the intermediate outputs as well because we need them for backpropagation (see slides)
        return z1, a1, z2, a2

def relu_derivative(a):
    return np.where(a > 0, 1, 0)

def sigmoid_derivative(a):
    """
    For this derivative, it is not necessary to find a numerically stable version.
    Just take the base formula and derive it.    
    """
    return np.array(a*(1 - a))


def backward(net, x, y, z1, a1, z2, a2, learning_rate=0.01):
    """
    Function to backpropagate the gradients from the output to update the weights.
    Apply the chain rule and slowly work out the chain derivatives from the output back to the input
    Reminder that np.dot(array_1, array_2) and array.T exists to transpose an array for matrix multiplication
    """
    # This assumes that we are computing a MSE as the loss function.
    # Look at your slides to compute the gradient backpropagation for a mean-squared error using the chain rule.

    # Output layer error ; We used a sigmoid in this layer
    dE_dO = a2 - y
    dO_do = sigmoid_derivative(a2)
    dE_do = dE_dO * dO_do

    #print("dE_do", dE_do.shape)
    ### (REMEMBER for np.dot(A,B) columns of A MUST equal rows in B) ###
    
    # Backpropagate to hidden layer 
    #print("a1", a1.shape)
    dE_dW2 = np.dot(dE_do.T, a1)
    dE_db2 = np.sum(dE_do, axis=0, keepdims=True)
    dE_db2 = dE_db2.squeeze() # Squeeze is needed here to make the dimensions fit
    #print("dE_dW2", dE_dW2.shape)
    #print("dE_db2", dE_dW2.shape)

    # Hidden layer error ; We used a ReLU in this layer!
    # (O − t)⋅ g'(o)⋅wj
    dE_dH = np.dot(dE_do, net.W2.T)
    #print("dE_dH", dE_dH.shape)
    dE_dh = dE_dH * relu_derivative(a1)
    #print("dE_dh", dE_dh.shape)

    # Backpropagate to input layer
    dE_dW1 = np.dot(dE_dh.T, x)
    #print("dE_dh", dE_dh.shape)
    dE_db1 = np.sum(dE_dh, axis=0, keepdims=True) 
    dE_db1 = dE_db1.squeeze() # Squeeze is needed here to make the dimensions fit
    #print("dE_db1", dE_db1.shape)

    # Update weights and biases using gradient descent
    net.W1 -= learning_rate * dE_dW1.T
    #print("W1 shapes:", net.W1.shape, dE_dW1.T.shape)
    net.b1 -= learning_rate * dE_db1.T
    #print("b1 shapes:", net.b1.shape, dE_db1.T.shape)
    net.W2 -= learning_rate * dE_dW2.T
    #print("W2 shapes:", net.W2.shape, dE_dW2.T.shape)
    net.b2 -= learning_rate * dE_db2.T
    #print("b2 shapes:", net.b2.shape, dE_db2.T.shape)

def encode_OHE(X_in, max_pep_len=12):
    """
    One-hot encoding; Each of the four DNA bases (A, C, G, T) is represented by a unique integer mapping:
    A -> 0, C -> 1, G -> 2, T -> 3.
    Returns a tensor of encoded nucleotides of shape (batch_size, max_pep_len, n_features) for a batch of sequences.
    """
    mapping = {
        'A': 0,
        'C': 1,
        'G': 2,
        'T': 3
    }
    
    batch_size = len(X_in)
    n_features = len(mapping)
    
    X_out = np.zeros((batch_size, max_pep_len, n_features), dtype=np.int8)
    
    for nucleotides_index, row in X_in.iterrows():
        for aa_index in range(len(row.kmer)):
            X_out[nucleotides_index, aa_index] = mapping[row.kmer[aa_index]]
            
    return X_out, np.expand_dims(X_in.label.values,1)

# %% [markdown]
# ### Train & Validate model
# an encoder function is needed to parse my train and val data, in order to perform 2 crucial steps:
# 1. Fixing Sequence Length (Padding and Truncation)
# 2. Creating the 3D Tensor Structure; $$(\text{Batch Size}, \text{Sequence Length}, \text{Feature Dimensions})$$

# %% [markdown]
# #### Encode data for model

# %%
k_fold_data = k_fold_split(train_val_path, k=5, seed=42, TS=True)

# %%
#Purely to define split, no writing
for fold_idx, (train_data, val_data) in enumerate(k_fold_data):
    print(f"Fold {fold_idx+1}: Train data length: {len(train_data)}, Val data length: {len(val_data)}")
    display(train_data.groupby("label").agg(count=('kmer','count')))

    #Encoding data
    x_train_, y_train_ = encode_OHE(train_data)
    x_valid_, y_valid_ = encode_OHE(val_data)
    print(x_train_.shape)
    print(x_train_)

    #Initializing model
    input_size = x_train_.shape[1] # also known as "n_features"
    # Model and training hyperparameters
    learning_rate = 0.0001
    hidden_units = 50
    n_epochs = 500
    output_size = 1
    # Creating a model instance 
    # You can use either `xavier_initialization_normal` or `random_initialization_normal`
    # for the initialization_function argument of the class
    network = SimpleFFNN(input_size, hidden_units, output_size)#, 
                        #initialization_function=xavier_initialization_normal)

    # Training loops
    train_losses = []
    valid_losses = []

    # Run n_epochs of training
    for epoch in range(n_epochs):
        train_loss = train_network(network, x_train_, y_train_, learning_rate)
        valid_loss = eval_network(network, x_valid_, y_valid_)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        # For the first, every 5% of the epochs and last epoch, we print the loss 
        # to check that the model is properly training. (loss going down)
        if (n_epochs >= 10 and epoch % math.ceil(0.05 * n_epochs) == 0) or epoch == 0 or epoch == n_epochs:
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}\tValid Loss: {valid_loss:.4f}")

    # save model + plot losses (put your own savename to be used for the model and predictions)
    model_savepath = NN_files_path+'some_ffnn_model.pkl' # /path/to/your/stuff/filename.pkl
    save_ffnn_model(model_savepath, model=network)

    # plotting the losses 
    plot_losses(train_losses, valid_losses, n_epochs, title=f"Train & Val loss for {selected_bact_species} with {K}mers")

    break

# %% [markdown]
# ### Testing model

# %%



