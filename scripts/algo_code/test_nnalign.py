import numpy as np
import pandas as pd
import pickle
#from time import time
import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser(description="NNAlign test script")
parser.add_argument("-dir", action="store", dest="directory", type=str, help="Supply the path to the directory containing data for test [always c000]. data must be in the format: (peptide target)")
parser.add_argument("-savepath", action="store", dest="savepath", type=str, 
                    help='Path to save the result. Must be a directory including HLA name. Eg: /data/nnalign_out/A0201')
args = parser.parse_args()
directory = args.directory
models = [file for file in os.listdir(args.savepath) if file.endswith(".pkl")]
savepath = args.savepath

np.random.seed(1) # Set random seed, for reproducibility

def load_blosum(filename):
    """
    Read in BLOSUM values into matrix.
    """
    aa = ['A', 'R', 'N' ,'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
    df = pd.read_csv(filename, sep=r'\s+', comment='#', index_col=0)
    return df.loc[aa, aa]

def load_peptide_target(filename):
    """
    Read amino acid sequence of peptides and
    corresponding log transformed IC50 binding values from text file.
    """
    df = pd.read_csv(filename, sep=r'\s+', usecols=[0,1], names=['peptide','target'])
    return df.sort_values(by='target', ascending=False).reset_index(drop=True)

## Note that encode_peptides is now changed to NOT pad the peptides
## Instead, we save all the non-padded encodings to a list
def encode_peptides(X_in, blosum_file):
    """
    Encode AA seq of peptides using BLOSUM50.
    Returns a list of encoded peptides each of shape (pep_len, n_features)
    """
    blosum = load_blosum(blosum_file)
    
    batch_size = len(X_in)
    n_features = len(blosum)

    encoded_peptides = []

    for peptide_index, row in X_in.iterrows():

        pep_len = row.len
        X_out = np.zeros((pep_len, n_features))

        for aa_index in range(pep_len):
            X_out[aa_index, :] = blosum[ row.peptide[aa_index] ].values

        encoded_peptides.append(X_out)
            
    return encoded_peptides, np.expand_dims(X_in.target.values,1)

def plot_losses(train_losses, valid_losses, n_epochs):
    # Plotting the losses 
    fig,ax = plt.subplots(1,1, figsize=(9,5))
    ax.plot(range(n_epochs), train_losses, label='Train loss', c='b')
    ax.plot(range(n_epochs), valid_losses, label='Valid loss', c='m')
    ax.legend()
    fig.show()

def xavier_initialization_normal(input_dim, output_dim):
    shape = (input_dim, output_dim)
    stddev = np.sqrt(2 / (input_dim + output_dim))
    return np.random.normal(0, stddev, size=shape) * 0.1

def load_nnalign_model(filepath, model=None):
    with open(filepath, 'rb') as f:
        loaded_dict = pickle.load(f)
    if model is None:
            model = NNAlignFFNN(loaded_dict['input_size'], loaded_dict['hidden_size'], loaded_dict['output_size'], loaded_dict['motif_len'],loaded_dict['encoding_dim'])
    assert (model.W1.shape[0]==loaded_dict['input_size'] and model.W1.shape[1]==loaded_dict['hidden_size'] and model.W2.shape[1]==loaded_dict['output_size']), \
        f"Model and loaded weights size mismatch!. Provided model has weight of dimensions {model.W1.shape, model.W2.shape} ; Loaded weights have shape {loaded_dict['W1'].shape, loaded_dict['W2'].shape}"

    model.W1 = loaded_dict['W1']
    model.b1 = loaded_dict['b1']
    model.W2 = loaded_dict['W2']
    model.b2 = loaded_dict['b2']
    print(f"Model loaded successfully from {filepath}\nwith weights [ W1, W2 ] dimensions : {model.W1.shape, model.W2.shape}")
    return model

def weighted_average(p1, p2, p3, p4):
    """
    Takes 4 prediction values and returns their weighted average score.
    Weighted average is based on train_valid model score from train_stdout.log files
    """

blosum_file = f'../BLOSUM50'

class NNAlignFFNN:
    def __init__(self, input_size, hidden_size, output_size, motif_len, encoding_dim, initialization_function=xavier_initialization_normal):
        # Initialize weights and biases with small random values
        self.W1 = initialization_function(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = initialization_function(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)
        self.motif_len = motif_len
        self.encoding_dim = encoding_dim
        print(self.W1.shape, self.b1.shape, self.W2.shape, self.b2.shape, motif_len, encoding_dim)
    
    def relu(self, x):
        return np.maximum(0, x)

    # This version of sigmoid here is NOT numerically stable.
    # We need to split the cases where the input is positive or negative
    # because np.exp(-x) for something negative will quickly overflow if x is a large negative number
    # def sigmoid(self, x):
    #     return 1 / (1 + np.exp(-x))

    def sigmoid(self, x): 
        # This is equivalent to : 
        # if x>=0, then compute (1/(1+np.exp(-x)))
        # else: compute (np.exp(x)/(1+np.exp(x))))
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), 
                        np.exp(x) / (1 + np.exp(x)))

    def forward_inp(self, x):   
        """
        zi denotes the output of a hidden layer i
        ai denotes the output of an activation function at layer i
        (activations are relu, sigmoid, tanh, etc.)
        """

        # First layer
        z1 = np.dot(x, self.W1) + self.b1  # np.dot does for hidden layer node j, z_j = sum(x_i*w_ij for i in input_size)
        a1 = self.relu(z1)

        # Output layer
        z2 = np.dot(a1, self.W2) + self.b2 # np.dot does for output node o, z_o = sum(a1_j*w_jo for j in hidden_size)
        a2 = self.sigmoid(z2)

        # Return all the intermediate outputs as well because we need them for backpropagation (see slides)
        return z1, a1, z2, a2

    def forward(self, x):
        """
        x is a list of non-padded input arrays 
        """

        x_fwd_all = []
        bounds = []  # list of tuples: holds the candidate forward index range per peptide
        x_templates = [] # list of arrays: holds the candidate binding core encodings per peptide

        # First collect all input encodings to try 
        for i in range(len(x)):
            pep_len = x[i].shape[0]

            # Peptide is shorter than motif length. Insertion
            if pep_len < self.motif_len:
                len_ins = self.motif_len - pep_len
                num_ins = pep_len + 1 # Number of insertion positions to try

                x_fwd = np.zeros((num_ins, self.motif_len, self.encoding_dim))

                # Try all insertion positions
                for idx, j in enumerate(range(num_ins)):

                    # Encode the binding core
                    x_fwd[idx][:j] = x[i][:j]                             # Up to insertion <--- TO FILL IN 
                    x_fwd[idx][j:j + len_ins] = 0.0                      # Insertion wildcards
                    x_fwd[idx][j + len_ins:] = x[i][j:]                   # After insertion <--- TO FILL IN 
                
                bounds.append((len(x_fwd_all), len(x_fwd_all) + num_ins))
                x_templates.append(x_fwd)
                x_fwd_all.extend(x_fwd)
        
            else: 
                del_len = pep_len - self.motif_len                       # potential deletion length, if any
                num_del = pep_len - del_len - 1 if del_len > 0 else 0    # Number of deletion positions to try, if any
                num_offsets = pep_len - self.motif_len + 1               # Number of ungapped core offsets to try (position start of core if no gap)

                x_fwd = np.zeros((num_offsets + num_del, self.motif_len, self.encoding_dim))

                # Try all continuous binding cores within the peptide, without deletions
                for idx, j in enumerate(range(num_offsets)):

                    # Encode the binding core
                    x_fwd[idx] = x[i][j:j+self.motif_len]                          # <--- TO FILL IN

                if del_len > 0:
                    # Try all deletion positions starting after the first residue and ending before the last residue
                    for idx, j in enumerate(range(1, pep_len - del_len)):

                        # Encode the binding core
                        x_fwd[num_offsets + idx][:j] = x[i][:j]          # Up to deletion
                        x_fwd[num_offsets + idx][j:] = x[i][j+del_len:]  # After deletion <--- TO FILL IN 
                    
                bounds.append((len(x_fwd_all), len(x_fwd_all) + num_offsets + num_del))
                x_templates.append(x_fwd)
                x_fwd_all.extend(x_fwd)

        # Stack and forward once - more efficient than individually forwarding potential encodings for each peptide
        x_fwd_all = np.stack(x_fwd_all)  # shape (total_candidates, motif_len, encoding_dim)
        z1_all, a1_all, z2_all, a2_all = self.forward_inp(x_fwd_all.reshape(x_fwd_all.shape[0], -1)) # reshape to flatten the last two dimensions

        # Collect best candidates per input
        x_all, z1, a1, z2, a2 = [], [], [], [], []

        for i, (start, end) in enumerate(bounds):
            best_idx = np.argmax(a2_all[start:end])
            x_all.append(x_templates[i][best_idx]) # the optimal peptide encodings are used during backpropagation
            z1.append(z1_all[start + best_idx])
            a1.append(a1_all[start + best_idx])
            z2.append(z2_all[start + best_idx])
            a2.append(a2_all[start + best_idx])

        return np.array(z1), np.array(a1), np.array(z2), np.array(a2), np.array(x_all)

### Cross-validation of files ###
#Extract files in dir
filesindir = os.listdir(directory)
test_file = [f"{directory}/{file}" for file in filesindir if file.split("/")[-1].startswith("c000")] #remove test data
predictions = []

# Loading the test peptides.
test_raw = load_peptide_target(test_file[0])
test_raw['len'] = test_raw['peptide'].apply(len)
x_test_, y_test_ = encode_peptides(test_raw, blosum_file)

### Extract model predictions ###
for m in models:
    # Reload the model and evaluate it
    reloaded_network = load_nnalign_model(f'{savepath+"/"+m}')

    BINDER_THRESHOLD=0.426
    # Thresholding the targets
    y_test_thresholded = (y_test_>=BINDER_THRESHOLD).astype(int)

    _, _, _, test_predictions_scores, _ = reloaded_network.forward(x_test_)

    test_raw["predictions"] = test_predictions_scores

    # Saving the predictions
    test_raw['predictions'] = test_predictions_scores
    predictions.append(test_raw[['peptide','predictions','target']].to_numpy()) #save all predictions in one list

### Find weighted average of prediction values ###
combined_pred_array = []
combined_preds = []
for row_idx in range(predictions[0].shape[0]):
    values_for_current_row = []
    for pred_array in predictions:
        values_for_current_row.append(float(pred_array[row_idx, 1]))
    average = np.mean(values_for_current_row) #apply weighted average
    combined_preds.append(average) #save for plots
    combined_pred_array.append([predictions[0][row_idx, 0], average, predictions[0][row_idx, 2]]) #predictions col 0 and 2 are the same

#Write combined weighted average preds.
with open(savepath+"/"+f"{m.split('_')[0]+'_'+'_'.join(m.split('_')[2:])}_predictions.txt", "w") as pred_out_file:
    for pep_line in combined_pred_array:
        pred_out_file.write(",".join(map(str, pep_line))+"\n")

### Save results as plot ###
from sklearn.metrics import roc_auc_score, roc_curve
test_auc = roc_auc_score(y_test_thresholded.squeeze(), combined_preds)
test_fpr, test_tpr, _ = roc_curve(y_test_thresholded.squeeze(), combined_preds)

f,a = plt.subplots(1,1 , figsize=(9,9))
a.set_title(f"NNAlign performance on allele: {savepath.split('/')[-1]}")
a.plot([0,1],[0,1], ls=':', lw=0.5, label='Random prediction: AUC=0.500', c='k')
a.plot(test_fpr, test_tpr, ls='--', lw=1, label=f'Neural Network: AUC={test_auc:.3f}', c='b')
a.legend()
plt.savefig(savepath+"/"+"test_NNAlign.out.jpg")
