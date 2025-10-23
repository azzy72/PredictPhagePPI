import numpy as np
import pandas as pd
import os
import pickle
from time import time
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser(description="NNAlign train script")
parser.add_argument("-dir", action="store", dest="directory", type=str, help="Supply the path to the directory containing data for train and validation. data must be in the format: (peptide target)")
parser.add_argument("-use_hobohm1", action="store_true", dest="use_hobohm1", default=False, help="True false use files from hobohm1 (default: False)")
parser.add_argument("-len", action="store", dest="motif_len", type=int, help="Length of peptide binding motif (type: int)")
parser.add_argument("-nh", action="store", dest="n_hidden", type=int, default=16, help="Number of hidden units")
parser.add_argument("-lr", action="store", dest="learning_rate", type=float, default=0.0001, help="Learning rate")
parser.add_argument("-ne", action="store", dest="n_epochs", type=int, default=500, help="Number of epochs")
parser.add_argument("-savepath", action="store", dest="savepath", type=str, 
                    help='Path to save the result. Must be a directory including HLA name. Eg: /data/nnalign_out/A0201')
args = parser.parse_args()
directory = args.directory
hobohm1 = args.use_hobohm1
motif_length = args.motif_len
hidden_size = args.n_hidden
n_epochs = args.n_epochs
learning_rate = args.learning_rate
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

def load_peptide_target_multiple_files(file_names):
    """
    Read amino acid sequence of peptides and
    corresponding log transformed IC50 binding values from multiple text files - concatenate them into one df.
    Used on training data sets.
    """
    df_train = pd.concat(
        [pd.read_csv(f, sep=r'\s+', usecols=[0,1], names=['peptide', 'target']) for f in file_names],
        ignore_index=True
    )
    return df_train.sort_values(by='target', ascending=False).reset_index(drop=True)

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

# Misc. functions
def invoke(early_stopping, loss, model, implement=False):
    if implement == False:
        return False
    else:
        early_stopping(loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            return True
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_losses(train_losses, valid_losses, n_epochs):
    # Plotting the losses 
    fig,ax = plt.subplots(1,1, figsize=(9,5))
    ax.plot(range(n_epochs), train_losses, label='Train loss', c='b')
    ax.plot(range(n_epochs), valid_losses, label='Valid loss', c='m')
    ax.legend()
    fig.show()

def relu_derivative(a):
    # to find
    # return XX
    return (a > 0).astype(float)

def sigmoid_derivative(a):
    # to find
    # return XX
    return a * (1 - a)

def backward(net, x, y, z1, a1, z2, a2, learning_rate=0.01):
    """
    Function to backpropagate the gradients from the output to update the weights.
    """
    # This assumes that we are computing a MSE as the loss function.
    # Look at your slides to compute the gradient backpropagation for a mean-squared error using the chain rule.

    # reshape to flatten the last two dimensions
    x = x.reshape(x.shape[0], -1)

    # Calculate loss gradient
    error = a2 - y # Derivative of MSE loss
    d_output = error * sigmoid_derivative(a2)

    # Shape of d_output is (3951, 1) Number of data points. Corresponds to delta in the slides

    # Backpropagate to hidden layer
    # a1.T.shape = (50, 3951)
    # d_output.shape = (3951, 1)
    d_W2 = np.dot(a1.T, d_output) # d_W2.shape = (50, 1). np.dot sums over the data point dimension
    d_b2 = np.sum(d_output, axis=0, keepdims=True) # Sum over the data points
    d_b2 = d_b2.squeeze() # Squeeze to remove the extra dimension

    # d_output.shape = (3951, 1)
    # net.W2.T.shape = (1, 50)
    error_hidden_layer = np.dot(d_output, net.W2.T) # error_hidden_layer.shape = (3951, 50).
    d_hidden_layer = error_hidden_layer * relu_derivative(a1) # This is delta_2 from the slides.

    # Backpropagate to input layer
    d_W1 = np.dot(x.T, d_hidden_layer) # d_W1.shape = (231, 50). np.dot sums over the data point dimension
    d_b1 = np.sum(d_hidden_layer, axis=0, keepdims=True)
    d_b1 = d_b1.squeeze() # Squeeze to remove the extra dimension

    # Update weights and biases using gradient descent
    # net.W1 -= XX
    # net.b1 -= XX
    net.W1 -= learning_rate * d_W1
    net.b1 -= learning_rate * d_b1
    net.W2 -= learning_rate * d_W2
    net.b2 -= learning_rate * d_b2
    
def train_network(net, x_train, y_train, learning_rate):
    """
    Trains the network for a single epoch, running the forward and backward pass, and compute and return the loss.
    """
    # Forward pass
    z1, a1, z2, a2, x  = net.forward(x_train)
    # backward pass
    backward(net, x, y_train, z1, a1, z2, a2, learning_rate)
    loss = np.mean((a2 - y_train) ** 2)
    return loss
        
def eval_network(net, x_valid, y_valid):
    """
    Evaluates the network ; Note that we do not update weights (no backward pass)
    """
    z1, a1, z2, a2, _ = net.forward(x_valid)
    loss = np.mean((a2-y_valid)**2)
    return loss

# Weights initialization function.
# xavier initialization is technically more stable and preferred 
# (See slides)
def xavier_initialization_normal(input_dim, output_dim):
    shape = (input_dim, output_dim)
    stddev = np.sqrt(2 / (input_dim + output_dim))
    return np.random.normal(0, stddev, size=shape) * 0.1

def random_initialization_normal(input_dim, output_dim):
    return np.random.randn(input_dim, output_dim) * 0.1

# Model saving and loading functions
def save_nnalign_model(filepath, model):
    if not filepath.endswith('.pkl'):
        filepath = filepath+'.pkl'
    with open(filepath, 'wb') as f:
        dict_to_save = {'input_size': model.W1.shape[0], 
                        'hidden_size':model.W1.shape[1], 
                        'output_size':model.W2.shape[1], 
                        'motif_len': model.motif_len, 
                        'encoding_dim': model.encoding_dim,
                        'W1': model.W1, 'b1':model.b1, 'W2':model.W2, 'b2':model.b2}
        pickle.dump(dict_to_save, f)
        print(f'Saved NNAlign model at {filepath}')

# datapath for blosum50
blosum_file = f'../BLOSUM50'

### Define NN class ###
class NNAlignFFNN:
    def __init__(self, input_size, hidden_size, output_size, motif_len, encoding_dim, initialization_function=xavier_initialization_normal):
        # Initialize weights and biases with small random values
        self.W1 = initialization_function(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = initialization_function(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)
        self.motif_len = motif_len
        self.encoding_dim = encoding_dim
    
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
filesindir = [f"{directory}/{file}" for file in filesindir if not file.split("/")[-1].startswith("c000")] #remove test data
subset_files = []

#Flag for use of hobohm; decides file ending - hobohm files ends with hobohm
if hobohm1: subset_files = [file for file in filesindir if file.endswith("_hobohm1")]
else: subset_files = [file for file in filesindir if "_clean" in file]

#iterate through c000-004 files; never use c000 - that is used for testing.
log_stats = []
subset_files = sorted(subset_files)

for i in range(len(subset_files)):
    s = time()
    #Creating 1 for validate and 3 for train
    validate_file = subset_files[i]
    train_file = [f for j, f in enumerate(subset_files) if j != i]

    #Create temporary concatenated train file at directory - removed after each iteration
    # Now you can use `train_file` and `validate_file` for processing
    print(f"Fold {i+1}:")
    print(f"  Validate: {validate_file.split('/')[-1]}")
    print(f"  Train: {[file.split('/')[-1] for file in train_file]}")

    # Loading the peptides.
    train_raw = load_peptide_target_multiple_files(train_file)
    valid_raw = load_peptide_target(validate_file)
    # 

    train_raw['len'] = train_raw['peptide'].apply(len)
    valid_raw['len'] = valid_raw['peptide'].apply(len)

    x_train_, y_train_ = encode_peptides(train_raw, blosum_file)
    x_valid_, y_valid_ = encode_peptides(valid_raw, blosum_file)


    # Reshaping the matrices so they're flat because feed-forward networks are "one-dimensional"
    # Define sizes
    input_size = 9 * 21 # also known as "n_features"
    # Model and training hyperparameters
    output_size = 1
    # Creating a model instance 
    # You can use either `xavier_initialization_normal` or `random_initialization_normal`
    # for the initialization_function argument of the class
    network = NNAlignFFNN(input_size, hidden_size, output_size, 9, 21)#, 
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
        #if (n_epochs >= 10 and epoch % math.ceil(0.05 * n_epochs) == 0) or epoch == 0 or epoch == n_epochs:
            #print(f"Epoch {epoch}: \n\tTrain Loss:{train_loss:.4f}\tValid Loss:{valid_loss:.4f}")


    # Put your own savename to be used for the model and predictions
    # saving the model to a file
    # Use the .pkl extension to save python pickled files
    try:
        save_nnalign_model(f"{savepath}/NNAlign_model_{validate_file.split('/')[-1]}.pkl", model=network)
        endtime = time() - s
        print(f"Succesfully saved NNAlign model: NNAlign_model_{validate_file.split('/')[-1]}.pkl - time: {endtime:.2f}\ttrain_loss: {train_loss:.4f}\tvalid_loss: {valid_loss:.4f}\n")

    except:
        print(f"Error in saving NNAlign model for valid data: {validate_file.split('/')[-1]}")
    
    log_stats.append([round(endtime,2), round(train_loss,4), round(valid_loss,4)])

### Validation and Training loss - meaning ###
#High valid and train loss: underfitting
#Low valid and train loss: good fit
#high valid and low train loss: overfitting

### Create ensemble ###
#weighted average on the basis of validation loss
# contribution = pred * ( 1/validation loss / sum(1 / validation loss))
#OR
# model[x]_score = 1 - (valid_loss[x] * (valid_loss[x] - train_loss[x]))
#log_stats now contains a list for each model at each index.
#each index list is a list of float values in the order: time, train_loss, valid_loss

for i, stat in enumerate(log_stats):
    print(f"{subset_files[i]} score:{1-(stat[2]*(stat[2]-stat[1]))}")




