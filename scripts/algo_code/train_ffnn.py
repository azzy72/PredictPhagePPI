### IMPORTS ###
import numpy as np
import pandas as pd
import math, os
import pickle
from time import time
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser(description="FFNN train script")
parser.add_argument("-dir", action="store", dest="directory", type=str, help="Supply the path to the directory containing data for train and validation. data must be in the format: (peptide target)")
#parser.add_argument("-use_hobohm1", action="store_true", dest="use_hobohm1", default=False, help="True false use files from hobohm1 (default: False)")
parser.add_argument("-nh", action="store", dest="n_hidden", type=int, default=16, help="Number of hidden units")
parser.add_argument("-ne", action="store", dest="n_epochs", type=int, default=500, help="Number of epochs")
parser.add_argument("-lr", action="store", dest="learning_rate", type=float, default=0.0001, help="Learning rate")
parser.add_argument("-savepath", action="store", dest="savepath", type=str, 
                    help='Path to save the result. Must be a directory including HLA name. Eg: /data/nnalign_out/A0201')
args = parser.parse_args()
directory = args.directory
#hobohm1 = args.use_hobohm1
hidden_size = args.n_hidden
n_epochs = args.n_epochs
learning_rate = args.learning_rate
savepath = args.savepath

# STATIC VARIABLES
blosum_file = '../BLOSUM50'

# Utility functions you will re-use
# Data-related utility functions
def load_blosum(filename):
    """
    Read in BLOSUM values into matrix.
    """
    aa = ['A', 'R', 'N' ,'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
    df = pd.read_csv(filename, sep='\s+', comment='#', index_col=0)
    return df.loc[aa, aa]

def load_peptide_target(filename):
    """
    Read amino acid sequence of peptides and
    corresponding log transformed IC50 binding values from text file.
    """
    df = pd.read_csv(filename, sep='\s+', usecols=[0,1], names=['peptide','target'])
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

def encode_peptides(X_in, blosum_file, max_pep_len=9):
    """
    Encode AA seq of peptides using BLOSUM 50.
    Returns a tensor of encoded peptides of shape (1, max_pep_len, n_features) for a single batch
    """
    blosum = load_blosum(blosum_file)
    
    batch_size = len(X_in)
    n_features = len(blosum)
    
    X_out = np.zeros((batch_size, max_pep_len, n_features), dtype=np.int8)
    
    for peptide_index, row in X_in.iterrows():
        for aa_index in range(len(row.peptide)):
            X_out[peptide_index, aa_index] = blosum[ row.peptide[aa_index] ].values
            
    return X_out, np.expand_dims(X_in.target.values,1)


# Weights initialization function.
# xavier initialization is technically more stable and preferred 
# (See slides)
def xavier_initialization_normal(input_dim, output_dim):
    shape = (input_dim, output_dim)
    stddev = np.sqrt(2 / (input_dim + output_dim))
    return np.random.normal(0, stddev, size=shape) * 0.1

def random_initialization_normal(input_dim, output_dim):
    return np.random.randn(input_dim, output_dim) * 0.1

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

class SimpleFFNN:
    def __init__(self, input_size, hidden_size, output_size, initialization_function=xavier_initialization_normal):
        # Initialize weights and biases with small random values
        # initialization_function(input_dim, output_dim) -> np.array of shape (input_dim, output_dim)
        self.W1 = initialization_function(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = initialization_function(hidden_size, output_size)
        self.b2 = np.zeros(output_size)
        
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

### Cross-validation of files ###
#Extract files in dir
filesindir = os.listdir(directory)
filesindir = [f"{directory}/{file}" for file in filesindir if not file.split("/")[-1].startswith("c000")] #remove test data
subset_files = []

#Flag for use of hobohm; decides file ending - hobohm files ends with hobohm
#if hobohm1: subset_files = [file for file in filesindir if file.endswith("_hobohm1")]
#else: subset_files = [file for file in filesindir if file.endswith("_clean")]
subset_files = [file for file in filesindir if "_clean" in file]

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

    train_raw['len']=train_raw['peptide'].apply(len)
    max_pep_len = train_raw.peptide.apply(len).max()
    x_train_, y_train_ = encode_peptides(train_raw, blosum_file, max_pep_len)
    x_valid_, y_valid_ = encode_peptides(valid_raw, blosum_file, max_pep_len)
    # We now have matrices of shape (N_datapoints, max_pep_len, n_features)

    # Reshaping the matrices so they're flat because feed-forward networks are "one-dimensional"
    x_train_ = x_train_.reshape(x_train_.shape[0], -1)
    x_valid_ = x_valid_.reshape(x_valid_.shape[0], -1)
    # Define sizes
    input_size = x_train_.shape[1] # also known as "n_features"
    # Model and training hyperparameters
    output_size = 1
    # Creating a model instance 
    # You can use either `xavier_initialization_normal` or `random_initialization_normal`
    # for the initialization_function argument of the class
    network = SimpleFFNN(input_size, hidden_size, output_size)#, 
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
            #print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}\tValid Loss: {valid_loss:.4f}")

    # save model + plot losses (put your own savename to be used for the model and predictions)
    try:
        save_ffnn_model(f"{savepath}/FFNN_model_{validate_file.split('/')[-1]}.pkl", model=network)
        endtime = time() - s
        print(f"Succesfully saved FFNN model: FFNN_model_{validate_file.split('/')[-1]}.pkl - time: {endtime:.2f}\ttrain_loss: {train_loss:.4f}\tvalid_loss: {valid_loss:.4f}\n")

    except:
        print(f"Error in saving FFNN model for valid data: {validate_file.split('/')[-1]}")
    
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




