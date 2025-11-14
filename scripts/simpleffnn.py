##########################
###   simpleffnn.py    ###
##########################
# Contains the SimpleFFNN class definition as well as helper functions

### Packages -----------------------------
import random, shutil, os, sys
import numpy as np
import pandas as pd
import math
import pickle
import sys
import matplotlib.pyplot as plt

## Simple Feed-Forward Neural Network ------
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


### Weight Initialization Functions -----
def xavier_initialization_normal(input_dim, output_dim):
    shape = (input_dim, output_dim)
    stddev = np.sqrt(2 / (input_dim + output_dim))
    return np.random.normal(0, stddev, size=shape) * 0.1

def random_initialization_normal(input_dim, output_dim):
    return np.random.randn(input_dim, output_dim) * 0.1

### Handling FFNN functions -------------
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

def encode_new_data(X_in):
    """
    Encode the large integer feature using log transformation and normalization.
    Returns a tuple: (X_out, Y_out)
      X_out: tensor of shape (batch_size, 1)
      Y_out: tensor of shape (batch_size, 1)
    """
    
    # 1. Extract the large integer feature (assuming it's the first column)
    # Convert to float for safe log calculation
    feature_column = X_in.iloc[:, 0].astype(float)
    
    # 2. Log Transformation
    # Use log10 or natural log (np.log) - log10 is often easier to interpret
    # Add a small epsilon or check for zero if your data could contain 0,
    # but based on your example, it's safe to use log10.
    log_transformed = np.log10(feature_column)
    
    # 3. Simple Normalization (Min-Max or Z-score)
    # Using Min-Max scaling for this example: scale to [0, 1]
    min_val = log_transformed.min()
    max_val = log_transformed.max()
    
    # Handle case where min == max to avoid division by zero (e.g., if batch_size=1)
    if max_val == min_val:
        normalized_feature = np.zeros_like(log_transformed)
    else:
        normalized_feature = (log_transformed - min_val) / (max_val - min_val)
        
    # 4. Prepare X_out and Y_out
    # X_out shape: (batch_size, 1) - One feature per sample
    X_out = np.expand_dims(normalized_feature.values, 1)
    
    # Y_out shape: (batch_size, 1) - Target values
    # Assuming the target is the second column (0 or 1)
    Y_out = np.expand_dims(X_in.iloc[:, 1].values, 1)
            
    return X_out.astype(np.float32), Y_out.astype(np.int8)

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