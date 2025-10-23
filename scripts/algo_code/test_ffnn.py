import numpy as np
import pandas as pd
import pickle, os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from argparse import ArgumentParser
parser = ArgumentParser(description="FFNN test script")
parser.add_argument("-dir", action="store", dest="directory", type=str, help="Supply the path to the directory containing data for test [always c000]. data must be in the format: (peptide target)")
parser.add_argument("-savepath", action="store", dest="savepath", type=str, 
                    help='Path to save the result. Must be a directory including HLA name. Eg: /data/nnalign_out/A0201')
args = parser.parse_args()
directory = args.directory
models = [file for file in os.listdir(args.savepath) if file.endswith(".pkl")]
savepath = args.savepath

#### FUNCTIONS #####

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

def xavier_initialization_normal(input_dim, output_dim):
    shape = (input_dim, output_dim)
    stddev = np.sqrt(2 / (input_dim + output_dim))
    return np.random.normal(0, stddev, size=shape) * 0.1

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

def load_ffnn_model(filepath, model=None):
    if filepath.endswith("/"): #is a directory
        filepath = [os.path.join(filepath, f) for f in os.listdir(filepath) if f.endswith('.pkl')][0] #obtain the first .pkl file in dir
        #remove file extension .pkl from filepath

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

blosum_file = f'../BLOSUM50'

### Cross-validation of files ###
#Extract files in dir
filesindir = os.listdir(directory)
test_file = [f"{directory}/{file}" for file in filesindir if file.split("/")[-1].startswith("c000")] #remove test data
predictions = []

# Loading the test peptides.
test_raw = load_peptide_target(test_file[0])
max_pep_len = test_raw.peptide.apply(len).max()
x_test_, y_test_ = encode_peptides(test_raw, blosum_file, max_pep_len)
x_test_ = x_test_.reshape(x_test_.shape[0], -1)

### Extract model predictions ###
for m in models:
    # Reload the model and evaluate it
    reloaded_network = load_ffnn_model(f'{savepath+"/"+m}')

    BINDER_THRESHOLD=0.426
    # Thresholding the targets
    y_test_thresholded = (y_test_>=BINDER_THRESHOLD).astype(int)

    _, _, _, test_predictions_scores = reloaded_network.forward(x_test_)

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
with open(savepath+"/"+"FFNN_"+savepath.split("/")[-2]+savepath.split("/")[-1]+"_predictions.txt", "w") as pred_out_file:
    for pep_line in combined_pred_array:
        pred_out_file.write(",".join(map(str, pep_line))+"\n")

### Save results as plot ###
from sklearn.metrics import roc_auc_score, roc_curve
test_auc = roc_auc_score(y_test_thresholded.squeeze(), combined_preds)
test_fpr, test_tpr, _ = roc_curve(y_test_thresholded.squeeze(), combined_preds)

f,a = plt.subplots(1,1 , figsize=(9,9))
a.set_title(f"FFNN performance on allele: {savepath.split('/')[-2]}")
a.plot([0,1],[0,1], ls=':', lw=0.5, label='Random prediction: AUC=0.500', c='k')
a.plot(test_fpr, test_tpr, ls='--', lw=1, label=f'Neural Network: AUC={test_auc:.3f}', c='b')
a.legend()
plt.savefig(savepath+"/"+"test_FFNN.out.jpg")


