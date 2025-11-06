##########################
### data_generators.py ###
##########################
# Contains data generator classes for batch training neural networks
import numpy as np
import random
from keras.utils import Sequence


class MinhashPairGenerator(Sequence):
    """
    Generates batches of paired minhash vectors (X) and hostrange scores (y).
    """

    def __init__(self, list_IDs, phage_minhashes, bacteria_minhashes, interaction_scores, batch_size=32, shuffle=True):
        # 1. Store the list of keys (IDs) for this specific generator instance
        self.keys = list_IDs 
        
        self.phage_mh = phage_minhashes
        self.bacteria_mh = bacteria_minhashes
        self.scores = interaction_scores
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # 2. Call on_epoch_end to perform initial shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.keys) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        
        # Determine the key indices for this batch
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        
        # Get the list of (phage, bacteria) tuples for this batch
        batch_keys = self.keys[start_idx:end_idx]
        
        # Generate data
        X, y = self.__data_generation(batch_keys)
        
        return X, y

    def on_epoch_end(self):
        """Updates indices after each epoch"""
        if self.shuffle:
            random.shuffle(self.keys)

    def __data_generation(self, batch_keys):
        """
        Generates data containing batch_size samples.
        X is the concatenated minhash vector (phage_mh + bacteria_mh).
        y is the hostrange score.
        """
        
        # Get the dimensions of the input vectors
        phage_dim = next(iter(self.phage_mh.values())).shape[0]
        bacteria_dim = next(iter(self.bacteria_mh.values())).shape[0]
        total_dim = phage_dim + bacteria_dim
        
        # Initialize numpy arrays for the batch
        # X: (batch_size, total_dim), y: (batch_size, 1)
        X = np.empty((len(batch_keys), total_dim))
        y = np.empty((len(batch_keys), 1))

        # Iterate through the (phage, bacteria) pairs
        for i, (p_name, b_name) in enumerate(batch_keys):
            
            # Fetch the two minhash vectors
            phage_vec = self.phage_mh[p_name]
            bacteria_vec = self.bacteria_mh[b_name]
            
            # 1. Concatenate them to create the feature vector X
            X[i,] = np.concatenate((phage_vec, bacteria_vec))
            
            # 2. Fetch the label (hostrange score)
            y[i] = self.scores[p_name][b_name]

        return X, y