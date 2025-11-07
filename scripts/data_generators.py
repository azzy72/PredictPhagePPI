##########################
### data_generators.py ###
##########################
# Contains data generator classes for batch training neural networks
import numpy as np
import random
from keras.utils import Sequence


import numpy as np
import random
from keras.utils import Sequence

class MinhashPairGenerator(Sequence):
    """
    Generates batches of paired minhash vectors (X) and interaction scores (y).
    Assumes interaction_scores structure: [bacteria_name][phage_name]
    """
    def __init__(self, list_IDs, phage_minhashes, max_phage_dim, bacteria_minhashes, max_bact_dim, interaction_scores, batch_size=32, shuffle=True, cleanPhage=True):
        self.keys = list_IDs 
        self.phage_mh = phage_minhashes
        self.max_phage_dim = max_phage_dim
        self.bacteria_mh = bacteria_minhashes
        self.max_bacteria_dim = max_bact_dim
        self.scores = interaction_scores
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

        #In case phages are written like "Pectobacterium_phage_Crus"
        if cleanPhage:
            self.clean_phage_names()

    def __len__(self):
        return int(np.floor(len(self.keys) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        batch_keys = self.keys[start_idx:end_idx]
        X, y = self.__data_generation(batch_keys)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.keys)
    
    def clean_phage_names(self):
        """phage names in sketches are extended, return only phage name"""
        self.phage_mh = {key.split("_")[-1]: value for key, value in self.phage_mh.items()}



    def __data_generation(self, batch_keys):
        # Use the consistent global dimensions passed/calculated
        phage_dim = self.max_phage_dim      # Example: 20000
        bacteria_dim = self.max_bacteria_dim # Example: 29810
        total_dim = phage_dim + bacteria_dim # Example: 49810
        
        X = np.empty((len(batch_keys), total_dim))
        y = np.empty((len(batch_keys), 1))

        for i, (b_name, p_name) in enumerate(batch_keys):
            phage_vec = np.array(self.phage_mh[p_name])
            bacteria_vec = np.array(self.bacteria_mh[b_name])
            
            # --- PADDING IMPLEMENTATION ---
            
            # Pad phage vector to max length
            phage_vec_padded = np.pad(
                phage_vec, 
                (0, phage_dim - len(phage_vec)), # Pad at the end
                mode='constant'
            )
            
            # Pad bacteria vector to max length
            bacteria_vec_padded = np.pad(
                bacteria_vec, 
                (0, bacteria_dim - len(bacteria_vec)), 
                mode='constant'
            )
            # --- END PADDING ---

            # Concatenate the padded vectors
            X[i,] = np.concatenate((phage_vec_padded, bacteria_vec_padded))
            y[i] = self.scores[b_name][p_name]

        return X, y
    

class RandomForestIncidenceGenerator(Sequence):
    """
    Generates batches of paired minhash incidence vectors (X) and interaction scores (y).

    Assumes:
    1. interaction_scores structure: interaction_scores[bacteria_name][phage_name]
    2. binary_matrix is the pre-computed N x M incidence matrix (N=entities, M=unique_minhashes)
    3. entity_to_index maps entity names to matrix row indices.
    4. list_IDs contains (bacteria_name, phage_name) pairs representing all training data.
    """
    def __init__(self, list_IDs, binary_matrix, entity_to_index, interaction_scores, batch_size=32, shuffle=True):
        self.list_IDs = list_IDs  # List of (bact_name, phage_name) tuples for training
        self.binary_matrix = binary_matrix
        self.entity_to_index = entity_to_index
        self.scores = interaction_scores
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

        # Determine the feature dimension (M, number of unique minhashes)
        self.minhash_dim = self.binary_matrix.shape[1]
        self.total_dim = 2 * self.minhash_dim # Concatenated vector size

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        """Updates indices after each epoch"""
        if self.shuffle:
            random.shuffle(self.list_IDs)
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        # Get list of IDs for the current batch
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        batch_IDs = self.list_IDs[start_idx:end_idx]
        
        X, y = self.__data_generation(batch_IDs)
        return X, y

    def __data_generation(self, batch_IDs):
        """Generates data containing batch_size samples"""
        
        # X will have shape (batch_size, 2 * minhash_dim)
        X = np.empty((len(batch_IDs), self.total_dim), dtype=self.binary_matrix.dtype)
        # y will have shape (batch_size, 1)
        y = np.empty((len(batch_IDs), 1), dtype=np.float32)

        # Generate data
        for i, (b_name, p_name) in enumerate(batch_IDs):
            
            # --- 1. Fetch the Indices ---
            bact_idx = self.entity_to_index[b_name]
            phage_idx = self.entity_to_index[p_name]
            
            # --- 2. Fetch the Binary Vectors ---
            # These are already 0s/1s and have the correct fixed dimension (M)
            bact_features = self.binary_matrix[bact_idx, :]
            phage_features = self.binary_matrix[phage_idx, :]

            # --- 3. Concatenate Features (X) ---
            # Input to the model: [Bact Features | Phage Features]
            X[i,] = np.concatenate((bact_features, phage_features))
            
            # --- 4. Fetch Target (y) ---
            # Assuming interaction_scores is nested: scores[bacteria][phage]
            y[i] = self.scores[b_name][p_name]

        return X, y