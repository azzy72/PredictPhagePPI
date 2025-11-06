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
    
    