########################
###    Imports.py    ###
########################
# Contains functions for importing data in a specific format


##### Imports -----------
import os, sys
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
import numpy as np
import shutil
from manipulations import construct_presence_matrix, short_species_name, clean_dict_keys
from manipulations import binarize_host_range, hostrange_df_to_dict, hostrange_bact
from sourmash import load_one_signature
import re

##### Paths -------------
raw_data_path = "../raw_data/"
data_prod_path = "../data_prod/"


def call_hostrange_df(file : str, sheet_name : str = "sum_hostrange", TS : bool = False, sparse : bool = False) -> list:
    """
    Used to retrieve the hostrange data of PFU between bacterias and bacteriophages, as well as bacteria strain lookup dictionary.

    Args:
        *fasta* (str): Full path to hostrange data sheet [excel]
        *sheet_name* (str): Sheet name in excel file
        *TS* (bool): Troubleshoot on or off (like verbose)
        *sparse* (bool): Whether to return a sparse hostrange DataFrame.
    
    Returns:
        *list[0]* (dict): bacteria lookup dictionary with bacteria species and strain names (strain names are unique)
        *list[1]* (pd.DataFrame): hostrange pd.Dataframe with bacteria strain as index and phage names as row columns

    """
    # Load the host range data from the Excel file
    host_range_df = pd.read_excel(
        file,
        sheet_name=sheet_name,
        header=1).drop(columns=["isolate ID", "Hostrange_analysis", "Phage"])

    # Create a lookup dictionary for bacteria species based on Seq ID - dict
    bact_lookup = host_range_df[["Seq ID", "Species"]].drop_duplicates(subset=['Seq ID']).set_index('Seq ID').to_dict()['Species']
    if TS: 
        print("Bacteria lookup dictionary created with", len(bact_lookup), "entries.")
        print(bact_lookup)

    # Make Seq ID to phage name mapping - pandas df
    host_range_df = host_range_df.drop(columns=["Species"]).set_index('Seq ID').rename_axis('phage').reset_index()
    if TS: print(host_range_df.head())
    
    return [bact_lookup, host_range_df]

def load_minhash_sketches(in_dir : str, TS : bool = False, output_as_np : bool = False):
    """
    Load Sourmash Minhash sketches from a directory and concatenate them in a dictionary.

    Args:
        **in_dir** (str): Path to signatures
        **TS** (bool): Troubleshoot on or off
        **output_as_np** (bool): Output minhash vector (values in dict) as np.array rather than the default python list
    
    Returns:
        Dictionary of minhashes, with (extended) phage names / bacteria names as keys and it's minhash composition as a vector [list / np.array]
    
    """
    minhash_data = {}
    for filename in os.listdir(in_dir):
        if filename.endswith(('.sig', '.json')): # sourmash signature files
            filepath = os.path.join(in_dir, filename)
            if TS: print(f"filepath: {filepath}")
            try:
                # sourmash.load_signatures returns an iterator
                sig = load_one_signature(filepath)
                
                if not sig:
                    print(f"Warning: No signatures found in {filename}. Skipping.")
                    continue
                
                name = str(sig)
                if output_as_np:
                    hashes = np.array(sorted(sig.minhash.hashes.keys()))
                else:
                    hashes = sorted(sig.minhash.hashes.keys())
                minhash_data[name] = hashes

            except Exception as e:
                print(f"Error loading sketch file {filename}: {e}. Skipping.")

    return minhash_data

def presence_matrix(phage_minhash_dir : str = None, bact_minhash_dir : str = None, n : int = 0, k : int = 0, reversecomp_data : bool = True, TS : bool = False):
    """
    Create a binary presence matrix from minhash sketches.
    Combines the workflows of loading minhash sketches (load_minhash_sketches()) and generating a binary presence matrix (manipulations.construct_presence_matrix()).
    Returns the binary presence matrix along with the list of unique minhashes and entity-to-index mapping.
    The presence matrix contains both phage and bacteria rows, so check for host interactions for each pairwise mapping.
    If phage_minhash_dir or bact_minhash_dir are None, default paths will be used based on n and k values.

    Args:
        **phage_minhash_dir** (str): Directory containing phage minhash sketches for a specific run (n & kmer size)
        **bact_minhash_dir** (str): Directory containing bacteria minhash sketches for a specific run (n & kmer size)
        **n** (int): Number of minhashes used in the sketches
        **k** (int): Kmer size used in the sketches
        **reversecomp_data** (bool): Whether reverse complements were used in the minhash sketches
        **TS** (bool): Troubleshoot on or off
    
    Returns:
        **presence_matrix** (np.array): Binary presence matrix with shape (num_entities, num_unique_minhashes)\n
        **entity_to_index** (list): Mapping from entity names to their corresponding row indices in the presence matrix\n
        **minhash_to_index** (dict): Mapping from minhashes to their corresponding column indices in the presence matrix\n
        **phage_minhash_data** (dict): Dictionary of phage minhash sketches\n
        **bact_minhash_data** (dict): Dictionary of bacteria minhash sketches
    """
    ### Load minhash sketches
    if phage_minhash_dir is None:
        phage_minhash_dir = data_prod_path+f"SM_sketches/PhageMinhash_n{n}_k{k}/"
    if bact_minhash_dir is None:
        bact_minhash_dir = data_prod_path+f"SM_sketches/BactMinhash_n{n}_k{k}/"
    
    if reversecomp_data:
        phage_minhash_dir = phage_minhash_dir[:-1]+"_rev/"
        bact_minhash_dir = bact_minhash_dir[:-1]+"_rev/"
    
    if TS: 
        print(f"Loading phage minhash sketches from: {phage_minhash_dir}")
        print(f"Loading bacteria minhash sketches from: {bact_minhash_dir}")
    
    try:
        phage_minhash_data = load_minhash_sketches(phage_minhash_dir, 
                                                   TS=False, output_as_np=True)
        bact_minhash_data = load_minhash_sketches(bact_minhash_dir, 
                                                  TS=False, output_as_np=True)
    except Exception as e:
        print(f"Error loading minhash sketches: {e}")
        return None, None, None, None, None
    
    phage_minhash_data = clean_dict_keys(phage_minhash_data)
    
    ### Extract unique minhashes
    unique_minhashes = set() #for both phage and bacteria combined

    for key, val in phage_minhash_data.items():
        for minhash in val:
            unique_minhashes.add(minhash)

    for key, val in bact_minhash_data.items():
        for minhash in val:
            unique_minhashes.add(minhash)

    if TS: print(f"\nUnique minhashes extracted with len: {len(unique_minhashes)}")

    ### Obtaining presence matrix
    # Combine both dictionaries into a single dictionary
    all_entities_minhashes = {**phage_minhash_data, **bact_minhash_data}

    # Get an ordered list of all entity names (will be the row labels)
    entity_names = sorted(list(all_entities_minhashes.keys()))

    # Get an ordered list of unique minhashes (will be the column labels)
    # Sorting is crucial to ensure consistency in the matrix columns
    sorted_minhashes = sorted(list(unique_minhashes))

    # Determine dimensions
    N = len(entity_names)  # Number of rows (entities)
    M = len(sorted_minhashes)  # Number of columns (unique minhashes)

    # Create a dictionary for quick lookup of minhash indices
    minhash_to_index = {minhash: i for i, minhash in enumerate(sorted_minhashes)}
    entity_to_index = {name: i for i, name in enumerate(entity_names)}

    # Initialize the binary matrix with all zeros
    binary_matrix = np.zeros((N, M), dtype=int)

    # Iterate through each entity (row)
    for i, entity_name in enumerate(entity_names):
        # Get the list of minhashes for the current entity
        minhashes_present = all_entities_minhashes[entity_name]

        # Iterate through the minhashes present in the entity
        for minhash in minhashes_present:
            # Get the column index for this minhash
            j = minhash_to_index[minhash]

            # Set the corresponding cell in the matrix to 1
            binary_matrix[i, j] = 1

    ### Proof of concept outputs ###
    if TS:
        print("\nBinary presence matrix created with shape:", binary_matrix.shape)
        print("Sample rows (entities):", entity_names[:5])
        print("Sample columns (minhashes):", sorted_minhashes[:5])
    
    return binary_matrix, entity_to_index, minhash_to_index, phage_minhash_data, bact_minhash_data
