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
from manipulations import binarize_host_range, short_species_name, hostrange_df_to_dict, hostrange_bact
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
        Dictionary of minhashes, with (extended) phage names as keys and it's minhash composition as a vector [list / np.array]
    
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
