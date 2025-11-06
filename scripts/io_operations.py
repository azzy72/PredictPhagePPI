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




# def load_minhash_sketch(in_dir : str, out_dir : str, sel_phage : str = None, sel_bact : str = None, sel_bact_type : str = "strains", bact_lookup : dict = None, host_range_df : pd.DataFrame = None, combine_hostrange_approach : str = "acceptive", TS : bool = True): 
#     """
#     Load and rewrite minhash sketches as txt files in a new dir 

#     Args:
#         **in_dir** (str): Full path to minhash sketches (input) directory
#         **out_dir** (str): Full path to txt sketches (output) directory. Parent directory for resulting directories.
#         **sel_phage** (str): Phage name to select for in host range analysis. Mutually exclusive with *sel_bact*
#         **sel_bact** (str | list): Select a specific bacteria species for host range analysis. Mutually exclusive with *sel_phage*
#         **sel_bact_type** (str) ["strains" / "species"]: whether the sel_bact variable is a strain or a species name.
#         **bact_lookup** (dict) [*Optional*]: Bacteria strains (keys) vs specie (values) lookup dict. If none is given, new dict is calculated using call_hostrange_df()
#         **host_range_df** (pd.DataFrame) [*Optional*]: host range dataframe between bacteria strains and phage names. If none is given, new dict is calculated using call_hostrange_df()
#         **combine_hostrange_approach** (str) ["acceptive" / "consensus"]: different approaches for determining combined hostrange (see manipulations.hostrange_bact())
#         **TS** (bool) [*Optional*]: Troubleshoot on or off
    
#     Returns:
#         None
        
#     """
#     ### Input Control ###
#     sel_phage_flag = False
#     sel_bact_flag = False

#     # Making parent out_dir
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)
#     else:
#         shutil.rmtree(out_dir)
#         os.makedirs(out_dir)
    
#     # Obtaining bact_lookup and host_range_df from call_hostrange_df() if not given.
#     if bact_lookup is None or host_range_df is None:
#         res = call_hostrange_df(raw_data_path + "phagehost_KU/Hostrange_data_all_crisp_iso.xlsx")
#         if bact_lookup is None:
#             bact_lookup = res[0]
#         if host_range_df is None:
#             host_range_df = res[1]

#     if sel_bact is not None:
#         if sel_phage is not None:
#             raise ValueError("Choose either sel_phage or sel_bact, not both")
#         # Handling sel_bact string case
#         if type(sel_bact) == str:
#             sel_bact = [sel_bact] #place into list

#         # Handling sel_bact_type and filtering bact_lookup based on type given (strains / species)
#         sel_bact_type = sel_bact_type.lower()     
#         if sel_bact_type == "strains":
#             bact_lookup = {key: value for key, value in bact_lookup.items() if key in sel_bact}
#         elif sel_bact_type == "species":
#             bact_lookup = {key: value for key, value in bact_lookup.items() if value in sel_bact}
#         else:
#             raise ValueError("sel_bact_type must be either: [strains / species]")

#         sel_bact_flag = True
    
#     elif sel_phage is not None:
#         # Handling sel_phage string case
#         if type(sel_phage) == str:
#             sel_phage = [sel_phage] #place into list
        
#         # Filtering host_range_df based on type given 
#         host_range_df = host_range_df[host_range_df.columns.isin(sel_phage)]

#         sel_phage_flag = True
    
#     # Obtain all the strains of bacteria, regardless if its sorted or not
#     selected_seqIDs = bact_lookup.keys()
#     if TS: print("Seq IDs for selected species:", selected_seqIDs)

#     # Handling combine_hostrange_approach values
#     combine_hostrange_approach = combine_hostrange_approach.lower()
#     if combine_hostrange_approach not in ["acceptive", "consensus"]:
#         raise ValueError("combine_hostrange_approach must be either: [acceptive / consensus]")

#     # Converting host_range_df to dictionary
#     host_range_data = hostrange_df_to_dict(host_range_df)


#     ### Finding bacteria specific sketches + combining with hostrange data ###
#     if sel_phage_flag:
#         for selected_phage in host_range_df.columns.to_list(): 
#             ### PREPPING HOST RANGE DATA ###
#             if TS: print(f"\nProcessing host range data for the phage: {selected_phage}")

#             #Extracting combined hostranges using selective approaches
#             combined_host_range = hostrange_bact(host_range_data, selected_seqIDs, approach=combine_hostrange_approach)
#             if TS: print("Combined host range data for selected species:", combined_host_range)

#             ### LOADING MINHASH SKETCHES ###
#             minhash_data = {}
#             if TS: print(f"\nLoading sketches from: {in_dir}")
#             for filename in os.listdir(in_dir):
#                 if filename.endswith(('.sig', '.json')): # sourmash signature files
#                     filepath = os.path.join(in_dir, filename)
#                     if TS: print(f"filepath: {filepath}")
#                     try:
#                         # sourmash.load_signatures returns an iterator
#                         sig = load_one_signature(filepath)
                        
#                         if not sig:
#                             print(f"Warning: No signatures found in {filename}. Skipping.")
#                             continue
                        
#                         phage_name = str(sig)
#                         if phage_name in combined_host_range.keys():
#                             if TS: print(f"Considering phage: {phage_name} from combined_host_range")
#                             # Extract the hash values (sorted for consistency)
#                             hashes = sorted(sig.minhash.hashes.keys())
#                             minhash_data[phage_name] = hashes
#                         else:
#                             print(f"Warning: Sketch for {phage_name} found, but no matching entry in host range data. Skipping.")
#                             print(f"host range: {combined_host_range.keys()}\n")

#                     except Exception as e:
#                         print(f"Error loading sketch file {filename}: {e}. Skipping.")
    
#     elif sel_bact_flag:
#         for selected_bact_species in set(bact_lookup.values()): 
#             ### PREPPING HOST RANGE DATA ###
#             if TS: print(f"\nProcessing host range data for bacteria species: {selected_bact_species}")

#             #Extracting combined hostranges using selective approaches
#             combined_host_range = hostrange_bact(host_range_data, selected_seqIDs, approach=combine_hostrange_approach)
#             if TS: print("Combined host range data for selected species:", combined_host_range)

#             ### LOADING MINHASH SKETCHES ###
#             minhash_data = {}
#             if TS: print(f"\nLoading sketches from: {in_dir}")
#             for filename in os.listdir(in_dir):
#                 if filename.endswith(('.sig', '.json')): # sourmash signature files
#                     filepath = os.path.join(in_dir, filename)
#                     if TS: print(f"filepath: {filepath}")
#                     try:
#                         # sourmash.load_signatures returns an iterator
#                         sig = load_one_signature(filepath)
                        
#                         if not sig:
#                             print(f"Warning: No signatures found in {filename}. Skipping.")
#                             continue
                        
#                         phage_name = str(sig)
#                         if phage_name in combined_host_range.keys():
#                             if TS: print(f"Considering phage: {phage_name} from combined_host_range")
#                             # Extract the hash values (sorted for consistency)
#                             hashes = sorted(sig.minhash.hashes.keys())
#                             minhash_data[phage_name] = hashes
#                         else:
#                             print(f"Warning: Sketch for {phage_name} found, but no matching entry in host range data. Skipping.")
#                             print(f"host range: {combined_host_range.keys()}\n")

#                     except Exception as e:
#                         print(f"Error loading sketch file {filename}: {e}. Skipping.")


#     if TS: print(f"Loaded sketches for {len(minhash_data)} phages.")

#     ### OUTPUTTING MINHASH TXT FILES ###
#     out_dir = out_dir + f"{str(selected_bact_species).replace(' ', '_')}/"
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)

#     for phage_name in minhash_data.keys():
#         if TS: print(f"Phage: {phage_name}, Number of hashes: {len(minhash_data[phage_name])}")
#         if TS: print(f"Writing {out_dir}+{phage_name}_{short_species_name(selected_bact_species)}...")
#         with open(os.path.join(out_dir, f"{phage_name}.txt"), 'w') as f:
#             for hash_value in minhash_data[phage_name]:
#                 f.write(f"{hash_value}\t{combined_host_range[phage_name]}\n")
    
#     if TS: print("Completed!")