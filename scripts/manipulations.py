########################
### Manipulations.py ###
########################
# Contains functions for manipulating dataframes and other data structures
# No plot or analysis functions should be here
# only functions that transform data from one format to another
# or perform operations on dataframes, lists, etc.


##### Imports -----------
import os, sys
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
import numpy as np
import sourmash


##### Paths -------------
raw_data_path = "../raw_data/"
data_prod_path = "../data_prod/"


##### Functions ---------
def fasta_to_kmerdf(fasta, k=8, quiet=False, sparse=False, relative=True) -> pd.DataFrame:
    """
    Convert a fasta file of sequences to a k-mer frequency DataFrame.

    Args:
        *fasta* (str): List of sequences in fasta format.
        *k* (int): Length of the k-mers. Default is 8.
        *quiet* (bool): If True, suppress progress output. Default is False.
        *sparse* (bool): If True, return a sparse DataFrame. Default is False.
        *relative* (bool): If True, return relative frequencies instead of counts. Default is True.
    
    Returns:
        *pd.DataFrame*: DataFrame with k-mer frequencies for each sequence. sparse or non-sparse depending on sparse arg.
    """
    if type(fasta) == str:  # If a file path is provided, read the fasta file
        try:
            records = list(SeqIO.parse(fasta, "fasta"))
        except FileNotFoundError as e:
            print(f"Error: {e}. Please check the file path.")
            return pd.DataFrame()
    elif type(fasta) == list:  # If a list of filenames
        records = []
        for file in fasta:
            try:
                records.extend(list(SeqIO.parse(file, "fasta")))
            except FileNotFoundError as e:
                print(f"Error: {e}. Please check the file path.")
                continue
    else:
        print("Error: fasta argument must be a file path or list of file paths.")
        return pd.DataFrame()
    if not quiet: print(f"Total sequences to process: {len(records)}")
    
    kmer_list = []
    seq_id_list = []
    if not quiet: print(f"Converting {len(records)} sequences to {k}-mer frequency DataFrame..." )

    for record in tqdm(records, desc="Processing sequences", unit="seq"):
        seq = str(record.seq)
        seq_id = record.id
        kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
        if relative:
            kmer_counts = pd.Series(kmers).value_counts(normalize=True) 
        else:
            kmer_counts = pd.Series(kmers).value_counts() 
        kmer_list.append(kmer_counts)
        seq_id_list.append(seq_id)
    
    if not quiet: print("Combining k-mer counts into DataFrame...")
    if sparse:
        kmer_df = pd.DataFrame.sparse.from_spmatrix(pd.DataFrame(kmer_list).sparse.to_coo()).fillna(0)
    else: 
        kmer_df = pd.DataFrame(kmer_list).fillna(0)

    kmer_df.index = seq_id_list
    if not quiet: print(f"Generated k-mer DataFrame with shape: {kmer_df.shape}")
    return kmer_df

def binarize_host_range(host_range_dict, TS = False, continous = True, acceptive = False) -> dict:
    """
    Convert the values of a dictionary made of nan and float values, to numericalize and normalize.
    Normalize by taking the log first, them min-max normalize. Log first saves computation.

    Args:
        **host_range_dict** (dict): nested dictionary with strains as outer keys, phage as inner keys and host range values as values.
        **TS** (bool): Troubleshooting flag for verbose output.
        **continous** (bool): If continous is true, return normalized values between 0 and 1 suitable for a Regression model. Else return binary values (0 or 1) suitable for a Classification model.
        **acceptive** (bool): If true, any non-zero value is considered as 1 in binary mode.
    Returns:
        **host_range_norm** (dict): nested dictionary with strains as outer keys, phage as inner keys and normalized host range values as values.
        
    Normalization formula:
    $$\text{Normalized}_x = \frac{\log(1 + x)}{\log(1 + \text{highest\_val})}$$
    """
    ### Numericalize 
    highest_val = 0
    host_range_bin = {}

    if continous:
        if TS: print("\n--- Normalized Dictionary ---")
        for bact, phage_d in host_range_dict.items():
            binary_dict = {}
            if TS: print(f"\nfor bact: {bact}")
            for host, val in phage_d.items():
                if TS: 
                    print(f"for phage: {host}, {val}")
                try: 
                    val = float(val)
                except: #val is non-numeric
                    if TS: print("val to float failed")
                    binary_dict[host] = 0
                    continue

                if val == 0 or pd.isna(val):
                    binary_dict[host] = 0 
                else:
                    binary_dict[host] = val
                    if val > highest_val:
                        highest_val = val

            host_range_bin[bact] = binary_dict
        
        ### Normalize
        denominator = np.log(1 + highest_val)
        host_range_norm = {}

        # Iterate through the dictionary and normalize each list
        for bact_strain, phage_dict in host_range_bin.items():
            if TS: print(f"Handling bact: {bact_strain}")
            normalized_dict = {}
            for list_name, values in phage_dict.items():
                np_values = np.array(values)
                log_transformed_data = np.log(1 + np_values) # Apply the log transformation and normalize
                normalized_values = log_transformed_data / denominator
                normalized_dict[list_name] = normalized_values
            host_range_norm[bact_strain] = normalized_dict

        if TS:
            print("\n--- Normalized Dictionary ---")
            for bact_strain, norm_dict in host_range_norm.items():
                print(f"bact_strain: {bact_strain}")
                for list_name, normalized_values in norm_dict.items():
                    print(f"**{list_name}**:")
                    print(np.round(normalized_values, 4)) # Print the array rounded for better readability
        
        return host_range_norm   
    
    else: #for binary data
        if TS: print("\n--- Normalized Dictionary ---")
        for bact, phage_d in host_range_dict.items():
            binary_dict = {}
            if TS: print(f"\nfor bact: {bact}")
            for host, val in phage_d.items():
                if TS: 
                    print(f"for phage: {host}, {val}")
                try: 
                    val = float(val)
                except: #val is non-numeric
                    if TS: print("val to float failed")
                    if acceptive:
                        binary_dict[host] = 1
                    else:
                        binary_dict[host] = 0
                    continue

                if val == 0 or pd.isna(val):
                    binary_dict[host] = 0 
                else:
                    binary_dict[host] = 1

            host_range_bin[bact] = binary_dict
        return host_range_bin

def binarize_value(val):
    if isinstance(val, str) or pd.isna(val):
        return 0
    return val

def short_species_name(full_name):
    """
    Shorten species names like: *Pectobacterium brasiliense* --> *P. brasiliense*
    """
    if len(full_name.split(" ")) < 2:
        return full_name
    else:
        return full_name.split(" ")[0][0] + "." + full_name.split(" ")[1]
    
def hostrange_bact(host_range_data, seqID_list, approach="acceptive", threshold=0.5, TS = False) -> dict:
    """
    Used to obtain the host range given bacteria. Can handle multiple bacteria IDs (as mulitple IDs can be from the same family).
    As such, an approach for obtaining proper hostrange must be considered.
    """
    combined_host_range = {}
    # Acceptive approach: if any seqID has a non-zero value for a host, set to 1
    if approach == "acceptive":
        for seqID in seqID_list:
            curr_host_range = binarize_host_range(host_range_data[seqID])
            for host, val in curr_host_range.items():
                if host not in combined_host_range:
                    combined_host_range[host] = val
                else:
                    if not pd.isna(val) and val != 0:
                        combined_host_range[host] = 1
        return combined_host_range
    
    # Count occurrences of non-zero values for each host, if higher than threshold, set to 1
    elif approach == "consensus":
        host_counts = {}
        for seqID in seqID_list:
            curr_host_range = binarize_host_range(host_range_data[seqID])
            for host, val in curr_host_range.items():
                if host not in host_counts:
                    host_counts[host] = 0
                if not pd.isna(val) and val != 0:
                    host_counts[host] += 1
        for host, count in host_counts.items():
            if TS: print(f"Host: {host}, Count: {count}, Total SeqIDs: {len(seqID_list)}, Ratio: {count / len(seqID_list)}")
            if count / len(seqID_list) >= threshold:
                combined_host_range[host] = 1
            else:
                combined_host_range[host] = 0
        return combined_host_range
    
def construct_SM_sketches(fasta, k : int, outdir : str, quiet : bool = False, sourmash_parameters = [50000, 0], include_reverse : bool = False) -> int:
    """
    Construct sourmash sketches given a fasta input.
    
    Args:
        *fasta* (str | list): List of sequences in fasta format.
        *k* (int): Length of the k-mers. 
        *outdir* (str): directory for storing sketches (each signature in its own file)
        *quiet* (bool): If True, suppress progress output. Default is False.
        *sourmash_parameters* (list): specify sourmash.MinHash(n, scaled)
        *include_reverse* (bool): include the reverse strand to sketches
    
    Returns:
        *exit_status* (binary): 0 for success, 1 for failure.
    """
    ### Input Control ###
    if type(outdir) is not str:
        raise ValueError("outdir must be a path")
    
    if not os.path.exists(data_prod_path+"SM_sketches/"):
        try:
            os.makedirs(data_prod_path+"SM_sketches/", exist_ok=True)
            if not quiet: print(f"Created output directory: {data_prod_path}SM_sketches/")
        except OSError as e:
            raise ValueError(f"Could not create outdir {data_prod_path}SM_sketches/: {e}")
    
    # Ensure outdir exists (create if missing)
    if not os.path.exists(data_prod_path+"SM_sketches/"+outdir):
        try:
            os.makedirs(data_prod_path+"SM_sketches/"+outdir, exist_ok=True)
            if not quiet: print(f"Created output directory: {outdir}")
        except OSError as e:
            raise ValueError(f"Could not create outdir {outdir}: {e}")
    elif not os.path.isdir(data_prod_path+"SM_sketches/"+outdir):
        raise ValueError(f"outdir exists but is not a directory: {outdir}")

    outpath = data_prod_path+"SM_sketches/"+outdir

    # Ensuring sourmash parameters are appropriate
    if sourmash_parameters[0] > 0 and sourmash_parameters[1] > 0:
        raise ValueError("One of the sourmash parameters should be 0")

    for p in sourmash_parameters:
        if type(p) is not int:
            raise ValueError("sourmash parameters must be both integers")

    # Handling both cases of fasta input
    if type(fasta) == str:  # If a file path is provided, read the fasta file
        try:
            records = list(SeqIO.parse(fasta, "fasta"))
        except FileNotFoundError as e:
            print(f"Error: {e}. Please check the file path.")
            return 1
    elif type(fasta) == list:  # If a list of filenames
        records = []
        for file in fasta:
            try:
                records.extend(list(SeqIO.parse(file, "fasta")))
            except FileNotFoundError as e:
                print(f"Error: {e}. Please check the file path.")
                continue

    ### Constructing minhashes for all records ###
    if not quiet: print("------- Constructing MinHashes -------")
    minhashes = []
    for rec in tqdm(records, desc="Constructing minhashes for all records", unit="seq"):
        #print("Record:", rec.id, len(rec.seq))
        try:
            mh = sourmash.MinHash(n=sourmash_parameters[0], ksize=k, scaled=sourmash_parameters[1]) #each record gets its own minhash | scaled=1000 to limit 
            for i in range(0, len(rec.seq) - k + 1):
                kmer = str(rec.seq[i:i+k])
                mh.add_sequence(kmer, force=True)
                if include_reverse:
                    mh.add_sequence(kmer[::-1], force=True)
            minhashes.append(mh)
        except:
            raise SystemError("Error in constructing minhashes")
    
    ### Saving sketches ###
    if not quiet: print("------- Saving Sketches -------")
    if "bact" in fasta:
        outfile_prefix = "bact"
    elif "phage" in fasta:
        outfile_prefix = "phage"
    else:
        outfile_prefix = "out"

    for i in range(len(minhashes)):
        try:
            with open(outpath+f"{outfile_prefix}{i}_minhash_37.sig", "wt") as sigfile:
                sig1 = sourmash.SourmashSignature(minhashes[i], name=records[i].id)
                sourmash.save_signatures([sig1], sigfile)
        except:
            raise SystemError(f"Error in saving sourmash sketch for: {records[i].id}")

def hostrange_df_to_dict(host_range_df : pd.DataFrame) -> dict:
    """
    Simple function to return host range dataframe into a dictionary, cleaning it meanwhile

    Args:
        **host_range_df** (pd.DataFrame): input host range dataframe with strains and phage names (strains must not be index, but in col 1)
    
    Returns:
        **host_range_data** (dict): nested dictionary with strains as outer keys, phage as inner keys and host range values as values.

    """

    host_range_data = {}
    for index, row in host_range_df.iterrows():
        cleaned_index = row[1:].index.str.replace(" ", "")
        curr_bact_series = row[1:]
        curr_bact_series.index = cleaned_index
        host_range_data[row['phage']] = curr_bact_series.to_dict()
    return host_range_data

def get_max_dim(mh_dict):
    # Use max() over the lengths of all values
    return max(len(v) for v in mh_dict.values())

def clean_dict_keys(in_dict : dict, sep : str = "_", take : str = "last") -> dict:
    """
    Clean the keys in a dictionary by splitting by sep and taking the last/first val.
    If name can't be split, return name (do nothing)
    """
    out_dict = {}
    for key, val in in_dict.items():
        if sep in key:
            if take == "first":
                out_dict[key.split("_")[0]] = val
            elif take == "last":
                out_dict[key.split("_")[-1]] = val
            else:
                raise ValueError("Can only take the first or the last value")
        else: 
            out_dict[key] = val
    return out_dict

def construct_presence_matrix(phage_dict : dict, bact_dict : dict, TS : bool = False) -> [pd.DataFrame, pd.DataFrame]:
    """
    Construct a presence/absence matrix given a dictionary of phage & bacteria names with its minhashes.
    The matrix will have rows as sequence IDs and columns as minhashes, with 1 indicating presence and 0 absence.
    Both phage and bacteria are given to the function, to ensure that their presence matrices will have the same columns (hashes).

    Args:
        **phage_dict** (dict): dictionary with keys as phage names and values as sourmash.MinHash objects.
        **bact_dict** (dict): dictionary with keys as bacteria strain IDs and values as sourmash.MinHash objects.
        **TS** (bool): Troubleshooting flag for verbose output.
    
    Returns:
        **list of presence_dfs** [pd.DataFrame, pd.DataFrame]: list with phage and bacteria DataFrames with presence/absence matrix. 
    """
    all_hashes = np.unique(np.concatenate(list(phage_dict.values()) + list(bact_dict.values())))
    
    phage_pres_df = pd.DataFrame(0, index=list(phage_dict.keys()), columns=all_hashes, dtype=np.uint8)
    for name, hashes in phage_dict.items(): # Fill presence (set to 1 where the hash exists for that name)
        phage_pres_df.loc[short_species_name(name), hashes] = 1 # assign 1 to the columns corresponding to the hashes for this name
    if TS: print("Phage binary presence matrix shape (rows, cols):", phage_pres_df.shape)

    bact_pres_df = pd.DataFrame(0, index=list(bact_dict.keys()), columns=all_hashes, dtype=np.uint8)
    for name, hashes in bact_dict.items(): # Fill presence (set to 1 where the hash exists for that name)
        bact_pres_df.loc[name, hashes] = 1 # assign 1 to the columns corresponding to the hashes for this name
    if TS: print("Bact binary presence matrix shape (rows, cols):", bact_pres_df.shape)

    return phage_pres_df, bact_pres_df