########################
### Manipulations.py ###
########################
# Contains functions for manipulating dataframes and other data structures
# No plot or analysis functions should be here
# only functions that transform data from one format to another
# or perform operations on dataframes, lists, etc.


##### Imports -----------
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
import numpy as np


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

def binarize_host_range(host_range_dict):
    """
    Convert the values of a dictionary made of float, to binary values (0 or 1)
    """
    binary_dict = {}
    for host, val in host_range_dict.items():
        if pd.isna(val) or val == 0:
            binary_dict[host] = 0
        else:
            binary_dict[host] = 1
    return binary_dict

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