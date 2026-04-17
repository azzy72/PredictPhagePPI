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
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from paths import raw_data_path, data_prod_path, path_to_nn_runs

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
        return full_name.split(" ")[0][0] + ". " + full_name.split(" ")[1]
    
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
    
def construct_SM_sketches(raw_in : str, k : int, outdir : str, quiet : bool = False, sourmash_parameters = [50000, 0], include_reverse : bool = False) -> int:
    """
    Construct sourmash sketches given a fasta input.
    
    Args:
        *raw_in* (str): Path to the input fasta file or directory containing fasta files.
        *k* (int): Length of the k-mers. 
        *outdir* (str): directory for storing sketches created in data_prod_path+"SM_sketches/" (each signature in its own file)
        *quiet* (bool): If True, suppress progress output. Default is False.
        *sourmash_parameters* (list): specify sourmash.MinHash(n, scaled)
        *include_reverse* (bool): include the reverse strand to sketches
    
    Returns:
        *exit_status* (binary): 0 for success, 1 for failure.
    """
    import sourmash

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
    if not quiet: print(f"Output path for sketches: {outpath}")

    # Ensuring sourmash parameters are appropriate
    if sourmash_parameters[0] > 0 and sourmash_parameters[1] > 0:
        raise ValueError("One of the sourmash parameters should be 0")

    for p in sourmash_parameters:
        if type(p) is not int:
            raise ValueError("sourmash parameters must be both integers")

    # Handling both cases of fasta input
    raw_is_dir = os.path.isdir(raw_in)

    if raw_is_dir:  # If a directory path is provided, read all fasta files in the directory
        try:
            records = []
            rec_names = []
            for file in os.listdir(raw_in):
                rec_names.append(file.split("_reoriented.fna")[0])
                if file.endswith(".fasta") or file.endswith(".fna"):
                    records_inner = []
                    for rec in SeqIO.parse(os.path.join(raw_in, file), "fasta"):
                        records_inner.append(rec)
                    records.append(records_inner)
        except FileNotFoundError as e:
            print(f"Error: {e}. Please check the file path.")
            return 1
    else:
        try:
            records = list(SeqIO.parse(raw_in, "fasta"))
        except FileNotFoundError as e:
            print(f"Error: {e}. Please check the file path.")
            return 1

    ### Constructing minhashes for all records ###
    if not quiet: print("------- Constructing MinHashes -------")
    minhashes = []
    for rec in tqdm(records, desc="Constructing minhashes for all records", unit="seq"):
        if raw_is_dir:
            try:
                mh = sourmash.MinHash(n=sourmash_parameters[0], ksize=k, scaled=sourmash_parameters[1]) #each record gets its own minhash | scaled=1000 to limit memory usage
                for rec_inner in rec:
                    for i in range(0, len(rec_inner.seq) - k + 1):
                        kmer = str(rec_inner.seq[i:i+k])
                        mh.add_sequence(kmer, force=True)
                        if include_reverse:
                            mh.add_sequence(kmer[::-1], force=True)
                minhashes.append(mh)
            except:
                raise SystemError("Error in constructing minhashes")
        else:
            try:
                mh = sourmash.MinHash(n=sourmash_parameters[0], ksize=k, scaled=sourmash_parameters[1]) #each record gets its own minhash | scaled=1000 to limit memory usage
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
    if "bact" in raw_in:
        outfile_prefix = "bact"
    elif "phage" in raw_in:
        outfile_prefix = "phage"
    else:
        outfile_prefix = "out"

    for i in range(len(minhashes)):
        try:
            with open(outpath+f"{outfile_prefix}{i}_minhash.sig", "wt") as sigfile:
                if raw_is_dir:
                    sig1 = sourmash.SourmashSignature(minhashes[i], name=rec_names[i])
                else:
                    sig1 = sourmash.SourmashSignature(minhashes[i], name=records[i].id)
                sourmash.save_signatures([sig1], sigfile)
        except:
            raise SystemError(f"Error in saving sourmash sketch for: {records[i].id}")
    print("------- Process Completed -------")

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

def construct_interaction_pairs(phage_minhash_data : dict, bact_minhash_data : dict, host_range_data : dict, phage_names : list, bacteria_names : list, subset : int = None, outfile : str = None) -> [dict, dict, dict, dict, dict, dict]:
    """
    Construct a dictionary of interaction pairs given the minhash data for phages and bacteria and the host range data.
    The dictionary will have keys as (phage_hash, bact_hash) pairs and values as the interaction score from the host range data.

    Args:
        **phage_minhash_data** (dict): dictionary with keys as phage names and values as lists of minhashes.
        **bact_minhash_data** (dict): dictionary with keys as bacteria strain IDs and values as lists of minhashes.
        **host_range_data** (dict): nested dictionary with strains as outer keys, phage as inner keys and host range values as values.
        **phage_names** (list): list of phage names to consider (should match keys in phage_minhash_data)
        **bacteria_names** (list): list of bacteria names to consider (should match keys in bact_minhash_data)
        **subset** (int): number of combinations to consider (default is None, meaning all combinations)
        **outfile** (str): path to output file (default is None, meaning no file is written)
    Returns:
        **interaction_pairs** (dict): dictionary with keys as (phage_hash, bact_hash) pairs and values as the interaction score from the host range data.
        **occurence_pairs** (dict): dictionary with keys as (phage_hash, bact_hash) pairs and values as the number of occurrences of that pair across all phage-bacteria combinations.
        **interaction_freq_pairs** (dict): dictionary with keys as (phage_hash, bact_hash) pairs and values as the normalized interaction score for that pair across all phage-bacteria combinations (interaction score divided by occurrence count).
        **occurence_freq_pairs** (dict): dictionary with keys as (phage_hash, bact_hash) pairs and values as the normalized occurrence for that pair across all phage-bacteria combinations (interaction score divided by occurrence count).
        **expected_interactions** (dict): dictionary with keys as (phage_hash, bact_hash) pairs and values as the expected interaction score for that pair across all phage-bacteria combinations.
        **hash_lookup** (dict): dictionary with keys as hash values and values as a list of strains (phage or bacteria) that have that hash in their minhash sketch."""
    
    interaction_pairs = dict()
    occurence_pairs = dict()
    interaction_freq_pairs = dict()
    occurence_freq_pairs = dict()
    expected_interactions = dict()
    hash_lookup = dict()
    c = 0
    total_combinations = len(phage_names) * len(bacteria_names)
    
    # Create out directory if it doesn't exist
    outdir = outfile.rsplit("/", 1)[0] if outfile is not None else None
    if outdir is not None and not os.path.exists(outdir):
        try:
            os.makedirs(outdir, exist_ok=True)
            print(f"Created output directory: {outdir}")
        except OSError as e:
            print(f"Could not create outdir {outdir}: {e}")
            outdir = None  # Set to None to avoid further issues with saving

    for pname in phage_names:
        for bname in bacteria_names:
            # Supports nested dict format and tuple-key format
            interaction_score = host_range_data.get(bname, {}).get(pname, host_range_data.get((bname, pname), 0))

            for pkmer in phage_minhash_data.get(pname, []):
                for bkmer in bact_minhash_data.get(bname, []):
                    pair = (pkmer, bkmer)
                    if pair in interaction_pairs:
                        interaction_pairs[pair] += interaction_score
                    else:
                        interaction_pairs[pair] = interaction_score
                    occurence_pairs[pair] = occurence_pairs.get(pair, 0) + 1

                    # Populate hash lookup
                    if pkmer not in hash_lookup:
                        hash_lookup[pkmer] = [pname]
                    else:
                        if pname not in hash_lookup[pkmer]:
                            hash_lookup[pkmer].append(pname)
                    if bkmer not in hash_lookup:
                        hash_lookup[bkmer] = [bname]
                    else:
                        if bname not in hash_lookup[bkmer]:
                            hash_lookup[bkmer].append(bname)
    
            c += 1
            print(f"Int/Occ: Processed combination {c}/{total_combinations} (Phage: {pname}, Bacteria: {bname})", end="\r")
            if subset is not None and c >= subset:
                if sum(interaction_pairs.values()) > 0: #continue if no interaction has been found
                    print(f"\nReached subset limit of {subset} combinations, stopping pair construction.")
                    break

    if not sum(interaction_pairs.values()) > 0:
        print("Warning: Sum of interaction scores is 0, cannot calculate interaction frequencies.")
    if not sum(occurence_pairs.values()) > 0:
        print("Warning: Sum of occurrence counts is 0, cannot calculate occurrence frequencies.")

    ### Calculating frequencies (normalized by total interactions/occurrences) ###
    print("\nCalculating interaction, occurrence & expected frequencies...")
    c = 0
    if subset is not None:
        keys_in_subset = []
    for pair in interaction_pairs.keys():
        if sum(interaction_pairs.values()) > 0:
            interaction_freq_pairs[pair] = interaction_pairs[pair] / sum(interaction_pairs.values())
        else:
            interaction_freq_pairs[pair] = 0
        
        if sum(occurence_pairs.values()) > 0:
            occurence_freq_pairs[pair] = occurence_pairs[pair] / sum(occurence_pairs.values())
        else:
            occurence_freq_pairs[pair] = 0
        
        expected_interactions[pair] = interaction_freq_pairs[pair] * occurence_pairs[pair]
        
        c += 1
        print(f"Int/Occ Freq: Processed pair {c}/{len(interaction_pairs)}", end="\r")

        keys_in_subset.append(pair)
        if subset is not None and c >= subset:
            print(f"\nReached subset limit of {subset} pairs, stopping frequency calculation.")
            break
    
    if outfile is not None:
        try:
            with open(outfile, "w") as f:
                f.write("phage_hash\tbact_hash\tinteraction_score\toccurrence_count\tinteraction_freq\toccurrence_freq\texpected_interaction\n")
                for pair in keys_in_subset if subset is not None else interaction_pairs.keys():
                    f.write(f"{pair[0]}\t{pair[1]}\t{interaction_pairs[pair]}\t{occurence_pairs[pair]}\t{interaction_freq_pairs[pair]}\t{occurence_freq_pairs[pair]}\t{expected_interactions[pair]}\n")
            print(f"Interaction pairs saved to {outfile}")
        except Exception as e:
            print(f"Error saving interaction pairs to {outfile}: {e}")
    
    print(f"Total phage-bacteria combinations processed: {c}")
    return interaction_pairs, occurence_pairs, interaction_freq_pairs, occurence_freq_pairs, expected_interactions, hash_lookup

def aggregate_interaction_pairs(nn_runs : list, outdir : str = None, logging : bool = False):
    """
    Read the top_interaction_pair_kmers.csv files of directories in nn_runs, and aggregate them to unqiue decoded kmers with a count column, turning the metadata from a str column to a list of str.
    Args:
        **nn_runs** (list): list of directories containing the top_interaction_pair_kmers.csv files to aggregate.
        **outdir** (str): directory to save the aggregated file (default is None, a default directory will be created in data_prod_path/pfi_interaction_kmers_analysis/)
    """
    # Handling outdir
    if outdir is not None:
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir, exist_ok=True)
                if logging: print(f"Created output directory: {outdir}")
            except OSError as e:
                print(f"Could not create outdir {outdir}: {e}")
                outdir = None  # Set to None to avoid further issues with saving
    else:
        outdir = data_prod_path+"pfi_interaction_kmers_analysis/"
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir, exist_ok=True)
                if logging: print(f"Created default output directory: {outdir}")
            except OSError as e:
                print(f"Could not create default outdir {outdir}: {e}")
                return

    total_interaction_pairs_df = pd.DataFrame()
    df_length = len(total_interaction_pairs_df)
    for run_dir in nn_runs:
        #Check if the file exists
        top_interaction_pairs_path = os.path.join(path_to_nn_runs, run_dir, "top_interaction_pair_kmers.csv")
        if not os.path.exists(top_interaction_pairs_path):
            #Check if user has given the full path to the run_dir
            top_interaction_pairs_path = os.path.join(run_dir, "top_interaction_pair_kmers_downsized.csv")
            if not os.path.exists(top_interaction_pairs_path):
                print(f"File not found: {top_interaction_pairs_path}")
                continue
        
        if logging: print(f"Reading file: {top_interaction_pairs_path}")
        top_interaction_pairs_df = pd.read_csv(top_interaction_pairs_path)
        total_interaction_pairs_df = pd.concat([total_interaction_pairs_df, top_interaction_pairs_df], ignore_index=True)
        
        if len(total_interaction_pairs_df) == df_length:
            print(f"Warning: No new rows added from file: {top_interaction_pairs_path}")
        
        df_length = len(total_interaction_pairs_df)
        if logging: print(f"Processed file: {top_interaction_pairs_path}, Total rows: {df_length}.\nNow aggregating...")

    # Plotting
    kmer_abundance = Counter(total_interaction_pairs_df['decoded_kmer'].tolist())
    kmer_abundance_df = pd.DataFrame(kmer_abundance.items(), columns=['decoded_kmer', 'abundance'])

    plt.figure(figsize=(10, 6))
    sns.barplot(x='decoded_kmer', y='abundance', data=kmer_abundance_df.sort_values(by='abundance', ascending=False).head(20))
    plt.xticks(rotation=90)
    plt.title('Top 20 Most Abundant k-mers in Top Interaction Pairs')
    plt.xlabel('Decoded k-mer')
    plt.ylabel('Abundance')
    plt.tight_layout()
    plt.savefig(outdir+"top_20_pfi_kmers_abundance.png")

    #Aggregate to unique kmers, summing the counts and turning the metadata into a list of str
    # Aggregate total_interaction_pairs_df by combining "decoded_kmer" column, and recreating the other columns into a list of unique values for each "decoded_kmer"
    total_aggr_df = total_interaction_pairs_df.groupby('decoded_kmer').agg({
        'feature_index': lambda x: list(set(x)),
        'entity': lambda x: list(set(x)),
        'organism': lambda x: list(set(x))
    }).reset_index()

    total_aggr_df.to_csv(outdir+"top_interaction_kmers_aggregated.csv", index=False)

