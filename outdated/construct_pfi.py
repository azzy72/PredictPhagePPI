#!/usr/bin/python3

import os, sys, re
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse, logging
from time import time
from datetime import datetime
import pickle

from manipulations import binarize_host_range, hostrange_df_to_dict, calc_PFI, clean_bact_names
from io_operations import call_hostrange_df, presence_matrix
from paths import raw_data_path, data_prod_path

data2 = False

def parse_arguments():
    parser = argparse.ArgumentParser(description="Construct Presence-Feature-Interaction (PFI) matrix for phage-bacteria interactions.")
    parser.add_argument("--nk", nargs=2, type=int, metavar=('N', 'K'), help="Use the same N and K values for both phages and bacteria. Provide as: --nk N K")
    parser.add_argument("--split-nk", nargs=4, type=int, metavar=('BN', 'BK', 'PN', 'PK'), help="Use different N and K values for phages and bacteria. Provide as: --split-nk BN BK PN PK")
    parser.add_argument("--use-encoded", action='store_true', help="Use encoded sketches instead of sourmash sketches for presence matrix construction.")
    parser.add_argument("--data2", action='store_true', help="Use data2 instead of data1 for presence matrix construction.")
    args = parser.parse_args()

    if args.nk and args.split_nk:
        parser.error("Cannot use both --nk and --split-nk arguments at the same time. Please choose one.")
    elif not args.nk and not args.split_nk:
        parser.error("Must provide either --nk or --split-nk arguments to specify N and K values.")

    return args

def main():
    args = parse_arguments()
    time_start = time()

    ### 1. Resolve N/K values ###
    if args.nk:
        n = bn = pn = args.nk[0]
        k = bk = pk = args.nk[1]
        presmat_suffix = f"n{n}_k{k}"
    else:
        bn, bk, pn, pk = args.split_nk
        n, k = bn, bk # Reference n/k for folder naming
        presmat_suffix = f"bn{bn}_bk{bk}_pn{pn}_pk{pk}"

    prefix = "encoded_sketches" if args.use_encoded else "SM_sketches"
    if args.data2:
        prefix = f"{prefix}_data2"

    #use regex to find directories with n{bn}_k{bk} and n{pn}_k{pk} in their names, since dir prefix depends on method
    files_prefix_dirs = os.listdir(os.path.join(data_prod_path, prefix))
    
    phage_dir_pattern = re.compile(f".*n{pn}_k{pk}.*")
    input_phage_path = [file for file in files_prefix_dirs if phage_dir_pattern.match(file) and "phage" in file.lower()]
    if len(input_phage_path) == 0:
        raise FileNotFoundError(f"No directory found for phage minhash data with n={pn} and k={pk} in {os.path.join(data_prod_path, prefix)}")
    elif len(input_phage_path) > 1:
        raise ValueError(f"Multiple directories found for phage minhash data with n={pn} and k={pk} in {os.path.join(data_prod_path, prefix)}: {input_phage_path}")
    input_phage_path = f"{prefix}/{input_phage_path[0]}/"

    bact_dir_pattern = re.compile(f".*n{bn}_k{bk}.*")
    input_bact_path = [file for file in files_prefix_dirs if bact_dir_pattern.match(file) and "bact" in file.lower()]
    if len(input_bact_path) == 0:
        raise FileNotFoundError(f"No directory found for bacteria minhash data with n={bn} and k={bk} in {os.path.join(data_prod_path, prefix)}")
    elif len(input_bact_path) > 1:
        raise ValueError(f"Multiple directories found for bacteria minhash data with n={bn} and k={bk} in {os.path.join(data_prod_path, prefix)}: {input_bact_path}")
    input_bact_path = f"{prefix}/{input_bact_path[0]}/"

    presmat_path = f"{prefix}/PresMat_{presmat_suffix}/"
    print(f"Recognized data paths\ninput_phage_path:\t{input_phage_path}\ninput_bact_path:\t{input_bact_path}\npresmat_path:\t{presmat_path}")

    pfi_outdir_name = f"pfi_n{n}_k{k}/"

    outdir = os.path.join(data_prod_path, prefix, pfi_outdir_name)
    if os.path.exists(outdir):
        #clear existing files in the directory
        for file in os.listdir(outdir):
            os.remove(os.path.join(outdir, file))
    else:
        os.makedirs(outdir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(outdir, f'pfi_log.txt'),
        filemode='w', # 'a' for append, 'w' for overwrite
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.info(f'Run started: {datetime.now()}\n')
    logging.info('Params:')
    for key, value in vars(args).items():
        logging.info(f'  {key}: {value}')
    logging.info('##################')
    print("Logging enabled. Output directory created at:", outdir)

    ### Host Range Setup ###
    bact_lookup, host_range_df = call_hostrange_df(os.path.join(raw_data_path, "phagehost_KU/Hostrange_data_all_crisp_iso.xlsx"), data2=data2)
    host_range_data = binarize_host_range(hostrange_df_to_dict(host_range_df), continous=False)
    host_range_data = {clean_bact_names(bact.replace("_reoriented", "")): interactions for bact, interactions in host_range_data.items()} # if "_reoriented" is in the bacteria names in host_range_data, remove it to match the bacteria names in the presence matrix.
    print(f"Host range data loaded with {len(host_range_data)} bacteria and {len(next(iter(host_range_data.values())))} phages.")

    ### Presence Matrix Setup ###
    ### 3. Load Data ###
    #bact_clusters = pd.read_csv(os.path.join(data_prod_path, "bact_clusters_with_genus.csv"), index_col=0)
    # Load Presence Matrix
    full_presmat_path = os.path.join(data_prod_path, presmat_path)
    if not os.path.exists(full_presmat_path):
        print("Reconstructing presence_matrix...")
        binary_matrix, entity_to_index, minhash_to_index, phage_minhash_data, bact_minhash_data = presence_matrix(
            phage_minhash_dir=os.path.join(data_prod_path, input_phage_path),
            bact_minhash_dir=os.path.join(data_prod_path, input_bact_path),
            k=[bk, pk], n=[bn, pn], reversecomp_data=False, TS=True, data2=args.data2)

        #Write files to full_presmat_path for faster loading in the future
        os.makedirs(full_presmat_path, exist_ok=True)
        with open(os.path.join(full_presmat_path, "binary_matrix.pkl"), "wb") as f: pickle.dump(binary_matrix, f)
        with open(os.path.join(full_presmat_path, "entity_to_index.pkl"), "wb") as f: pickle.dump(entity_to_index, f)
        with open(os.path.join(full_presmat_path, "phage_minhash_data.pkl"), "wb") as f: pickle.dump(phage_minhash_data, f)
        with open(os.path.join(full_presmat_path, "bact_minhash_data.pkl"), "wb") as f: pickle.dump(bact_minhash_data, f)
        with open(os.path.join(full_presmat_path, "minhash_to_index.pkl"), "wb") as f: pickle.dump(minhash_to_index, f)

    else:
        print("Loading presence matrix from pre-saved files...")
        with open(os.path.join(full_presmat_path, "binary_matrix.pkl"), "rb") as f: binary_matrix = pickle.load(f)
        with open(os.path.join(full_presmat_path, "entity_to_index.pkl"), "rb") as f: entity_to_index = pickle.load(f)
        with open(os.path.join(full_presmat_path, "phage_minhash_data.pkl"), "rb") as f: phage_minhash_data = pickle.load(f)
        with open(os.path.join(full_presmat_path, "bact_minhash_data.pkl"), "rb") as f: bact_minhash_data = pickle.load(f)
        with open(os.path.join(full_presmat_path, "minhash_to_index.pkl"), "rb") as f: minhash_to_index = pickle.load(f) 
    
    ### 4. Calculate PFI ###
    hash_lookup = None
    pfi_failed = False
    out_pfi = f"pfi_n{n}_k{k}.txt"
    pfi_objects_dir = f"pfi_objects_{prefix}_n{n}_k{k}/"
    hash_lookup = "hash_lookup.csv"

    pfi_analyzer = calc_PFI(host_range_data=host_range_data, outdir=outdir, outname_pfi=out_pfi, pfi_objects_dir=pfi_objects_dir, logging=True)
    interaction_pairs, occurence_pairs, interaction_freq_pairs, occurence_freq_pairs, expected_interactions, hash_lookup = pfi_analyzer.construct_interaction_pairs(phage_minhash_data=phage_minhash_data, bact_minhash_data=bact_minhash_data)


if __name__ == "__main__":
    main()