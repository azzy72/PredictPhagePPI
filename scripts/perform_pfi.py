#!/usr/bin/python3

import argparse
import os
import pandas as pd
from manipulations import calc_PFI, aggregate_interaction_pairs
from analysis import plot_interaction_pairs
from paths import data_prod_path
from datetime import datetime

def parse_arguments():
    parser = argparse.ArgumentParser(description="Perform Pairwise Feature Interaction (PFI) analysis on the minhash data for phages and bacteria and the host range data, by constructing interaction pairs and calculating their frequencies.")
    parser.add_argument("--rundir", type=str, required=True, help="Directory containing the minhash data for phages and bacteria and the host range data. Should be a directory containing 'phage_minhash_data.pkl', 'bacteria_minhash_data.pkl'.")
    parser.add_argument("--nk", nargs=2, type=int, metavar=('N', 'K'), default=(500, 12),
                        help="Unified n and k values (e.g., -nk 500 12)")

def main():
    args = parse_arguments()

    # Outdir in data_prod_path+"kmer_pairs_for_downsamples/"+f"mlp_interaction_pairs_n{n}_k{k}/"
    outdir = os.path.join(data_prod_path, "kmer_pairs_for_downsamples", f"mlp_interaction_pairs_n{args.nk[0]}_k{args.nk[1]}")
    # Check if args.rundir and outdir are directories
    if not os.path.isdir(args.rundir):
        print(f"Error: {args.rundir} is not a valid directory.")
        return
    if not os.path.isdir(outdir):
        print(f"Did not recognize {outdir} as a valid directory, making it now.")
        os.makedirs(outdir, exist_ok=True)

    # Load data
    try:
        # with open(os.path.join(args.rundir, "bacteria_names.txt"), "r") as f:
        #     bacteria_names = [line.strip() for line in f]
        # with open(os.path.join(args.rundir, "phage_names.txt"), "r") as f:
        #     phage_names = [line.strip() for line in f]
        phage_minhash = pd.read_pickle(os.path.join(args.rundir, "phage_minhash_data.pkl"))
        bacteria_minhash = pd.read_pickle(os.path.join(args.rundir, "bacteria_minhash_data.pkl"))
        bact_clusters = pd.read_csv(os.path.join(data_prod_path, "bact_clusters_with_genus.csv"), index_col=0)
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Perform PFI analysis
    pfi_analyzer = calc_PFI()
    interaction_pairs, occurence_pairs, interaction_freq_pairs, occurence_freq_pairs, expected_interactions, hash_lookup = pfi_analyzer.construct_interaction_pairs(phage_minhash_data=phage_minhash, bacteria_minhash_data=bacteria_minhash)
    
    # Optionally aggregate and plot results
    plot_interaction_pairs(interaction_pairs, occurence_pairs, expected_interactions, hash_lookup, 
                           sort_by_ratio=True, logging=True, outdir=args.rundir)

    #aggregate_interaction_pairs([args.rundir], outdir=args.rundir)

    # Record time of success and terminate
    print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} --- Finished pfi analysis ---')


if __name__ == "__main__":
    main()