#!/usr/bin/python3

import argparse
import os
import pandas as pd
from manipulations import calc_PFI
from analysis import aggregate_interaction_pairs, plot_interaction_pairs

def parse_arguments():
    parser = argparse.ArgumentParser(description="Perform Pairwise Feature Interaction (PFI) analysis on the minhash data for phages and bacteria and the host range data, by constructing interaction pairs and calculating their frequencies.")
    parser.add_argument("--rundir", type=str, required=True, help="Directory containing the minhash data for phages and bacteria and the host range data. Should be a directory containing 'phage_minhash_data.pkl', 'bacteria_minhash_data.pkl'.")


def main():
    args = parse_arguments()
    # Check if args.rundir is a directory
    if not os.path.isdir(args.rundir):
        print(f"Error: {args.rundir} is not a valid directory.")
        return
    
    # Load data
    with open(os.path.join(args.rundir, ""))
    phage_minhash = pd.read_pickle(os.path.join(args.rundir, "phage_minhash_data.pkl"))
    bacteria_minhash = pd.read_pickle(os.path.join(args.rundir, "bacteria_minhash_data.pkl"))
    
    # Perform PFI analysis
    pfi_analyzer = calc_PFI(phage_names=)
    pfi_analyzer.construct_interaction_pairs(phage_minhash_data=phage_minhash, bacteria_minhash_data=bacteria_minhash)
    
    # Optionally aggregate and plot results
    aggregate_interaction_pairs([args.rundir], outdir=args.rundir)
    plot_interaction_pairs(args.rundir)


if __name__ == "__main__":
    main()