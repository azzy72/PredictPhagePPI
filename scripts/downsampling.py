#!/usr/bin/python3

import os, sys, argparse
from time import time
from manipulations import construct_SM_sketches
from decompositions import KmerCodec, Decompose
from paths import raw_data_path, data_prod_path

def parse_arguments():
    parser = argparse.ArgumentParser(description="FFNN Training Script")

    # Parameters: Mutual exclusivity for n/k vs specific bn/bk/pn/pk
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--nk", nargs=2, type=int, metavar=('N', 'K'),
                        help="Unified n and k values (e.g., -nk 500 12)")
    group.add_argument("--split_nk", nargs=4, type=int, metavar=('BN', 'BK', 'PN', 'PK'),
                        help="Split values for Bact (n, k) and Phage (n, k)")
    
    parser.add_argument("--method", choices=['sourmash', 'minhash', 'ohe'], help="Downsampling method to use (default: sourmash)", default='sourmash')
    parser.add_argument("--hash", choices=["xxhash", "mmh3", "ohe_custom"], default='mmh3', help="Hash function to use for OHE method (default: mmh3)")
    parser.add_argument("--data2", action="store_true", help="Use the second dataset with EOP values instead of binary interactions")
    parser.add_argument("--all_phages", action="store_true", help="Include all phages in the dataset (overrides n for phages)")
    parser.add_argument("--test", action="store_true", help="For testing, writes output paths with 'TEST_' prefix and only processes the first 10 sequences from each input file")

    args = parser.parse_args()
    return args

def main():
    print("Starting downsampling script...")
    args = parse_arguments()
    if args.hash == "ohe_custom":
        approach = "One-hot encoding with custom hash function"
    else:
        approach = f"{args.method.capitalize()} with hash function: {args.hash}"
    print(f"Selected downsampling approach: {approach}")
    print(f"Downsampling method: {args.method}")
    print(f"All phages included: {args.all_phages}")
    print(f"Test mode enabled: {args.test}")
    phage_in_path = raw_data_path+"phagehost_KU/phage_cleaned.fasta" if not args.data2 else raw_data_path+"phagehost_KU/data2_phages.fasta"
    bact_in_path = raw_data_path+"phagehost_KU/concatenated_bacteria/" if not args.data2 else raw_data_path+"phagehost_KU/data2_bacts.fasta"

    if args.method == 'sourmash' or args.method == 'minhash':
        ### Resolve N/K values ###
        if args.nk:
            n = bn = pn = args.nk[0]
            k = bk = pk = args.nk[1]
            bact_outdir = f"BactMinhash_n{n}_k{k}/" if not args.data2 else f"BactMinhash_data2_n{n}_k{k}/"
            phage_outdir = f"PhageMinhash_n{n}_k{k}/" if not args.data2 else f"PhageMinhash_data2_n{n}_k{k}/"
        else:
            bn, bk, pn, pk = args.split_nk
            n, k = bn, bk # Reference n/k for folder naming
            bact_outdir = f"BactMinhash_n{bn}_k{bk}/" if not args.data2 else f"BactMinhash_data2_n{bn}_k{bk}/"
            phage_outdir = f"PhageMinhash_n{pn}_k{pk}/" if not args.data2 else f"PhageMinhash_data2_n{pn}_k{pk}/"
        
        if args.data2:
            par_outdir = "SM_sketches_data2/"
        else:
            par_outdir = "SM_sketches/"
        
        if args.test:
            phage_outdir = "TEST_" + phage_outdir
            bact_outdir = "TEST_" + bact_outdir
            par_outdir = "TEST_" + par_outdir

        if args.all_phages:
            par_outdir = par_outdir.rstrip("/") + "_allphages/"
            #pn = 10**1000000000 # Effectively infinite n for all phages

        try:
            ### Phage Minhash Sketch Construction ###
            ps = 0 #the scaled argument
            if args.all_phages:
                pn = 0
                ps = 1

            construct_SM_sketches(raw_in = phage_in_path, 
                                k = pk, 
                                outdir = phage_outdir, 
                                parent_outdir = par_outdir,
                                quiet = False,
                                sourmash_parameters=[pn, ps])

            ### Bacteria Minhash Sketch Construction ###
            construct_SM_sketches(raw_in = bact_in_path, 
                                k = bk, 
                                outdir = bact_outdir, 
                                parent_outdir = par_outdir,
                                quiet = False,
                                sourmash_parameters=[bn, 0])
        except Exception as e:
            print(f"Error during sourmash sketch construction: {e}")
            sys.exit(1)

    elif args.method == 'ohe':
        ### Resolve N/K values ###
        if args.nk:
            n = bn = pn = args.nk[0]
            k = bk = pk = args.nk[1]
            bact_outdir = f"Bact{args.hash}_n{n}_k{k}/" if not args.data2 else f"Bact{args.hash}_data2_n{n}_k{k}/"
            phage_outdir = f"Phage{args.hash}_n{n}_k{k}/" if not args.data2 else f"Phage{args.hash}_data2_n{n}_k{k}/"
        else:
            bn, bk, pn, pk = args.split_nk
            n, k = bn, bk # Reference n/k for folder naming
            bact_outdir = f"Bact{args.hash}_n{bn}_k{bk}/" if not args.data2 else f"Bact{args.hash}_data2_n{bn}_k{bk}/"
            phage_outdir = f"Phage{args.hash}_n{pn}_k{pk}/" if not args.data2 else f"Phage{args.hash}_data2_n{pn}_k{pk}/"
            n, k = bn, bk # Reference n/k for folder naming

        try:
            codec = KmerCodec()
            ohe_outdir = f"encoded_sketches/" if not args.data2 else f"encoded_sketches_data2/"

            if args.test:
                phage_outdir = "TEST_" + phage_outdir
                bact_outdir = "TEST_" + bact_outdir
                ohe_outdir = "TEST_" + ohe_outdir

            if args.all_phages:
                ohe_outdir = ohe_outdir.rstrip("/") + "_allphages/"

            with Decompose(k=pk, n=pn, codec=codec, output_dir=data_prod_path+ohe_outdir, entity_type="phage", sourmash_like=True,
                           custom_dir_name=phage_outdir, hash_func=args.hash, sample_all=args.all_phages) as decompose_phage:
                decompose_phage.decompose(raw_in=phage_in_path)

            with Decompose(k=bk, n=bn, codec=codec, output_dir=data_prod_path+ohe_outdir, entity_type="bacteria", sourmash_like=True,
                           custom_dir_name=bact_outdir, hash_func=args.hash) as decompose_bact:
                decompose_bact.decompose(raw_in=bact_in_path)
            print(f"{approach} completed successfully.\n")
        except Exception as e:
            print(f"Error during {approach}: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()