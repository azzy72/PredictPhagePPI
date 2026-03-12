#!/usr/bin/python3

import os, sys, argparse
from time import time
from manipulations import construct_SM_sketches
from decompositions import KmerCodec, Decompose
from paths import raw_data_path, data_prod_path

def parse_arguments():
    parser = argparse.ArgumentParser(description="Downsample bacteria and phage genomes")

    # Parameters: Mutual exclusivity for n/k vs specific bn/bk/pn/pk
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--nk", nargs=2, type=int, metavar=('N', 'K'),
                        help="Unified n and k values (e.g., -nk 500 12)")
    group.add_argument("--split_nk", nargs=4, type=int, metavar=('BN', 'BK', 'PN', 'PK'),
                        help="Split values for Bact (n, k) and Phage (n, k)")

    parser.add_argument("--method", choices=['sourmash', 'ohe'], help="Downsampling method to use (default: sourmash)", default='sourmash')

def main():
    args = parse_arguments()
    time_start = time()

    if args.method == 'sourmash':
        ### Resolve N/K values ###
        if args.nk:
            n = bn = pn = args.nk[0]
            k = bk = pk = args.nk[1]
            bact_outdir = f"BactMinhash_n{n}_k{k}/"
            phage_outdir = f"PhageMinhash_n{n}_k{k}/"
        else:
            bn, bk, pn, pk = args.split_nk
            n, k = bn, bk # Reference n/k for folder naming
            bact_outdir = f"BactMinhash_n{bn}_k{bk}/"
            phage_outdir = f"PhageMinhash_n{pn}_k{pk}/"

        ### Phage Minhash Sketch Construction ###
        construct_SM_sketches(fasta = raw_data_path+"phagehost_KU/phage_cleaned.fasta", 
                            k = pk, 
                            outdir = phage_outdir, 
                            quiet = False,
                            sourmash_parameters=[pn, 0])

        ### Bacteria Minhash Sketch Construction ###
        construct_SM_sketches(fasta = raw_data_path+"phagehost_KU/bacteriaKU_cleaned.fasta", 
                            k = bk, 
                            outdir = bact_outdir, 
                            quiet = False,
                            sourmash_parameters=[bn, 0])
    
    elif args.method == 'ohe':
        ### Resolve N/K values ###
        if args.nk:
            n = bn = pn = args.nk[0]
            k = bk = pk = args.nk[1]
            bact_inner_dirname = f"encode4bit_n{n}_k{k}/"
            phage_inner_dirname = f"encode4bit_n{n}_k{k}/"
        else:
            bn, bk, pn, pk = args.split_nk
            n, k = bn, bk # Reference n/k for folder naming
            bact_inner_dirname = f"encode4bit_n{bn}_k{bk}/"
            phage_inner_dirname = f"encode4bit_n{pn}_k{pk}/"

        print("One-hot encoding downsampling method is not yet implemented.")
        codec = KmerCodec()
        with Decompose(k=pk, n=pn, codec=codec, output_dir=data_prod_path+"encoded_sketches/", 
                                    entity_type="phage") as decompose_phage:
            decompose_phage.decompose(raw_data_path+"phagehost_KU/phage_cleaned.fasta")

        with Decompose(k=bk, n=bn, codec=codec, output_dir=data_prod_path+"encoded_sketches/", 
                                   entity_type="bacteria") as decompose_bact:
            decompose_bact.decompose(raw_data_path+"phagehost_KU/bacteriaKU_cleaned.fasta")