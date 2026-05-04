#!/usr/bin/python3

from py_compile import main
import os, sys, argparse
from time import time
from paths import scripts_path

def parse_arguments():
    parser = argparse.ArgumentParser(description="FFNN Training Script")

    # Parameters: Mutual exclusivity for n/k vs specific bn/bk/pn/pk
    parser.add_argument("--n", type=int, nargs='+',
                        help="A list of n values for downsampling (e.g., --n 500 1000 2000)")

    parser.add_argument("--k", type=int, nargs='+',
                        help="A list of k values for downsampling (e.g., --k 12 15 20)")

    parser.add_argument("--method", choices=['sourmash', 'minhash', 'ohe'], help="Downsampling method to use (default: sourmash)", default='sourmash')
    parser.add_argument("--data2", action="store_true", help="Use the second dataset with EOP values instead of binary interactions")

    args = parser.parse_args()
    return args

def main_batch():
    print("Starting batch downsampling script...")
    args = parse_arguments()

    print(f"Downsampling method: {args.method}")
    for n in args.n:
        for k in args.k:
            try: 
                # Call downsampling.py with the current n and k values as arguments
                command = f"python3 {scripts_path}downsampling.py --nk {n} {k} --method {args.method}"
                if args.data2:
                    command += " --data2"
                print(f"Running command: {command}")
                os.system(command)
            except Exception as e:
                print(f"Error during batch downsampling for n={n}, k={k}: {e}")
                continue

if __name__ == "__main__":
    main_batch()