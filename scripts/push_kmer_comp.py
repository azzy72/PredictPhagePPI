### Simply python code to push given code to a bash executable script on the pupil servers ###
import os
from manipulations import fasta_to_kmerdf
from analysis import perform_pca

# Define input fasta file and parameters
bact_dir = "raw_data/phagehost_KU/bacteria_fasta/"
bact_fastas = [bact_dir+file for file in os.listdir(bact_dir) if file.endswith('.fasta') or file.endswith(".fna")][:5]

# Convert fasta files to k-mer frequency DataFrame
k = 1000
bact_kmer_df = fasta_to_kmerdf(bact_fastas, k=k, quiet=False, sparse=False, relative=True)
print(bact_kmer_df)
print("Bacteria k-mer DataFrame shape:", bact_kmer_df.shape)