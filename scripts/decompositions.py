import os, sys
from Bio import SeqIO
from tqdm import tqdm
import shutil
import json
import hashlib
data_prod_path = "../data_prod/"
raw_data_path = "../raw_data/"

class KmerCodec:
    def __init__(self):
        # Map bases to 2-bit values
        self.base_to_bits = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        # Map 2-bit values back to bases
        self.bits_to_base = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

    def encode(self, kmer):
        """Converts a k-mer string into a unique integer."""
        encoded_int = 0
        for base in kmer:
            # Shift the existing bits 2 places left to make room for the new base
            # Then use OR (|) to add the 2 bits for the current base
            encoded_int = (encoded_int << 2) | self.base_to_bits[base]
        return encoded_int
    
    def encode_with_revcomp(self, kmer):
        """Encodes a k-mer and its reverse complement, returning the smaller integer."""
        encoded_forward = self.encode(kmer)
        # Compute reverse complement
        reverse_kmer = ''.join(self.bits_to_base[3 - self.base_to_bits[base]] for base in reversed(kmer))
        encoded_reverse = self.encode(reverse_kmer)
        # Return the smaller of the two encodings to ensure consistency
        return min(encoded_forward, encoded_reverse)

    def decode(self, encoded_int, k):
        """Backtracks an integer into the original k-mer string of length k."""
        bases = []
        for _ in range(k):
            # Use a bitmask (3 is 11 in binary) to extract the last 2 bits
            bits = encoded_int & 3
            bases.append(self.bits_to_base[bits])
            # Shift the integer 2 places right to move to the next base
            encoded_int >>= 2
        
        # Since we extracted from right-to-left, we must reverse the list
        return "".join(reversed(bases))


class Decompose:
    def __init__(self, k, n, codec, output_dir, entity_type, sourmash_like=True):
        allowed_entity_types = {"phage": "phage", "bacteriophage": "phage", "bacteria": "bact", "bact": "bact"}
        if entity_type.lower() not in allowed_entity_types.keys():
            raise ValueError(f"Invalid entity_type '{entity_type}'. Allowed values are: {', '.join(allowed_entity_types.keys())}")

        self.k = k
        self.n = n
        self.codec = codec
        self.output_dir = output_dir
        self.entity_type = allowed_entity_types[entity_type.lower()]
        self.temp_dir = "../data_prod/tmp"
        self.sourmash_like = sourmash_like

    def __enter__(self):
        """Sets up the environment when entering the 'with' block."""
        os.makedirs(self.temp_dir, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Guaranteed cleanup when exiting the 'with' block."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def decompose(self, fasta_path):
        """Main method to decompose genomes from a FASTA file."""
        yield f"Initialized Decompose with k={self.k}, n={self.n}, entity_type='{self.entity_type}', sourmash_like={self.sourmash_like}"

        #Saving like sourmash
        if self.sourmash_like:
            curr_dir = self.output_dir+f"{self.entity_type}_sig_n{self.n}_k{self.k}/"
            try:
                os.makedirs(curr_dir)
            except FileExistsError:
                shutil.rmtree(curr_dir)
                os.makedirs(curr_dir)
            for i, record in tqdm(enumerate(SeqIO.parse(fasta_path, "fasta")), desc=f"Processing {self.entity_type} FASTA", unit="rec"):
                record_name = record.id
                sig = self.decompose_genome(str(record.seq).upper())
                sig = self.prepare_sourmash_structure(sig, record_name)
                self.save_sketches_to_one_file(sig, os.path.join(curr_dir, f"{self.entity_type}{i}_{record_name.lower()}.sig"))

        #Saving customly in one file
        if not self.sourmash_like:
            for record in tqdm(SeqIO.parse(fasta_path, "fasta"), desc=f"Processing {self.entity_type} FASTA", unit="rec"):
                sig = self.decompose_genome(str(record.seq).upper())
                self.save_sketches_to_one_file(sig, os.path.join(self.temp_dir, f"signatures_{self.k}_{self.n}.txt"))
            
            self.concatenate_sketches()

    def decompose_genome(self, genome_seq):
        kmers = [genome_seq[i:i+self.k] for i in range(len(genome_seq) - self.k + 1)]
        if not kmers: return []
        
        sampling_interval = max(1, len(kmers) // self.n)
        signatures = []
        
        for i in range(0, len(kmers), sampling_interval):
            if len(kmers[i]) == self.k:
                # Assuming your codec has this method now
                encoded = self.codec.encode(kmers[i]) 
                signatures.append(encoded)
        
        return signatures[:self.n]

    def prepare_sourmash_structure(self, signatures, record_name):
        """
        Wraps signatures into a sourmash-compatible dictionary structure.
        """
        # Sort signatures to emulate a MinHash 'mins' list behavior
        sorted_sigs = sorted(signatures)
        
        # Calculate an md5sum of the signatures to follow the format
        sig_string = ",".join(map(str, sorted_sigs))
        md5sum = hashlib.md5(sig_string.encode()).hexdigest()

        # Create the inner signature block
        sig_data = {
            "num": len(sorted_sigs),
            "ksize": self.k,
            "seed": 42, # Default for sourmash
            "max_hash": 0,
            "mins": sorted_sigs,
            "md5sum": md5sum,
            "molecule": "dna"
        }

        # Create the outer wrapper (as a list containing one dictionary)
        full_signature = [{
            "class": "sourmash_signature",
            "email": "",
            "hash_function": "0.bit_encoding", # Custom label for your reversible method
            "filename": None,
            "name": record_name,
            "license": "CC0",
            "signatures": [sig_data],
            "version": 0.4
        }]

        return full_signature
    
    def save_sketches_to_one_file(self, sig, output_path):
        # Use 'a' (append) so multiple records don't overwrite each other
        with open(output_path, 'a') as f:
            json.dump(sig, f, indent=2)
            f.write("\n")

    def concatenate_sketches(self):
        output_path = os.path.join(self.output_dir, f"{self.entity_type}_sigs_{self.k}_{self.n}.txt")
        os.makedirs(self.output_dir, exist_ok=True)
        
        with open(output_path, 'w') as outfile:
            target_file = f"signatures_{self.k}_{self.n}.txt"
            file_path = os.path.join(self.temp_dir, target_file)
            if os.path.exists(file_path):
                with open(file_path) as infile:
                    shutil.copyfileobj(infile, outfile)
