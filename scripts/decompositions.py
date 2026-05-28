import os, sys
from Bio import SeqIO
from tqdm import tqdm
import shutil
import json
import hashlib
from paths import raw_data_path, data_prod_path
import random
import mmh3, hashlib, heapq, xxhash

class KmerCodec:
    def __init__(self):
        # Map bases to 4-bit values
        #self.base_to_bits = {'A': 0, 'C': 1, 'G': 2, 'T': 3} #2bit encoding
        self.base_to_bits = {'A': 1, 'C': 2, 'G': 4, 'T': 8} #4bit encoding
        # Map 4-bit values back to bases
        #self.bits_to_base = {0: 'A', 1: 'C', 2: 'G', 3: 'T'} #2bit decoding
        self.bits_to_base = {1: 'A', 2: 'C', 4: 'G', 8: 'T'} #4bit decoding

        self.complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

    def encode(self, kmer):
        """Converts a k-mer string into a unique integer."""
        encoded_int = 0
        for base in kmer:
            # Shift the existing bits 4 places left to make room for the new base
            # Then use OR (|) to add the 4 bits for the current base
            encoded_int = (encoded_int << 4) | self.base_to_bits[base]
        return encoded_int
    
    def encode_with_revcomp(self, kmer):
        """Encodes a k-mer and its reverse complement, returning the smaller integer."""
        encoded_forward = self.encode(kmer)
        # Compute reverse complement
        reverse_kmer = ''.join(self.complement_map[base] for base in reversed(kmer))
        encoded_reverse = self.encode(reverse_kmer)
        # Return the smaller of the two encodings to ensure consistency
        return min(encoded_forward, encoded_reverse)

    def decode(self, encoded_int, k):
        """Backtracks an integer into the original k-mer string of length k."""
        bases = []
        for _ in range(k):
            # Use a bitmask (15 is 1111 in binary) to extract the last 4 bits
            bits = encoded_int & 15
            bases.append(self.bits_to_base[bits])
            # Shift the integer 4 places right to move to the next base
            encoded_int >>= 4
        
        # Since we extracted from right-to-left, we must reverse the list
        return "".join(reversed(bases))


class Decompose:
    def __init__(self, k, n, codec, output_dir, entity_type, sourmash_like=True, custom_dir_name: str = None, random_sampling=False, hash_func="mmh3", sample_all=False):
        allowed_entity_types = {"phage": "phage", "bacteriophage": "phage",
                                "bacteria": "bact", "bact": "bact"}
        if entity_type.lower() not in allowed_entity_types:
            raise ValueError(f"Invalid entity_type '{entity_type}'. "
                            f"Allowed values are: {', '.join(allowed_entity_types)}")

        self.k = k
        self.sample_all = sample_all
        # In sample-all mode, n has no meaning — keep it for back-compat but ignore it.
        self.n = None if sample_all else n
        self.codec = codec
        self.output_dir = output_dir
        self.entity_type = allowed_entity_types[entity_type.lower()]
        self.temp_dir = "../data_prod/tmp"
        self.sourmash_like = sourmash_like
        self.random_sampling = random_sampling

        if hash_func not in ["xxhash", "mmh3", "ohe_custom"]:
            raise ValueError(f"Invalid hash function '{hash_func}'. "
                            "Allowed values are: 'xxhash', 'mmh3', 'ohe_custom'")
        self.hash_func = hash_func

        n_label = "all" if sample_all else f"n{self.n}"
        if custom_dir_name:
            self.inner_dir = os.path.join(self.output_dir, custom_dir_name)
            print(f"Using custom directory name: {self.inner_dir}")
        else:
            self.inner_dir = self.output_dir + f"{self.entity_type}_sig_{n_label}_k{self.k}/"
            print(f"Using standard directory name: {self.inner_dir}")

    def __enter__(self):
        """Sets up the environment when entering the 'with' block."""
        os.makedirs(self.temp_dir, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Guaranteed cleanup when exiting the 'with' block."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def decompose(self, raw_in):
        """Main method to decompose genomes from a FASTA file."""
        raw_is_dir = os.path.isdir(raw_in)
        sig = None
        hk_lookup_global = {}
        print(f"Initialized Decompose with k={self.k}, n={self.n}, entity_type='{self.entity_type}', sourmash_like={self.sourmash_like}")
        print(f"Input path '{raw_in}' is a directory: {raw_is_dir}")

        # Saving like sourmash
        if self.sourmash_like:
            print(f"Processing with sourmash-like output format. Output will be saved in '{self.inner_dir}'")
            # Consolidated directory setup (one pass instead of repeated checks)
            if os.path.exists(self.inner_dir):
                shutil.rmtree(self.inner_dir)
            os.makedirs(self.inner_dir)
            
            if raw_is_dir:
                print(f"Processing directory of FASTA files for {self.entity_type} decomposition.")
                self._process_directory(raw_in, hk_lookup_global)
            else:
                print(f"Processing single FASTA file for {self.entity_type} decomposition.")
                self._process_single_file(raw_in, hk_lookup_global)
            
            print(f"Sourmash-like decomposition completed successfully. Signatures saved in '{self.inner_dir}'.\n")

        # Saving customly in one file
        elif not self.sourmash_like:
            if raw_is_dir:
                raise NotImplementedError("Custom saving method is not implemented for directory input yet.")
            self._process_single_file_custom(raw_in, hk_lookup_global)
            self.concatenate_sketches()

        if sig is None and not hk_lookup_global:
            raise ValueError("No signatures were generated. Please check the input FASTA file(s) and parameters.\n")
        
        n_label = "all" if self.sample_all else f"n{self.n}"
        self.save_hk_lookup(hk_lookup_global, f"hk_lookup_{n_label}_k{self.k}")

    def _process_directory(self, raw_in, hk_lookup_global):
        """Helper to process directory of FASTA files."""
        for fasta_file in tqdm(os.listdir(raw_in), desc=f"Processing {self.entity_type} FASTA files", unit="file"):
            if not (fasta_file.endswith(".fasta") or fasta_file.endswith(".fna")):
                continue
            
            file_path = os.path.join(raw_in, fasta_file)
            record_name = fasta_file.split("_reoriented.fna")[0]
            sigs_fasta = []
            
            for i, record in enumerate(SeqIO.parse(file_path, "fasta")):
                sig, hk_lookup = self.decompose_genome(str(record.seq).upper())
                hk_lookup_global.update(hk_lookup)
                sigs_fasta.append(sig)
            
            sig = self.prepare_sourmash_structure(sigs_fasta, record_name)
            self.save_sketches_to_one_file(sig, os.path.join(self.inner_dir, f"{self.entity_type}_{record_name.lower()}.sig"))

    def _process_single_file(self, raw_in, hk_lookup_global):
        """Helper to process single FASTA file."""
        for i, record in tqdm(enumerate(SeqIO.parse(raw_in, "fasta")), desc=f"Processing {self.entity_type} FASTA", unit="rec"):
            sig, hk_lookup = self.decompose_genome(str(record.seq).upper())
            hk_lookup_global.update(hk_lookup)
            sig = self.prepare_sourmash_structure(sig, record.id)
            self.save_sketches_to_one_file(sig, os.path.join(self.inner_dir, f"{self.entity_type}{i}_{record.id.lower()}.sig"))

    def _process_single_file_custom(self, raw_in, hk_lookup_global):
        """Helper for custom single-file output."""
        for record in tqdm(SeqIO.parse(raw_in, "fasta"), desc=f"Processing {self.entity_type} FASTA", unit="rec"):
            sig, hk_lookup = self.decompose_genome(str(record.seq).upper())
            hk_lookup_global.update(hk_lookup)
            self.save_sketches_to_one_file(sig, os.path.join(self.temp_dir, f"signatures_{self.k}_{self.n}.txt"))

    def decompose_genome(self, genome_seq):
        """
        Corrected k-mer decomposition to return the N lowest hash values 
        and their corresponding k-mer strings.
        """
        n_target = self.n  # Assuming this is 500
        hk_lookup = {}     # Mapping: numeric_hash -> kmer
        max_heap = []      # Max-heap to track the smallest N values

        # 1. Slide over the genome without pre-creating a massive list (Saves RAM)
        for i in range(len(genome_seq) - self.k + 1):
            kmer = genome_seq[i:i+self.k]
            
            # Skip k-mers with N
            if "N" in kmer:
                continue

            # 2. Compute Hash Value
            if self.hash_func == "ohe_custom":
                hash_value = self.codec.encode_with_revcomp(kmer)
            elif self.hash_func == "xxhash":
                hash_value = xxhash.xxh64(kmer, 42).intdigest()
            elif self.hash_func == "mmh3":
                import mmh3
                hash_value = mmh3.hash64(kmer, 42, signed=False)[0] 
            else:
                raise ValueError("Unsupported hash function")

            # 3. Sample-all mode
            if self.sample_all:
                if hash_value not in hk_lookup:
                    hk_lookup[hash_value] = kmer
                continue

            # 3. Heap Logic for Min-Hash (Lowest 500)
            # Only process if this hash is unique to our current set
            if hash_value not in hk_lookup:
                if len(hk_lookup) < n_target:
                    hk_lookup[hash_value] = kmer
                    heapq.heappush(max_heap, (-hash_value, hash_value))
                else:
                    # Check if the current hash is smaller than the largest in our "small set"
                    current_max_val = max_heap[0][1]
                    if hash_value < current_max_val:
                        # Remove the old maximum
                        del hk_lookup[current_max_val]
                        # Add the new smaller hash
                        hk_lookup[hash_value] = kmer
                        heapq.heapreplace(max_heap, (-hash_value, hash_value))

        # 4. Final Signatures (The actual lowest 500 hashes, sorted if not sample all)
        signatures = sorted(hk_lookup.keys())
        return signatures, hk_lookup

    def prepare_sourmash_structure(self, signatures, record_name):
        """
        Wraps signatures into a sourmash-compatible dictionary structure.
        """
        # If `signatures` is a list of signature-lists (e.g. multiple records),
        # flatten it into a single list of ints. Otherwise use as-is.
        if signatures and any(isinstance(s, (list, tuple)) for s in signatures):
            flat_sigs = [val for sub in signatures for val in sub]
        else:
            flat_sigs = list(signatures) if signatures is not None else []

        # Sort signatures to emulate a MinHash 'mins' list behavior
        sorted_sigs = sorted(flat_sigs)

        # Calculate an md5sum of the signatures to follow the format
        sig_string = ",".join(map(str, sorted_sigs))
        md5sum = hashlib.md5(sig_string.encode()).hexdigest()

        # Create the inner signature block
        sig_data = {
            "num": len(sorted_sigs),
            "ksize": self.k,
            "seed": 42, # Default for sourmash
            "max_hash": 0 , # For using num
            #"max_hash": max(sorted_sigs) if sorted_sigs else 0,
            "mins": sorted_sigs,
            "md5sum": md5sum,
            "molecule": "DNA"
        }

        # Create the outer wrapper (as a list containing one dictionary)
        full_signature = [{
            "class": "sourmash_signature",
            "email": "",
            "hash_function": f"0.{self.hash_func}", # Custom label for your reversible method
            "filename": None,
            "name": record_name,
            "license": "CC0",
            "signatures": [sig_data],
            "version": 0.4
        }]

        return full_signature
    
    def save_hk_lookup(self, hk_lookup, record_name):
        # Read existing lookup if it exists, to avoid overwriting
        existing_lookup = {}
        output_path = os.path.join(self.output_dir, f"{record_name}.json")
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                existing_lookup = json.load(f)
                # Convert keys back to integers for merging
                existing_lookup = {int(k): v for k, v in existing_lookup.items()}

        # Merge the existing lookup with the new one
        for kmer_hash, kmer in hk_lookup.items():
            if kmer_hash not in existing_lookup:
                existing_lookup[kmer_hash] = kmer

        output_path = os.path.join(self.output_dir, f"{record_name}.json")
        with open(output_path, 'w') as f:
            # JSON object keys must be strings; convert numeric hashes to strings for serialization
            serializable = {str(k): v for k, v in existing_lookup.items()}
            json.dump(serializable, f, indent=2)

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