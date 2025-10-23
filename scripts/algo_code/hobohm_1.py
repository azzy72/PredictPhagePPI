#18/06-2025

#For c001_ba_clean
"""
Current stats for the code: 

# Elapsed time (m): 2.851578132311503
# Amount of unique sequences: 932 out of 1147 (81.26%)
# Ratio that was filtered away: 18.74%

"""
#When running the C++ code with a different homology score: 

"""

# Amount of unique sequences: 909 out of 1147 (79.25%)
# Ratio that was filtered away: 20.75%

"""




#Important note: If the code runs rapidly, it means that it filtered too many sequences. And the thresholds should preferably be adjusted! 


gap_open = -3                   # Default = -3
gap_extension = -1              # Default = -1

norm_alignment_threshold = 0.6  # Normalized alignment score | Default 0.7

percent_identity_threshold = 0.8 # Note: No longer relevant, since we mainly use normalized alignment score.

blosum_number = "62" #"62" or "50"



#Input (example): raw_data/A0101/c000_ba_clean 
#Output: data/raw_data/A0101/c000_ba_clean_hobohm1 | Do this for each allele. Filtered and should have less sequences. 

import numpy as np
import argparse
from time import time


#Important: CD to "code" folder 

# python3 hobohm_1.py -f ../data/raw_data/A0101/c000_ba_clean -o ../data/raw_data/A0101


"""
optional arguments:
  -h, --help         show this help message and exit
  -f database_file  File input data
  -o output_path    The path to the output folder
"""

parser = argparse.ArgumentParser(description="Hobohm1")
parser.add_argument('-f', '--query', dest='database_file', action="store", required=True, help='File input data')
parser.add_argument('-o', '--outpath', dest="output_path", action="store", help='The path to the output folder')
args = parser.parse_args()

database_file = args.database_file
output_path = args.output_path



###

alphabet_file = "../alphabet" #File should be in the GitHub folder. 
alphabet = np.loadtxt(alphabet_file, dtype=str)

###

blosum_file = f"../BLOSUM{blosum_number}_values" #File should be in the GitHub folder. 
_blosum = np.loadtxt(blosum_file, dtype=int).T

blosum = {}

for i, letter_1 in enumerate(alphabet):
    
    blosum[letter_1] = {}

    for j, letter_2 in enumerate(alphabet):
        
        blosum[letter_1][letter_2] = _blosum[i, j]

###

# ### Sequences


#Note: The data structure in the old file, database_list.tab, was:
#Protein Id \t Sequence

def load_sequences():
    
    database_list = np.loadtxt(database_file, dtype=str).reshape(-1,2)

    # ids = database_list[:, 0]
    # sequences = database_list[:, 1]
    
    sequences = database_list[:, 0]
    scores = database_list[:, 1]
    
    return sequences, scores


# ## Smith-Waterman O2

def smith_waterman(query, database, scoring_scheme, gap_open, gap_extension):
    
    P_matrix, Q_matrix, D_matrix, E_matrix, i_max, j_max, max_score = smith_waterman_alignment(query, database, scoring_scheme, gap_open, gap_extension)
    
    aligned_query, aligned_database, matches = smith_waterman_traceback(E_matrix, D_matrix, i_max, j_max, query, database, gap_open, gap_extension)
    
    return aligned_query, aligned_database, matches, max_score #It now also returns the "max_score" which is corresponding to the "alignment_score". 


def smith_waterman_alignment(query, database, scoring_scheme, gap_open, gap_extension):

    # Matrix imensions
    M = len(query)
    N = len(database)
    
    # D matrix change to float
    D_matrix = np.zeros((M+1, N+1), int)

    # P matrix
    P_matrix = np.zeros((M+1, N+1), int)
    
    # Q matrix
    Q_matrix = np.zeros((M+1, N+1), int)

    # E matrix
    E_matrix = np.zeros((M+1, N+1), dtype=object)

    # Main loop
    D_matrix_max_score, D_matrix_i_max, D_matrix_i_max = -9, -9, -9
    for i in range(M-1, -1, -1):
        for j in range(N-1, -1, -1):
            
            # Q_matrix[i,j] entry
            gap_open_database = D_matrix[i+1,j] + gap_open
            gap_extension_database = Q_matrix[i+1,j] + gap_extension
            max_gap_database = max(gap_open_database, gap_extension_database)
            
            Q_matrix[i,j] = max_gap_database
                
            # P_matrix[i,j] entry
            gap_open_query = D_matrix[i,j+1] + gap_open
            gap_extension_query = P_matrix[i,j+1] + gap_extension
            max_gap_query = max(gap_open_query, gap_extension_query)
            
            P_matrix[i,j] = max_gap_query
            
            # D_matrix[i,j] entry
            diagonal_score = D_matrix[i+1,j+1] + scoring_scheme[query[i]][database[j]]    
            
            # E_matrix[i,j] entry
            candidates = [(1, diagonal_score),
                          (2, gap_open_database),
                          (4, gap_open_query),
                          (3, gap_extension_database),
                          (5, gap_extension_query)]
            
            direction, max_score = max(candidates, key=lambda x: x[1])
            
            
            # check entry sign
            if max_score > 0:
                E_matrix[i,j] = direction
            else:
                E_matrix[i,j] = 0
            
            # check max score sign
            if max_score > 0:
                D_matrix[i, j] = max_score
            else:
                D_matrix[i, j] = 0

            # fetch global max score
            if max_score > D_matrix_max_score:
                D_matrix_max_score = max_score
                D_matrix_i_max = i
                D_matrix_j_max = j
            
    return P_matrix, Q_matrix, D_matrix, E_matrix, D_matrix_i_max, D_matrix_j_max, D_matrix_max_score


def smith_waterman_traceback(E_matrix, D_matrix, i_max, j_max, query, database, gap_open, gap_extension):
    # Matrix imensions
    M = len(query)
    N = len(database)

    # aligned query string
    aligned_query = []
    
    # aligned database string
    aligned_database = []
    
    # total identical matches
    matches = 0

        
    # start from max_i, max_j
    i, j = i_max, j_max
    while i < M and j < N:

        # E[i,j] = 0, stop back tracking
        if E_matrix[i, j] == 0:
            break
        
        # E[i,j] = 1, match
        if E_matrix[i, j] == 1:
            aligned_query.append(query[i])
            aligned_database.append(database[j])
            if ( query[i] == database[j]):
                matches += 1
            i += 1
            j += 1
        
        
        # E[i,j] = 2, gap opening in database
        if E_matrix[i, j] == 2:
            aligned_database.append("-")
            aligned_query.append(query[i])
            i += 1

            
        # E[i,j] = 3, gap extension in database
        if E_matrix[i, j] == 3:
                   
            count = i + 2
            score = D_matrix[count, j] + gap_open + gap_extension
            
            # Find length of gap
            while((score - D_matrix[i, j])*(score - D_matrix[i, j]) >= 0.00001):   
                count += 1
                score = D_matrix[count, j] + gap_open + (count-i-1)*gap_extension

            for k in range(i, count):
                aligned_database.append("-")
                aligned_query.append(query[i])
                i += 1
            
            
        # E[i,j] = 4, gap opening in query
        if E_matrix[i, j] == 4:
            aligned_query.append("-")
            aligned_database.append(database[j])
            j += 1
        
        
        # E[i,j] = 5, gap extension in query
        if E_matrix[i, j] == 5:
             
            count = j + 2
            score = D_matrix[i, count] + gap_open + gap_extension
            
            # Find length of gap
            while((score - D_matrix[i, j])*(score - D_matrix[i, j]) >= 0.0001): 
                count += 1
                score = D_matrix[i, count] + gap_open + (count-j-1)*gap_extension

            for k in range(j, count):
                aligned_query.append("-")
                aligned_database.append(database[j])
                j += 1

                
    return aligned_query, aligned_database, matches


# ## Hobohm 1

# ### Similarity Function
# 
# ### This code defines the threshold for similarity

def percent_identity(alignment_length, matches):
    
    if alignment_length == 0:
        return "keep", 0 #No overlay in alignment - it is super unique. 
        #Previously had a zero division error. 

    percent_identity = matches / alignment_length

    if percent_identity >= percent_identity_threshold:  # e.g., 80% identity threshold
        return "discard", percent_identity #If it is too similar - discard it. 
    else:
        return "keep", percent_identity #Not similar - keep it. 



def norm_alignment_score(query, database, alignment_score):
    
    query_self_score = 0

    for res1, res2 in zip(query,query): #The max score is the blosum substitution scoring, when the Blosom is compared to itself. 
        query_self_score += scoring_scheme[res1][res2] #E.g. substituting T -> T (query compared to itself). 
    
    database_self_score = 0

    for res1, res2 in zip(database,database):
        database_self_score += scoring_scheme[res1][res2]
    

    #Normalize by the Average of Self-Scores: 
    norm_alignment_score = alignment_score/(np.sqrt(query_self_score*database_self_score)) #alignment_score/sqrt(query_self_score*database_self_score)

    #Note - there is also another variant where you pick the shortest sequence:
    # alignment_score / <shortest sequence>_self_score
    #Then a threhold of 0.5 would be appropriate. 

    
    if norm_alignment_score >= norm_alignment_threshold: 
        return "discard", norm_alignment_score #Too similar - should be discarded.
    else:
        return "keep", norm_alignment_score #Not similar - should be added to accepted, unique sequences.



###

candidate_sequences, candidate_scores = load_sequences() #candidate_scores is new. 

print ("# Number of elements:", len(candidate_sequences))

accepted_sequences, accepted_scores = [], []

accepted_sequences.append(candidate_sequences[0]) #In Hobohm 1, we always start by approving the very first sequence, as being unique. 
accepted_scores.append(candidate_scores[0])


N = len(candidate_sequences)

print(f"(#) Unique. \t| First sequence is always unique \t- (1/{N})")


###


# parameters
scoring_scheme = blosum
#gap_open = -11 #Maybe the gap opening penalty should be less severe for MHC binding sequences (???)

t0 = time()

for i in range(1, N):
    for j in range(0, len(accepted_sequences)):
        query = candidate_sequences[i]
        database = accepted_sequences[j]
        
        # Before we can do similarity filtering with Hobohm-1, 
        # we first need to align the sequences:
        aligned_query, aligned_database, matches, alignment_score = smith_waterman(query, database, scoring_scheme, gap_open, gap_extension)
        alignment_length = len(aligned_query)

        # And then subsequently score the similarity via a homology score: 
        
        #One way to examine homology, is to use the percent identity:
        percent_identity_outcome, percent_identity_score = percent_identity(alignment_length, matches)
        
        norm_alignment_outcome, norm_alignment_val = norm_alignment_score(query, database,alignment_score)


        # If query (non-accepted sequence) is not sufficiently unique (relative to the accepted sequences) - then Hobohm1 will discard it:
        #if percent_identity_outcome == "discard" or norm_alignment_outcome == "discard":
        if norm_alignment_outcome == "discard":
            print (f"(!) Not uniq. \t| {query} is homolog to {database}|\t ({i+1}/{N})\n\t% identity: {round(percent_identity_score,2)} | norm alignment: {round(norm_alignment_val,2)}")
            break
            
    # Query (non-accepted sequence) is unique (compared to accepted sequences) - Hobohm1 adds it to the accepted sequence dataset:
    #if percent_identity_outcome == "keep" and norm_alignment_outcome == "keep":
    if norm_alignment_outcome == "keep":
        accepted_sequences.append(candidate_sequences[i])
        accepted_scores.append(candidate_scores[i])
        print(f"(#) Unique. \t| {query} is NOT homolog to {database} |\t ({i+1}/{N})\n\t% identity: {round(percent_identity_score,2)} | norm alignment: {round(norm_alignment_val,2)}")


print()
print("#"*50)
print()

t1 = time()

print ("Elapsed time (m):", (t1-t0)/60)
print(f"Amount of unique sequences: {len(accepted_scores)} out of {N} ({round((len(accepted_scores)/N)*100,2)}%)")
print(f"Ratio that was filtered away: {round((1-len(accepted_scores)/N)*100,2)}%")

###

"""
Saves the parsed results to the specified output file.
"""

input_file_name = database_file.split("/")[-1] # raw_data/A0101/c000_ba_clean  --> c000_ba_clean
output_file_name = input_file_name + "_hobohm1"


with open(output_path+f"/{output_file_name}", 'w') as file:
    for sequence, score in zip(accepted_sequences, accepted_scores):
        file.write(f"{sequence} {score}\n")

print()
print(f"Saved the results as: {output_file_name} | At this path: {output_path}")



