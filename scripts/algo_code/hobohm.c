#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <libgen.h> // For basename and dirname

// Define constants
#define MAX_LINE_LENGTH 1024
#define MAX_SEQUENCES 100000 // Adjust as needed for larger datasets
#define ALPHABET_SIZE 20 // Number of amino acids in your BLOSUM50 alphabet
#define HOMOLOGY_THRESHOLD 0.5 // Constant for the 0.7 threshold

// Global lookup table for characters to integer indices
int char_to_int_map[128]; // ASCII values for characters
char int_to_char_map[ALPHABET_SIZE]; // For debugging/display, mapping int index back to char

// Global BLOSUM50 scoring matrix
int blosum50_matrix[ALPHABET_SIZE][ALPHABET_SIZE];

// Function prototypes
void initialize_char_to_int_map();
int load_alphabet(const char* filename);
int load_blosum50(const char* filename);
int load_sequences(const char* filename, char*** sequences, double** scores, int* num_sequences);
void free_sequences(char** sequences, int num_sequences);
int char_to_int(char c);

// Smith-Waterman functions
void smith_waterman_alignment(const char* query, const char* database, int gap_open, int gap_extension,
                              int*** P_matrix_ptr, int*** Q_matrix_ptr, int*** D_matrix_ptr, int*** E_matrix_ptr,
                              int* i_max, int* j_max, int* max_score);
// Added aligned_length_ptr to traceback signature
void smith_waterman_traceback(int** E_matrix, int** D_matrix, int i_max, int j_max,
                              const char* query, const char* database, int gap_open, int gap_extension,
                              char** aligned_query, char** aligned_database, int* matches, int* aligned_length_ptr);
void free_matrices(int** P_matrix, int** Q_matrix, int** D_matrix, int** E_matrix, int M, int N);

// Homology function - now accepts aligned_length
char* homology_function(int matches, int original_query_length, int original_database_length, int aligned_length, double* homology_ratio);

int main(int argc, char* argv[]) {
    char* database_file = NULL;
    char* output_path = NULL;

    // Argument parsing (simple manual parsing for two arguments)
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--query") == 0) {
            if (i + 1 < argc) {
                database_file = argv[++i];
            } else {
                fprintf(stderr, "Error: -f/--query requires an argument.\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--outpath") == 0) {
            if (i + 1 < argc) {
                output_path = argv[++i];
            } else {
                fprintf(stderr, "Error: -o/--outpath requires an argument.\n");
                return 1;
            }
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            fprintf(stderr, "Usage: %s -f <database_file> -o <output_path>\n", argv[0]);
            return 1;
        }
    }

    if (database_file == NULL) {
        fprintf(stderr, "Error: Database file (-f/--query) is required.\n");
        fprintf(stderr, "Usage: %s -f <database_file> -o <output_path>\n", argv[0]);
        return 1;
    }
    if (output_path == NULL) {
        fprintf(stderr, "Error: Output path (-o/--outpath) is required.\n");
        fprintf(stderr, "Usage: %s -f <database_file> -o <output_path>\n", argv[0]);
        return 1;
    }

    initialize_char_to_int_map();

    // Load alphabet and BLOSUM50
    char alphabet_file[] = "../alphabet"; // Assuming relative path from executable
    char blosum_file[] = "../BLOSUM50_values"; // Assuming relative path from executable

    if (load_alphabet(alphabet_file) != 0) {
        return 1; // Error during alphabet loading
    }
    if (load_blosum50(blosum_file) != 0) {
        return 1; // Error during BLOSUM50 loading
    }

    char** candidate_sequences = NULL;
    double* candidate_scores = NULL;
    int num_candidate_sequences = 0;

    if (load_sequences(database_file, &candidate_sequences, &candidate_scores, &num_candidate_sequences) != 0) {
        return 1; // Error during sequence loading
    }

    printf("# Number of elements: %d\n", num_candidate_sequences);

    char** accepted_sequences = (char**)malloc(sizeof(char*) * num_candidate_sequences);
    double* accepted_scores = (double*)malloc(sizeof(double) * num_candidate_sequences);
    int num_accepted_sequences = 0;

    if (num_candidate_sequences > 0) {
        accepted_sequences[num_accepted_sequences] = strdup(candidate_sequences[0]);
        accepted_scores[num_accepted_sequences] = candidate_scores[0];
        num_accepted_sequences++;
        printf("(#) Unique. \t| First sequence is always unique \t- (1/%d)\n", num_candidate_sequences);
    } else {
        printf("No sequences to process.\n");
        free_sequences(candidate_sequences, num_candidate_sequences);
        free(accepted_sequences);
        free(accepted_scores);
        return 0;
    }

    // Smith-Waterman parameters
    int gap_open = -11;
    int gap_extension = -1;

    clock_t t0 = clock();

    for (int i = 1; i < num_candidate_sequences; i++) {
        char* current_query = candidate_sequences[i];
        int discard = 0; // Flag to indicate if current_query should be discarded

        for (int j = 0; j < num_accepted_sequences; j++) {
            char* current_database = accepted_sequences[j];

            // Declare matrix pointers as int**
            int** P_matrix = NULL;
            int** Q_matrix = NULL;
            int** D_matrix = NULL;
            int** E_matrix = NULL;
            int i_max_align, j_max_align, max_score_align;

            // Pass address of matrix pointers (int***)
            smith_waterman_alignment(current_query, current_database, gap_open, gap_extension,
                                     &P_matrix, &Q_matrix, &D_matrix, &E_matrix,
                                     &i_max_align, &j_max_align, &max_score_align);

            char* aligned_query_str = NULL;
            char* aligned_database_str = NULL;
            int matches = 0;
            int aligned_length = 0; // New variable to store the aligned length

            // Pass matrix pointers (int**) directly, and address of aligned_length
            smith_waterman_traceback(E_matrix, D_matrix, i_max_align, j_max_align,
                                     current_query, current_database, gap_open, gap_extension,
                                     &aligned_query_str, &aligned_database_str, &matches, &aligned_length);
            
            // Free matrices after traceback
            free_matrices(P_matrix, Q_matrix, D_matrix, E_matrix, strlen(current_query), strlen(current_database));

            double homology_ratio;
            
            // Pass aligned_length to homology function
            char* homology_outcome = homology_function(matches, strlen(current_query), strlen(current_database), aligned_length, &homology_ratio);

            if (strcmp(homology_outcome, "discard") == 0) {
                printf("(!) Discarded. \t| Seq %d (input score: %.6f) is similar to seq %d | Match Ratio: %.2f | Matches: %d | Orig Len Q: %zu | Orig Len D: %zu | Aligned Len: %d \t- (%d/%d)\n",
                        i + 1, candidate_scores[i], j + 1, homology_ratio, matches, strlen(current_query), strlen(current_database), aligned_length, i + 1, num_candidate_sequences);
                discard = 1;
                free(aligned_query_str);
                free(aligned_database_str);
                break; // Break inner loop, as it's already discarded
            }
            free(aligned_query_str);
            free(aligned_database_str);
        }

        if (!discard) {
            accepted_sequences[num_accepted_sequences] = strdup(current_query);
            accepted_scores[num_accepted_sequences] = candidate_scores[i];
            num_accepted_sequences++;
            // Print a single message when a sequence is truly accepted as unique
            printf("(#) Unique. \t| Seq %d (input score: %.6f) added as unique\t\t- (%d/%d)\n",
                   i + 1, candidate_scores[i], i + 1, num_candidate_sequences);
        }
    }

    printf("\n");
    printf("##################################################\n");
    printf("\n");

    clock_t t1 = clock();
    double elapsed_time_sec = (double)(t1 - t0) / CLOCKS_PER_SEC;
    printf("Elapsed time (m): %.2f\n", elapsed_time_sec / 60.0);
    printf("Amount of unique sequences: %d out of %d\n", num_accepted_sequences, num_candidate_sequences);

    // Save results
    char* basec = strdup(database_file); // Make a copy because basename might modify its argument
    char* input_file_name = basename(basec);
    
    char output_file_name[MAX_LINE_LENGTH];
    snprintf(output_file_name, MAX_LINE_LENGTH, "%s/%s_hobohm1", output_path, input_file_name);

    FILE* output_fp = fopen(output_file_name, "w");
    if (output_fp == NULL) {
        perror("Error opening output file");
        free_sequences(candidate_sequences, num_candidate_sequences);
        for (int k = 0; k < num_accepted_sequences; k++) {
            free(accepted_sequences[k]);
        }
        free(accepted_sequences);
        free(accepted_scores);
        free(basec);
        return 1;
    }

    for (int k = 0; k < num_accepted_sequences; k++) {
        fprintf(output_fp, "%s %.6f\n", accepted_sequences[k], accepted_scores[k]);
    }
    fclose(output_fp);

    printf("\n");
    printf("Saved the results as: %s_hobohm1 | At this path: %s\n", input_file_name, output_path);

    // Clean up
    free_sequences(candidate_sequences, num_candidate_sequences);
    for (int k = 0; k < num_accepted_sequences; k++) {
        free(accepted_sequences[k]);
    }
    free(accepted_sequences);
    free(accepted_scores);
    free(basec); // Free the duplicated string

    return 0;
}

// Initializes the global char_to_int_map
void initialize_char_to_int_map() {
    for (int i = 0; i < 128; i++) {
        char_to_int_map[i] = -1; // Initialize all to -1 (invalid)
    }
}

// Loads alphabet characters from file and populates char_to_int_map
int load_alphabet(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("Error opening alphabet file");
        return 1;
    }

    char line[MAX_LINE_LENGTH];
    int index = 0;
    while (fgets(line, sizeof(line), fp) != NULL) {
        // Remove newline character if present
        line[strcspn(line, "\n")] = 0;
        if (strlen(line) == 1) { // Ensure it's a single character
            if (index < ALPHABET_SIZE) {
                char_to_int_map[(int)line[0]] = index;
                int_to_char_map[index] = line[0];
                index++;
            } else {
                fprintf(stderr, "Warning: More characters in alphabet file than ALPHABET_SIZE (%d). Ignoring '%s'.\n", ALPHABET_SIZE, line);
            }
        }
    }
    fclose(fp);

    if (index != ALPHABET_SIZE) {
        fprintf(stderr, "Warning: Alphabet file has %d entries, expected %d. This might lead to issues with BLOSUM50.\n", index, ALPHABET_SIZE);
    }
    return 0;
}

// Loads BLOSUM50 matrix from file
int load_blosum50(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("Error opening BLOSUM50 file");
        return 1;
    }

    for (int i = 0; i < ALPHABET_SIZE; i++) {
        for (int j = 0; j < ALPHABET_SIZE; j++) {
            if (fscanf(fp, "%d", &blosum50_matrix[i][j]) != 1) {
                fprintf(stderr, "Error reading BLOSUM50 matrix at row %d, col %d. Check if file has %d*%d integer values.\n", i, j, ALPHABET_SIZE, ALPHABET_SIZE);
                fclose(fp);
                return 1;
            }
        }
    }
    // Check for extra data in BLOSUM50 file
    int temp_val;
    if (fscanf(fp, "%d", &temp_val) == 1) {
        fprintf(stderr, "Warning: Extra data found in BLOSUM50 file after %d*%d matrix. This might indicate incorrect file format.\n", ALPHABET_SIZE, ALPHABET_SIZE);
    }
    fclose(fp);
    return 0;
}

// Converts a character to its corresponding integer index using the global map
int char_to_int(char c) {
    int index = char_to_int_map[(int)c];
    if (index == -1) {
        fprintf(stderr, "Error: Character '%c' not found in alphabet map. Using 0 (check alphabet and BLOSUM50 files).\n", c);
        return 0; // Return 0 or handle error appropriately
    }
    return index;
}


// Loads sequences and their scores from the database file
int load_sequences(const char* filename, char*** sequences, double** scores, int* num_sequences) {
    FILE* fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("Error opening database file");
        return 1;
    }

    *sequences = (char**)malloc(sizeof(char*) * MAX_SEQUENCES);
    *scores = (double*)malloc(sizeof(double) * MAX_SEQUENCES);
    if (*sequences == NULL || *scores == NULL) {
        perror("Memory allocation failed for sequences/scores");
        fclose(fp);
        return 1;
    }

    char line[MAX_LINE_LENGTH];
    int count = 0;
    while (fgets(line, sizeof(line), fp) != NULL && count < MAX_SEQUENCES) {
        char seq[MAX_LINE_LENGTH];
        double score;

        // Remove newline character if present
        line[strcspn(line, "\n")] = 0;

        // Using sscanf to parse "SEQUENCE SCORE" format
        if (sscanf(line, "%s %lf", seq, &score) == 2) {
            (*sequences)[count] = strdup(seq); // Duplicate string to store
            if ((*sequences)[count] == NULL) {
                perror("Memory allocation failed for sequence string");
                // Clean up already allocated memory before returning
                for(int i = 0; i < count; ++i) free((*sequences)[i]);
                free(*sequences);
                free(*scores);
                fclose(fp);
                return 1;
            }
            (*scores)[count] = score;
            count++;
        } else {
            fprintf(stderr, "Warning: Could not parse line '%s'. Skipping.\n", line);
        }
    }
    fclose(fp);
    *num_sequences = count;
    return 0;
}

// Frees memory allocated for sequences
void free_sequences(char** sequences, int num_sequences) {
    if (sequences != NULL) {
        for (int i = 0; i < num_sequences; i++) {
            free(sequences[i]);
        }
        free(sequences);
    }
}

// Allocates and initializes a 2D integer matrix
int** create_matrix(int rows, int cols) {
    int** matrix = (int**)malloc(rows * sizeof(int*));
    if (matrix == NULL) {
        perror("Memory allocation failed for matrix rows");
        return NULL;
    }
    for (int i = 0; i < rows; i++) {
        matrix[i] = (int*)calloc(cols, sizeof(int)); // calloc initializes to 0
        if (matrix[i] == NULL) {
            perror("Memory allocation failed for matrix columns");
            // Free previously allocated rows
            for (int k = 0; k < i; k++) free(matrix[k]);
            free(matrix);
            return NULL;
        }
    }
    return matrix;
}

// Frees dynamically allocated matrices
void free_matrices(int** P_matrix, int** Q_matrix, int** D_matrix, int** E_matrix, int M, int N) {
    // M and N here refer to the original sequence lengths, so the matrices have M+1 rows and N+1 columns
    if (P_matrix) {
        for (int i = 0; i <= M; i++) free(P_matrix[i]);
        free(P_matrix);
    }
    if (Q_matrix) {
        for (int i = 0; i <= M; i++) free(Q_matrix[i]);
        free(Q_matrix);
    }
    if (D_matrix) {
        for (int i = 0; i <= M; i++) free(D_matrix[i]);
        free(D_matrix);
    }
    if (E_matrix) {
        for (int i = 0; i <= M; i++) free(E_matrix[i]);
        free(E_matrix);
    }
}

// Implements the Smith-Waterman alignment matrix filling
void smith_waterman_alignment(const char* query, const char* database, int gap_open, int gap_extension,
                              int*** P_matrix_ptr, int*** Q_matrix_ptr, int*** D_matrix_ptr, int*** E_matrix_ptr,
                              int* i_max, int* j_max, int* max_score_val) {
    int M = strlen(query);
    int N = strlen(database);

    // Allocate matrices (+1 for padding)
    *D_matrix_ptr = create_matrix(M + 1, N + 1);
    *P_matrix_ptr = create_matrix(M + 1, N + 1);
    *Q_matrix_ptr = create_matrix(M + 1, N + 1);
    *E_matrix_ptr = create_matrix(M + 1, N + 1); // E_matrix stores direction (int)

    if (*D_matrix_ptr == NULL || *P_matrix_ptr == NULL || *Q_matrix_ptr == NULL || *E_matrix_ptr == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for Smith-Waterman matrices.\n");
        // Proper error handling would involve freeing partially allocated memory here
        exit(1);
    }
    
    int** D_matrix = *D_matrix_ptr;
    int** P_matrix = *P_matrix_ptr;
    int** Q_matrix = *Q_matrix_ptr;
    int** E_matrix = *E_matrix_ptr;

    *max_score_val = 0; // Initialize max score to 0 as scores cannot be negative
    *i_max = 0;
    *j_max = 0;

    // Main loop: iterate from M-1 down to 0, and N-1 down to 0
    for (int i = M - 1; i >= 0; i--) {
        for (int j = N - 1; j >= 0; j--) {
            // Q_matrix[i,j] entry (gap in database, moving right in query)
            int gap_open_database = D_matrix[i + 1][j] + gap_open;
            int gap_extension_database = Q_matrix[i + 1][j] + gap_extension;
            Q_matrix[i][j] = (gap_open_database > gap_extension_database) ? gap_open_database : gap_extension_database;

            // P_matrix[i,j] entry (gap in query, moving down in database)
            int gap_open_query = D_matrix[i][j + 1] + gap_open;
            int gap_extension_query = P_matrix[i][j + 1] + gap_extension;
            P_matrix[i][j] = (gap_open_query > gap_extension_query) ? gap_open_query : gap_extension_query;

            // D_matrix[i,j] entry (match/mismatch or gap from P/Q)
            int diagonal_score = D_matrix[i + 1][j + 1] + blosum50_matrix[char_to_int(query[i])][char_to_int(database[j])];

            // Replicating Python's direction logic more closely:
            // Python's candidates: [(1, diagonal_score), (2, gap_open_database), (4, gap_open_query), (3, gap_extension_database), (5, gap_extension_query)]
            int candidates_scores[] = {
                diagonal_score,
                gap_open_database,
                gap_open_query,
                gap_extension_database,
                gap_extension_query
            };
            int candidates_directions[] = {1, 2, 4, 3, 5}; // Corresponding directions for traceback

            int current_max_score = -999999; // Sufficiently small number
            int direction = 0; // Default to 0 for no path (if all scores are negative)

            for(int k=0; k<5; ++k) {
                if (candidates_scores[k] > current_max_score) {
                    current_max_score = candidates_scores[k];
                    direction = candidates_directions[k];
                }
            }

            // check entry sign (Smith-Waterman condition: score cannot be negative)
            if (current_max_score > 0) {
                E_matrix[i][j] = direction;
                D_matrix[i][j] = current_max_score;
            } else {
                E_matrix[i][j] = 0;
                D_matrix[i][j] = 0;
            }

            // fetch global max score
            if (D_matrix[i][j] > *max_score_val) {
                *max_score_val = D_matrix[i][j];
                *i_max = i;
                *j_max = j;
            }
        }
    }
}


// Implements the Smith-Waterman traceback
// Added aligned_length_ptr to store the length of the aligned strings
void smith_waterman_traceback(int** E_matrix, int** D_matrix, int i_max, int j_max,
                              const char* query, const char* database, int gap_open, int gap_extension,
                              char** aligned_query, char** aligned_database, int* matches, int* aligned_length_ptr) {
    int M = strlen(query);
    int N = strlen(database);

    // Max possible length for aligned sequences (sum of lengths + 1 for null terminator)
    int max_aligned_len = M + N + 1;
    *aligned_query = (char*)malloc(max_aligned_len * sizeof(char));
    *aligned_database = (char*)malloc(max_aligned_len * sizeof(char));
    if (*aligned_query == NULL || *aligned_database == NULL) {
        perror("Memory allocation failed for aligned strings");
        exit(1);
    }
    (*aligned_query)[0] = '\0'; // Initialize as empty strings
    (*aligned_database)[0] = '\0';
    *matches = 0;

    char temp_aligned_query[max_aligned_len]; // Use temporary buffers for building strings in reverse
    char temp_aligned_database[max_aligned_len];
    int temp_idx = 0; // This will store the length of the aligned segment

    int i = i_max;
    int j = j_max;

    // Start traceback from the cell with the maximum score
    // Continue as long as the cell score is positive (in D_matrix) or a direction is indicated (in E_matrix)
    // and we are within bounds. The condition `E_matrix[i][j] == 0` is crucial for stopping.
    while (i < M && j < N && D_matrix[i][j] > 0 && E_matrix[i][j] != 0) {
        int direction = E_matrix[i][j];

        if (direction == 1) { // Diagonal (match/mismatch)
            temp_aligned_query[temp_idx] = query[i];
            temp_aligned_database[temp_idx] = database[j];
            if (query[i] == database[j]) {
                (*matches)++;
            }
            i++;
            j++;
        } else if (direction == 2 || direction == 3) { // Gap in database (deletion in query)
            temp_aligned_query[temp_idx] = query[i];
            temp_aligned_database[temp_idx] = '-';
            i++;
        } else if (direction == 4 || direction == 5) { // Gap in query (insertion in database)
            temp_aligned_query[temp_idx] = '-';
            temp_aligned_database[temp_idx] = database[j];
            j++;
        }
        temp_idx++;
    }

    temp_aligned_query[temp_idx] = '\0';
    temp_aligned_database[temp_idx] = '\0';

    // Reverse the strings
    int len_aligned = temp_idx; // Capture the length of the aligned segment
    for (int k = 0; k < len_aligned; k++) {
        (*aligned_query)[k] = temp_aligned_query[len_aligned - 1 - k];
        (*aligned_database)[k] = temp_aligned_database[len_aligned - 1 - k];
    }
    (*aligned_query)[len_aligned] = '\0';
    (*aligned_database)[len_aligned] = '\0';

    *aligned_length_ptr = len_aligned; // Return the aligned length
}

// Determines if sequences are homologous based on a similarity threshold
char* homology_function(int matches, int original_query_length, int original_database_length, int aligned_length, double* homology_ratio) {
    // New total_alignment_length as sum of original lengths minus the length of the local alignment
    int total_alignment_length = original_query_length + original_database_length - aligned_length;

    // Handle edge cases for the denominator
    // If aligned_length is 0 (no significant local alignment found), total_alignment_length will be L1 + L2.
    // If L1 + L2 - aligned_length results in 0 (e.g., L1=5, L2=5, aligned_length=10 - impossible,
    // or L1=0, L2=0, aligned_length=0), treat as no similarity.
    if (total_alignment_length <= 0) { // Should not be negative with proper aligned_length, but guarding
        *homology_ratio = 0.0;
        return "keep"; // If denominator is problematic, no meaningful comparison, so "keep" as unique.
    }

    // Calculate the match ratio
    *homology_ratio = (double)matches / total_alignment_length;

    // Compare with the defined threshold
    if (*homology_ratio > HOMOLOGY_THRESHOLD) {
        return "discard";
    } else {
        return "keep";
    }
}