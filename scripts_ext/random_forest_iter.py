import sourmash, os
from tqdm import tqdm
import pandas as pd
import numpy as np
from manipulations import clean_dict_keys
from io_operations import load_minhash_sketches

raw_data_path = "raw_data/"
data_prod_path = "data_prod/"
log_path = "logs/"
logfile = log_path + "random_forest_n_runs.txt"

with open(logfile, "w") as log:
    for n in [50, 500, 5000]:
        phage_minhash_data = load_minhash_sketches(data_prod_path+f"SM_sketches/PhageMinhash_n{n}_k18/", 
                                                TS=False, output_as_np=True)

        phage_minhash_data = clean_dict_keys(phage_minhash_data)

        bact_minhash_data = load_minhash_sketches(data_prod_path+f"SM_sketches/BactMinhash_n{n}_k18/", 
                                                TS=False, output_as_np=True)

        unique_minhashes = set() #for both phage and bacteria combined

        for key, val in tqdm(phage_minhash_data.items(), desc="Processing phages"):
            for minhash in val:
                unique_minhashes.add(minhash)

        for key, val in tqdm(bact_minhash_data.items(), desc="Processing bacterias"):
            for minhash in val:
                unique_minhashes.add(minhash)

        print(f"Unique minhashes extracted with len: {len(unique_minhashes)}", file=log)

        
        # ### Obtaining incidence matrix
        all_entities_minhashes = {**phage_minhash_data, **bact_minhash_data}

        # Get an ordered list of all entity names (will be the row labels)
        entity_names = sorted(list(all_entities_minhashes.keys()))

        # Get an ordered list of unique minhashes (will be the column labels)
        # Sorting is crucial to ensure consistency in the matrix columns
        sorted_minhashes = sorted(list(unique_minhashes))

        # Determine dimensions
        N = len(entity_names)  # Number of rows (entities)
        M = len(sorted_minhashes)  # Number of columns (unique minhashes)

        # Create a dictionary for quick lookup of minhash indices
        minhash_to_index = {minhash: i for i, minhash in enumerate(sorted_minhashes)}

        binary_matrix = np.zeros((N, M), dtype=int)

        # Iterate through each entity (row)
        for i, entity_name in enumerate(entity_names):
            # Get the list of minhashes for the current entity
            minhashes_present = all_entities_minhashes[entity_name]

            # Iterate through the minhashes present in the entity
            for minhash in minhashes_present:
                # Get the column index for this minhash
                j = minhash_to_index[minhash]

                # Set the corresponding cell in the matrix to 1
                binary_matrix[i, j] = 1
        
        from io_operations import call_hostrange_df
        bact_lookup, host_range_df = call_hostrange_df(raw_data_path + "phagehost_KU/Hostrange_data_all_crisp_iso.xlsx")

        from manipulations import hostrange_df_to_dict, binarize_host_range
        # Convert the host range data into a dictionary
        host_range_data = hostrange_df_to_dict(host_range_df)
        host_range_data = binarize_host_range(host_range_data, continous=False) #for classification model

        import sys
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        phage_names = phage_minhash_data.keys()
        bacteria_names = bact_minhash_data.keys()

        entity_to_index = {name: i for i, name in enumerate(entity_names)}
        minhash_to_index = {minhash: i for i, minhash in enumerate(sorted_minhashes)}

        X = []
        y = []
        rows_metadata = [] # To keep track of which entities form the row

        # Iterate through all valid phage-bacteria pairs (the required pairwise iteration)
        for bact_name in tqdm(bacteria_names, desc="Bacteria names iterated"):
            for phage_name in phage_names:
                # Get the interaction score (target variable y)
                try:
                    interaction_score = host_range_data[bact_name][phage_name]
                except KeyError:
                    continue

                # Get the feature vectors (rows from the incidence matrix)
                bact_index = entity_to_index[bact_name]
                phage_index = entity_to_index[phage_name]

                bact_features = binary_matrix[bact_index, :]
                phage_features = binary_matrix[phage_index, :]

                # Concatenate: [Bacterium Features | Phage Features]
                combined_features = np.concatenate((bact_features, phage_features))

                X.append(combined_features)
                y.append(interaction_score)
                #print(X)
                #print(y)
                rows_metadata.append((bact_name, phage_name))

        X = np.array(X)
        y = np.array(y)

        print("Unique values found in y:", set(y), file=log)
        print(f"Percent zeros in y: {round(([sum(val == 0 for val in y)][0]/len(y))*100,2)}%", file=log)

        # Check if we have enough data to proceed
        if X.shape[0] < 2:
            print(f"Error: Not enough data points ({X.shape[0]} found) for train-test split and Random Forest.", file=log)
            sys.exit(1)

        # --- Perform Random Forest Run ---

        # Generate indices to track rows through the split
        indices = np.arange(len(X))

        # Split X, y, and the original indices synchronously
        # The split arguments (random_state, stratify) MUST be identical for X/y and indices
        X_train, X_test, y_train, y_test, _, indices_test = train_test_split(
            X, y, indices, test_size=0.3, random_state=42
        )

        # Initialize and train the Random Forest Classifier
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf_classifier.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = rf_classifier.predict(X_test)

        # Calculate and report the accuracy
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n------- RESULTS FOR N={n} -------", file=log)
        print(f"Shape of the combined Feature Matrix X: {X.shape}", file=log)
        print(f"Number of target variables y: {len(y)}", file=log)
        print("\nRandom Forest Classification Results:", file=log)
        print(f"Feature Vector Size (2 * number of unique minhashes): {X.shape[1]}", file=log)
        print(f"Training set size: {X_train.shape[0]} samples", file=log)
        print(f"Testing set size: {X_test.shape[0]} samples", file=log)
        print(f"Test Accuracy: {accuracy:.4f}", file=log)

        # Create DataFrame to show test results and map back to entity names
        test_results_df = pd.DataFrame({
            'Bacterium': [rows_metadata[i][0] for i in indices_test],
            'Phage': [rows_metadata[i][1] for i in indices_test],
            'Actual_Interaction': y_test,
            'Predicted_Interaction': y_pred
        })

test_results_df.to_csv(data_prod_path+"RandomForest.csv")