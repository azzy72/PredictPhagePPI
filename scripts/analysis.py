########################
#####  Analysis.py  ####
########################
# Contains functions for analysis and plotting

##### Imports -----------
import pandas as pd
from pathlib import Path
import os
import json
from Bio.Blast import NCBIWWW, NCBIXML
from Bio import Entrez, SeqIO
from io import StringIO
import shap
from torch import embedding
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, precision_recall_curve, average_precision_score, confusion_matrix
from time import time, sleep
from datetime import datetime
from datetime import datetime
import networkx as nx
from scipy.cluster.hierarchy import linkage
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from captum import attr 
from captum.attr import IntegratedGradients
from decompositions import KmerCodec
from paths import raw_data_path, data_prod_path


def perform_pca(data: pd.DataFrame, n_components=2):
    """
    Perform PCA on the given DataFrame.

    Args:
        data (pd.DataFrame): Input data for PCA.
        n_components (int): Number of principal components to compute. Default is 2.
    
    Returns:
        pca (PCA): Fitted PCA object.
        score (np.array): PCA scores (data points/samples).
        coeff (np.array): PCA components (eigenvectors/feature directions).
    """
    # data input control if empty
    if data.empty:
        raise ValueError("Input data DataFrame is empty.")
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Perform PCA
    pca = PCA(n_components=n_components)
    score = pca.fit_transform(data_scaled)
    coeff = pca.components_.T
    return pca, score, coeff

def pca_biplot(score = None, coeff = None, PCA = None, data : pd.DataFrame = None, vec_labels = None, point_labels = None, n_features_to_plot = 10, hide_labels = [False, False], color_on = None) -> None:
    """
    Generates a PCA biplot with dynamically scaled arrows for loadings and 
    uses adjust_text to prevent label overlap.

    Args:
        score (np.array): PCA scores (data points/samples).
        coeff (np.array): PCA components (eigenvectors/feature directions).
        PCA (PCA): Fitted PCA object from sklearn. If provided, 'score' and 'coeff' must also be provided.
        data (pd.DataFrame): Input data for PCA. If provided, PCA will be computed
        vec_labels (list): Feature labels on eigenvectors (e.g., k-mers).
        point_labels (list): Sample labels on points (e.g., sequence IDs).
        n_features_to_plot (int): Maximum number of features (arrows) to display.
        hide_labels (list of bool): [hide_score_labels, hide_loading_labels]. If True, hides the respective labels.
        color_on (list): Optional list of labels for coloring the points. If None, coloring is based on unique labels.
    
    Returns:
        None (displays the plot)
    """
    ### Input validation and setup
    #generate generic vector labels
    if vec_labels is None and coeff is not None:
        vec_labels = [f'Feature_{i+1}' for l in range(coeff.shape[0])]
    elif vec_labels is None and data is not None:
        vec_labels = data.columns.tolist()
    
    #generate generic point labels
    if point_labels is None and score is not None:
        point_labels = [f'Sample_{i+1}' for i in range(score.shape[0])]
    elif point_labels is None and data is not None:
        point_labels = data.index.tolist()
    
    if data is None and PCA is None:
        raise ValueError("Either 'data' or 'PCA' must be provided.")
    
    if data is not None and PCA is None: #construct PCA from data
        pca, score, coeff = perform_pca(data)
        components_df = pd.DataFrame(score, columns=[f'PC{i+1}' for i in range(score.shape[1])], index=data.index)
    
    elif data is None and PCA is not None: #use provided PCA
        pca = PCA
        components_df = pd.DataFrame(score, columns=[f'PC{i+1}' for i in range(score.shape[1])])
        if score is None or coeff is None:
            raise ValueError("When providing 'PCA', both 'score' and 'coeff' must also be provided.")

    xs = score[:, 0]
    ys = score[:, 1]
    
    # 1. Calculate a dynamic scaling factor for the loading arrows
    # This factor ensures the longest arrow is proportional to the max spread of the scores.
    max_coeff_length = np.max(np.sqrt(coeff[:, 0]**2 + coeff[:, 1]**2))
    max_score_extent = np.max(np.abs(score))
    
    # Aim for the longest arrow to reach about 80% of the maximum score axis extent.
    scale_factor = (max_score_extent * 0.8) / max_coeff_length

    # 2. Create a color map for the labels
    if color_on is not None:
        unique_labels = list(set(color_on))
        cmap = plt.cm.get_cmap('tab10', len(unique_labels))  # choose any colormap
        label_to_color = {label: cmap(i) for i, label in enumerate(unique_labels)}
        colors = [label_to_color[lbl] for lbl in color_on]
    else:
        unique_labels = list(set(point_labels))
        cmap = plt.cm.get_cmap('tab10', len(unique_labels))  # choose any colormap
        label_to_color = {label: cmap(i) for i, label in enumerate(unique_labels)}
        colors = [label_to_color[lbl] for lbl in point_labels]

    plt.figure(figsize=(10, 8))
    
    # 4. Plot the scores (Phage samples)
    plt.scatter(xs, ys, c=colors, s=50)
    if not hide_labels[0]:
         # Annotate Points with Sequence IDs
        texts = [
            plt.text(
                components_df.PC1[i] + 1.5,
                components_df.PC2[i] + 2.5,
                point_labels[i],
                fontsize=9
            )
            for i in range(components_df.shape[0])
        ]
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'))

    # Prepare list for adjust_text
    texts = []

    # 5. Plot the loadings (K-mers)
    for i in range(min(len(coeff), n_features_to_plot)):
        # Apply the scaling factor
        x_arrow = coeff[i, 0] * scale_factor
        y_arrow = coeff[i, 1] * scale_factor
        
        # Draw the arrow
        plt.arrow(0, 0, x_arrow, y_arrow, 
                  color='black', 
                  alpha=0.5, 
                  linewidth=1.25,
                  head_width=2,  # Adjust size of the arrow head
                  head_length=2.5,
                  overhang=0.25)
        
        if not hide_labels[1]:
            # Add the text label and collect it for adjust_text
            texts.append(plt.text(x_arrow * 1.05, y_arrow * 1.05, 
                                vec_labels[i], 
                                color='black', 
                                fontsize=9))

    # Use adjust_text to automatically position the labels without overlap
    if texts:
        adjust_text(texts, 
                    arrowprops=dict(arrowstyle="-", color='k', lw=0.5, alpha=0.8))
        
    # 6. Add legend for colors
    if color_on is not None or point_labels is not None:
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=lbl,
                              markerfacecolor=label_to_color[lbl], markersize=8) 
                   for lbl in unique_labels]
        plt.legend(title='Color Groups', handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Set axes and labels
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}% Variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}% Variance)')
    plt.title("PCA Biplot with Scaled Loadings")
    plt.grid(True, linestyle='--')
    plt.axhline(0, color='gray', linewidth=0.5, linestyle='-')
    plt.axvline(0, color='gray', linewidth=0.5, linestyle='-')
    ##plt.show()

def plot_roc_curve_rf(rf, x_test, y_test, title=None, save=None):
    y_pred_prob = rf.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label=1)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    #print(f"ROC AUC: {roc_auc}")

    # Plot the ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    # roc curve for tpr = fpr 
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    if title is None:
        plt.title(f'ROC Curve with AUC: {round(roc_auc, 2)}')
    else:
        title = title + f". AUC: {round(roc_auc, 2)}"
        plt.title(title)

    plt.legend(loc="lower right")
    
    if save is None:
        plt.show()
    else:
        try: 
            plt.savefig(save)
        except Exception as e:
            print("Unable to save ROC fig!")

def plot_residuals(x_vals, y_vals, tile=None):
    
    plt.figure(figsize=(10, 6))
    # Scatter plot of Predicted Values (X-axis) vs. Residuals (Y-axis)
    plt.scatter(x_vals, y_vals, alpha=0.6, color='darkgreen')

    # Draw the horizontal zero line (the ideal residual)
    plt.hlines(y=0, xmin=x_vals.min(), xmax=x_vals.max(), color='red', linestyle='--', lw=2)

    # --- 3. Label and Title the Plot ---

    plt.title('Residuals Plot for Random Forest Regressor')
    plt.xlabel('Predicted Scores ($\hat{y}$)')
    plt.ylabel('Residuals ($y - \hat{y}$)')
    plt.grid(True, linestyle=':', alpha=0.6)

    ##plt.show()

def plot_losses(train_losses, valid_losses, n_epochs, title=None):
    # Plotting the losses 
    fig,ax = plt.subplots(1,1, figsize=(9,5))
    ax.plot(range(n_epochs), train_losses, label='Train loss', c='b')
    ax.plot(range(n_epochs), valid_losses, label='Valid loss', c='m')
    ax.legend()
    if title is not None:
        fig.suptitle(title)
    fig.show()

def f1_analysis(y_true, y_probs, logging : bool, outdir = None, logfile = None, filename = None, silent = False):
    # Baseline at 0.5
    pred_05 = (y_probs >= 0.5).astype(int)
    prec_05 = precision_score(y_true, pred_05, zero_division=0)
    rec_05 = recall_score(y_true, pred_05, zero_division=0)
    f1_05 = f1_score(y_true, pred_05, zero_division=0)
    if outdir is not None and logging:
        if logfile is None:
            logfile = open(outdir + 'f1_analysis_log.txt', 'a')
            print("New logfile create at F1 analysis!")
    elif outdir is None and logging:
        raise ValueError("Please specify an outdir when logging is on")

    print(f"Baseline (threshold=0.5) -> Precision: {prec_05:.4f}, Recall: {rec_05:.4f}, F1: {f1_05:.4f}")
    if logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Baseline (threshold=0.5) -> Precision: {prec_05:.4f}, Recall: {rec_05:.4f}, F1: {f1_05:.4f}', file=logfile)

    # Sweep thresholds to find best F1
    thresholds = np.linspace(0.0, 1.0, 201)
    f1s = []
    prcs = []
    recs = []
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        f1s.append(f1_score(y_true, preds, zero_division=0))
        prcs.append(precision_score(y_true, preds, zero_division=0))
        recs.append(recall_score(y_true, preds, zero_division=0))
    f1s = np.array(f1s)
    prcs = np.array(prcs)
    recs = np.array(recs)

    best_idx = np.argmax(f1s)
    best_t = thresholds[best_idx]
    best_f1 = f1s[best_idx]
    best_prec = prcs[best_idx]
    best_rec = recs[best_idx]

    print(f"Best threshold by F1 -> threshold={best_t:.3f}, Precision={best_prec:.4f}, Recall={best_rec:.4f}, F1={best_f1:.4f}")
    if logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Best threshold by F1 -> threshold={best_t:.3f}, Precision={best_prec:.4f}, Recall={best_rec:.4f}, F1={best_f1:.4f}', file=logfile)

    # Classification report at best threshold
    best_preds = (y_probs >= best_t).astype(int)
    report = classification_report(y_true, best_preds, zero_division=0)
    print("\nClassification report at best threshold:\n", report)
    if logging: 
        print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Classification report at best threshold:', file=logfile)
        for line in report.splitlines():
            print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} {line}', file=logfile)

    # Average precision (area under PR curve)
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_probs)
    avg_prec = average_precision_score(y_true, y_probs)
    print(f"Average precision (AP): {avg_prec:.4f}")
    if logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Average precision (AP): {avg_prec:.4f}', file=logfile)

    # Confusion matrix at best threshold
    cm = confusion_matrix(y_true, best_preds)
    print("Confusion matrix (rows=true, cols=pred):\n", cm)
    if logging: 
        print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Confusion matrix:', file=logfile)
        for i in range(cm.shape[0]):
            print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} {cm[i]}', file=logfile)

    # Plots: F1 vs threshold and Precision-Recall curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(thresholds, f1s, label='F1', color='C0')
    axes[0].plot(thresholds, prcs, label='Precision', color='C1', linestyle='--')
    axes[0].plot(thresholds, recs, label='Recall', color='C2', linestyle=':')
    axes[0].axvline(best_t, color='k', linestyle='--', label=f'best t={best_t:.3f}')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Score')
    axes[0].set_title('F1 / Precision / Recall vs Threshold')
    axes[0].legend()

    axes[1].plot(recall_curve, precision_curve, color='darkorange', lw=2)
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'Precision-Recall curve (AP={avg_prec:.4f})')
    axes[1].grid(True)

    plt.suptitle(f'F1 analysis (best t={best_t:.3f}, F1={best_f1:.4f})')
    if logging: 
        if filename is None: 
            outname = 'torchMLP_f1_analysis.png'
        else:
            outname = filename
        plt.savefig(outdir + outname, bbox_inches='tight')
        print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} F1 analysis figure saved as: {outdir+outname}', file=logfile)

    # if silent is False:
    #     plt.show()

def plot_entity_counts(df: pd.DataFrame, entity_column: str, logging : bool, outdir: str = None,):
    """
    Counts the occurrences of an entity column in the DataFrame and plots 
    the result as a sorted horizontal bar graph.

    Args:
        outdir: path to out directory for saving
        df: The DataFrame containing the True Positive results.
        entity_column: The name of the column to count (e.g., 'Phage_Name').
        logging: Whether to save logs and plots
    
    Returns: 
        None (displays and saves the plot)
    """
    if outdir is None and logging:
        raise ValueError("Please specify an outdir when logging is on")

    # 1. Count the occurrences of each unique entity
    entity_counts = df[entity_column].value_counts()

    # Print the counts
    entity_type = entity_column.replace('_Name', '').replace('_', ' ')
    print(f"--- {entity_type} True Positive Counts ---")
    print(entity_counts)
    print("-" * 40)

    # 2. Prepare the data for plotting
    entity_names = entity_counts.index
    counts = entity_counts.values

    # 3. Plot the counts as a horizontal bar graph
    plt.figure(figsize=(10, max(6, len(entity_names) * 0.4))) # Adjust height dynamically

    # Create horizontal bars
    bars = plt.barh(entity_names, counts, color='#3498db' if 'Phage' in entity_column else '#2ecc71')

    # Add labels and title
    plt.xlabel("Count of True Positive Interactions")
    plt.ylabel(entity_column.replace('_', ' '))
    plt.title(f"Frequency of {entity_type} in True Positive Set (N={len(df)})")
    plt.gca().invert_yaxis() # Display the highest count at the top

    # Add the count labels to the bars
    for bar in bars:
        plt.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
                 f'{int(bar.get_width())}', 
                 va='center', ha='left')

    plt.tight_layout()
    
    # Optional: Save the figure
    if logging:
        if 'Phage' in entity_column:
            plt.savefig(outdir + 'phage_tp_counts.png')
        else:
            plt.savefig(outdir + 'bacterium_tp_counts.png')
    
    ##plt.show()

def plot_bipartite_network(df: pd.DataFrame, id_lookup_bact: pd.DataFrame, logging : bool, outdir: str = None, limit: int = 50, conf_threshold=0.5):
    """
    Creates and plots a bipartite network graph of Phage-Bacterium True Positive 
    interactions, weighted by predicted probability, with bacterial nodes 
    colored by species.

    Args:
        df: DataFrame containing 'Bacterium_Name', 'Phage_Name', and 'Predicted_Probability' (sorted by confidence).
        id_lookup_bact: DataFrame with Bacterium metadata ('Bacterium_Name', 'Species').
        logging: Whether to save logs and plots
        limit: The maximum number of interactions to include in the plot.
        conf_threshold: Minimum predicted probability to include an interaction in the plot.
    
    Returns: 
        None (displays and saves the plot)
    """
    if outdir is None and logging:
        raise ValueError("Please specify an outdir when logging is on")

    # 1. Prepare Data and Subsample
    df_plot = df.head(limit).copy()
    
    if len(df) > limit:
        print(f"Plotting the top {limit} most confident interactions out of {len(df)}.")
    
    # Filter to only include predicted probabilities greater than conf_threshold
    initial_filtered_size = len(df_plot)
    df_plot = df_plot[df_plot['Predicted_Probability'] > conf_threshold].copy()
    
    if initial_filtered_size > len(df_plot):
        print(f"Removed {initial_filtered_size - len(df_plot)} interactions because Predicted_Probability <= conf_threshold.")

    if len(df_plot) == 0:
        print("No interactions remain after filtering by probability > conf_threshold. Plotting aborted.")
        return

    bacteria = df_plot['Bacterium_Name'].unique()
    phages = df_plot['Phage_Name'].unique()

    # 2. Add Species information to the plot data
    # Ensure id_lookup_bact has 'Bacterium_Name' and 'Species'
    df_merged = df_plot.merge(id_lookup_bact[['Bacterium_Name', 'Species']].drop_duplicates(), 
                              on='Bacterium_Name', how='left')
    
    # Get unique species for coloring
    unique_species = df_merged['Species'].dropna().unique()
    # Use a distinct colormap (Tab10 is good for categorical data)
    species_cmap = plt.cm.get_cmap('tab10', len(unique_species))
    species_to_color = {species: species_cmap(i) for i, species in enumerate(unique_species)}
    
    # Assign colors to bacterium nodes based on species
    bact_colors = []
    for node in bacteria:
        # Safely access species name, handling potential missing species
        species_info = df_merged[df_merged['Bacterium_Name'] == node]['Species']
        species_name = species_info.iloc[0] if not species_info.empty else None
        bact_colors.append(species_to_color.get(species_name, 'gray')) # Default to gray if species is missing

    # 3. Create the Graph
    G = nx.Graph()
    G.add_nodes_from(bacteria, bipartite=0, label='Bacterium')
    G.add_nodes_from(phages, bipartite=1, label='Phage')
    
    # Add edges, weighting them by the Predicted_Probability
    for _, row in df_plot.iterrows():
        bact_name = row['Bacterium_Name']
        phage_name = row['Phage_Name']
        prob = row['Predicted_Probability']
        G.add_edge(bact_name, phage_name, weight=prob, score=prob)

    # 4. Define Layout and Styling
    
    pos = nx.bipartite_layout(G, bacteria)

    edge_scores = np.array([G[u][v]['score'] for u, v in G.edges()])
    # Scale width for visual emphasis (e.g., width from 1 to 5)
    edge_widths = (edge_scores - edge_scores.min() + 0.1) * 4 / (edge_scores.max() - edge_scores.min() + 0.1)

    # Use a color map for edge confidence
    cmap = plt.cm.plasma
    # FIX: Normalize based on fixed scale (0 to 1) instead of data range
    norm = Normalize(vmin=0, vmax=1) 
    edge_colors = cmap(norm(edge_scores))
    
    PHAGE_COLOR = '#008000' # User change: Dark Green

    # 5. Draw the Graph
    plt.figure(figsize=(14, 14)) 
    # Updated title to reflect the filtering
    plt.title(f'Bipartite Network of Top {len(df_plot)} True Positive Interactions (P > {conf_threshold})', 
              fontsize=16, fontweight='bold')
    
    NODE_SIZE = 2500 
    
    # Draw Bacterium nodes (colored by species)
    nx.draw_networkx_nodes(G, pos, nodelist=bacteria, node_color=bact_colors, 
                           node_size=NODE_SIZE) 
    # Draw Phage nodes (single color)
    nx.draw_networkx_nodes(G, pos, nodelist=phages, node_color=PHAGE_COLOR, 
                           node_size=NODE_SIZE)
    
    # Draw edges with calculated width and color
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.8) 
    
    # Clean up Bacterium labels for visualization
    labels = {}
    for node in G.nodes():
        name = node
        # Check if the node is a Bacterium
        if name in bacteria:
            # Remove the '_reoriented' suffix if present
            if '_reoriented' in name:
                name = name.replace('_reoriented', '')
        labels[node] = name
    
    # Draw labels using the cleaned names
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='white', font_weight='bold') 
    
    # 6. Add Color Bar Legend (for edge scores)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) 
    cbar = plt.colorbar(sm, orientation='horizontal', pad=0.05, aspect=50, ax=plt.gca())
    # The label now correctly refers to the 0-1 range
    cbar.set_label('Predicted Probability (Model Confidence)', rotation=0, labelpad=15)
    
    # 7. Add Custom Legend (for node colors/species)
    species_patches = []
    # Add species patches
    for species, color in species_to_color.items():
        species_patches.append(Line2D([0], [0], marker='o', color='black', label=species, 
                                          markerfacecolor=color, markersize=15))
    # Add phage patch
    phage_patch = Line2D([0], [0], marker='o', color='black', label='Phage', 
                             markerfacecolor=PHAGE_COLOR, markersize=15)
    species_patches.append(phage_patch)

    plt.legend(handles=species_patches, loc='lower center', title="Entity Species/Type",
               frameon=True, fontsize=10)


    plt.axis('off')
    plt.tight_layout()
    if logging: plt.savefig(outdir + f'bipartisan_conf_interactions_p{conf_threshold}.png') 
    ###plt.show()

def model_idx_to_kmer(idx, num_features_per_entity, feature_indices, idx_to_minhash):
    """
    Maps a model feature index back to the encoded k-mer (minhash index).
    """
    original_col_idx = feature_indices[idx % num_features_per_entity]
    return idx_to_minhash[original_col_idx]

def regain_kmers(k: int, n: int, prefix : str, sourmash: bool, top_n: int = 20, idx_to_minhash: dict = None, 
                 mapping_func=None, mapping_args=None, attributions=None, 
                 TS: bool = False, logging: bool = False, logfile=None):
    """
    Standalone function to regain original k-mer features corresponding to top feature indices.
    
    Returns:
        tuple: (top_indices, top_values, decoded_kmers_list)
    """
    if sourmash:
        print("Sourmash-based model does not support k-mer decoding.")
        return [], [], []

    # 1. Determine top indices and values
    if idx_to_minhash is not None:
        top_idx = list(idx_to_minhash.keys()) 
        top_vals = "N/A"
        if TS: print(f"Using provided idx_to_minhash for top indices: {top_idx}")
    else:
        if attributions is None:
            raise ValueError("Attributions must be provided if idx_to_minhash is None.")
        avg_attr = attributions.mean(dim=0)
        abs_avg = avg_attr.abs()
        k_count = min(top_n, abs_avg.numel())
        topk = torch.topk(abs_avg, k_count)
        top_idx = topk.indices.cpu().numpy()
        top_vals = avg_attr[top_idx].cpu().numpy()
        if TS: 
            print(f"Top {top_n} indices:", top_idx)
            print("Mean attributions:", top_vals)
        if logging and logfile: 
            print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Top {top_n} indices: {top_idx}', file=logfile)
            print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Mean attributions: {top_vals}', file=logfile)

    # 2. Setup mapping
    if mapping_func is None:
        if mapping_args is None:
            raise ValueError("If no mapping_func is provided, mapping_args must be provided.")
        mapping_func = model_idx_to_kmer

    # 3. Load hk_translation_dict to translate hash to kmer
    hash_kmer_dict_path = os.path.join(data_prod_path, f"{prefix}/hk_lookup_n{n}_k{k}.json")
    if os.path.exists(hash_kmer_dict_path):
        with open(hash_kmer_dict_path, "r") as f:
            hk_translation_dict = json.load(f)
            try:
                hk_translation_dict = {int(k): v for k, v in hk_translation_dict.items()} # convert keys back to int after loading from json
            except Exception as e:
                raise ValueError(f"Error converting hk_translation_dict keys to int: {e}")

    else:
        print(f"Hash k-mer lookup dictionary not found at {hash_kmer_dict_path}. Please ensure the file exists or run the script to generate it.")
        hk_translation_dict = None

    # 4. Decode
    decoded_kmers_dict = {}  # Changed from list to dict
    for idx in top_idx:
        kmer_hash_val = mapping_func(idx, *mapping_args)
        decoded_kmers_dict[int(idx)] = hk_translation_dict[kmer_hash_val]
    
    if idx_to_minhash is not None:
        pass
    else:
        if TS: 
            print("Decoded kmers mapping:", decoded_kmers_dict)
        
        if logging and logfile: 
            print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Decoded kmers: {decoded_kmers_dict}', file=logfile)
    
    return top_idx, top_vals, decoded_kmers_dict

def get_strain_name(hash_value, hash_lookup):
    """Decode hash value to strain name using hash_lookup."""
    if hash_value in hash_lookup:
        strains = hash_lookup[hash_value]
        if isinstance(strains, list) and len(strains) > 0:
            return strains[0]  # Return first strain name from list
        elif isinstance(strains, str):
            return strains
    return str(hash_value)

def plot_interaction_pairs(interaction_pairs: dict, occurence_pairs: dict, expected_interactions: dict, hash_lookup: dict, hk_translation_dict: dict,
                           sort_by_ratio: bool = False, logging : bool = False, outdir: str = None, bact_clusters: pd.DataFrame = None):

    # Divide interaction score by occurrence count
    interaction_ratio_pairs = {
        pair: (interaction_pairs[pair] / occurence_pairs[pair] if occurence_pairs.get(pair, 0) != 0 else float("nan"))
        for pair in interaction_pairs.keys() & occurence_pairs.keys()
    }

    # Create DataFrame with decoded strain names from hash_lookup
    pair_df = pd.DataFrame({
        "Bacterium": ["_".join([hash_lookup[pair[0]][0], hk_translation_dict.get(hash_lookup[pair[0]][0], "Unknown")]) for pair in interaction_ratio_pairs.keys()],
        "Phage": ["_".join([hash_lookup[pair[1]][0], hk_translation_dict.get(hash_lookup[pair[1]][0], "Unknown")]) for pair in interaction_ratio_pairs.keys()],
        "Interaction_Ratio": list(interaction_ratio_pairs.values())
    })

    print(pair_df.head(10))

    # Filter zeros and pivot
    pair_no_zero_df = pair_df[pair_df["Interaction_Ratio"] > 0]
    pivot_df = pair_no_zero_df.pivot_table(index="Phage", columns="Bacterium", values="Interaction_Ratio", fill_value=0)
    print(f"pivot_df before sorting:\n", pivot_df.head(10))

    # Reorder axes if sort_by_ratio is True
    if sort_by_ratio:
        # 1. Sort Phages (rows) by their average interaction score
        phage_order = pivot_df.mean(axis=1).sort_values(ascending=False).index
        #print("Phage order after sorting:\n", phage_order.tolist())

        # 2. Sort Bacteria (columns) by their average interaction score
        bact_order = pivot_df.mean(axis=0).sort_values(ascending=False).index
        #print("Bacterium order after sorting:\n", bact_order.tolist())
        
        # Reindex the pivot table with these orders
        pivot_df = pivot_df.reindex(index=phage_order, columns=bact_order)

        #print(f"pivot_df after sorting {pivot_df.shape}:\n", pivot_df.head(10))
        #print("pivot_df index without kmer suffix:\n", ['_'.join(idx.split('_')[:3]) for idx in pivot_df.index])
        #print("bact_clusters:\n", bact_clusters)
        #print("bact_clusters reindexed:\n", bact_clusters.reindex(['_'.join(idx.split('_')[:3]) for idx in pivot_df.index]).head(10))
        #print(f"bact_clusters {bact_clusters.shape}\n", bact_clusters.head(10))

        #Alternative graph (clustermap)
        plt.figure(figsize=(12, 8))
        col_linkage = None
        col_colors = None
        if bact_clusters is not None:
            cluster_col = None
            for candidate in ["Cluster", "cluster", "Clusters", "clusters"]:
                if candidate in bact_clusters.columns:
                    cluster_col = candidate
                    break

            if cluster_col is not None:
                lookup_keys = [idx.rsplit('_', 1)[0] for idx in pivot_df.index]
                bact_clusters_unique = bact_clusters.loc[~bact_clusters.index.duplicated(keep='first')]
                bact_meta = bact_clusters_unique.reindex(lookup_keys)
                #print(f"Cluster column '{cluster_col}' found in metadata. Using it for column coloring.")
                #print(f"bact_meta {bact_meta.shape}: {bact_meta.head(10)}")

                bact_meta.rename(columns={cluster_col: 'Bacteria Clusters'}, inplace=True)
                cluster_col = 'Bacteria Clusters'
                #print(f"bact_meta with cluster_col: {bact_meta[cluster_col]}")

                # Overwrite index to contain the full bacterium names (with kmer suffix) to ensure proper alignment with pivot_df columns
                bact_meta.index = pivot_df.index
                #print(f"bact_meta index overwriting: {bact_meta.head(10)}")

                cluster_series = bact_meta[cluster_col].fillna("Unknown").astype(str)
                #print(f"Cluster series for coloring:\n{cluster_series.head(10)}")

                # Build hierarchical linkage from cluster membership to enforce bacteria-wise grouping
                cluster_matrix = pd.get_dummies(cluster_series, prefix="cluster")
                if len(cluster_matrix) >= 2:
                    col_linkage = linkage(cluster_matrix.values, method="average", metric="euclidean")

                unique_clusters = pd.Index(cluster_series.unique())
                #print(f"Unique clusters found for coloring: {unique_clusters.tolist()}")
                palette = sns.color_palette("tab20", n_colors=max(len(unique_clusters), 1))
                cluster_to_color = {cluster: palette[i] for i, cluster in enumerate(unique_clusters)}
                col_colors = cluster_series.map(cluster_to_color)

        g = sns.clustermap(
            pivot_df.T,
            cmap="viridis",
            standard_scale=None,
            row_cluster=False,
            col_cluster=True,
            col_linkage=col_linkage,
            col_colors=col_colors,
        )

        # Color x and y labels by cluster membership
        if col_colors is not None:
            for label in g.ax_heatmap.get_xticklabels():
                label_text = label.get_text()
                if label_text in pivot_df.columns:
                    cluster_value = cluster_series.get(label_text, "Unknown")
                    label.set_color(cluster_to_color.get(cluster_value, "black"))
            for label in g.ax_heatmap.get_yticklabels():
                label.set_color("black")

        if logging and outdir:
            plt.savefig(os.path.join(outdir, 'interaction_pairs_clustermap.png'))

    # Plotting raw interaction ratio heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, cmap="viridis", cbar_kws={"label": "Interaction Ratio"})
    
    title = "Interaction Ratio of Bacterium-Phage Pairs"
    plt.title(title)
    plt.xlabel("Bacterium")
    plt.ylabel("Phage")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()

    if logging and outdir:
        graph_name = 'interaction_pairs_sorted.png' if sort_by_ratio else 'interaction_pairs.png'
        plt.savefig(os.path.join(outdir, graph_name))
    
    # Plotting interaction ratio scaled with expected interaction
    #divide interaction ratio by expected interaction to get a fold-change like measure
    scaled_pivot_df = pivot_df.copy()
    for phage in scaled_pivot_df.index:
        for bact in scaled_pivot_df.columns:
            expected_value = expected_interactions.get((bact, phage), 1)  # Avoid division by zero
            if expected_value != 0:
                scaled_pivot_df.at[phage, bact] = pivot_df.at[phage, bact] / expected_value
            else:
                scaled_pivot_df.at[phage, bact] = float("nan")  # Set to NaN if expected value is zero
    plt.figure(figsize=(12, 8))
    sns.heatmap(scaled_pivot_df, cmap="viridis", cbar_kws={"label": "Interaction Ratio / Expected Interaction"})

    title = "Interaction Ratio of Bacterium-Phage Pairs Scaled by Expected Interaction"
    plt.title(title)
    plt.xlabel("Bacterium")
    plt.ylabel("Phage")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()

    if logging and outdir:
        graph_name = 'interaction_pairs_sorted_scaled.png' if sort_by_ratio else 'interaction_pairs_scaled.png'
        plt.savefig(os.path.join(outdir, graph_name))

class FeatureImportance():
    def __init__(self, model, outdir, metadata_test, id_lookup_bact, host_range_data, raw_data_path, data_prod_path, logfile, logging : bool, TS : bool = False):
        self.raw_data_path = raw_data_path
        self.data_prod_path = data_prod_path
        self.model = model
        self.ig = IntegratedGradients(model)
        self.attributions = None
        self.delta = None
        self.metadata_test = metadata_test
        self.id_lookup_bact = id_lookup_bact
        self.host_range_data = host_range_data
        self.TS = TS
        self.pca_prepped = False
        self.logging = logging
        self.logfile = logfile

        if outdir is None and logging:
            self.logging = False
            if self.TS: print("Logging turned off as no outdir was given!")
        
        else:
            self.outdir = outdir

    def compute_importance(self, input_tensor, target, delta : bool = False):
        """
        Computes feature importance attributions using Integrated Gradients for a given input tensor and target class. If delta is True, also computes the convergence delta to assess attribution completeness.
        Args:
            input_tensor (torch.Tensor): The input tensor for which to compute feature importances.
            target (int): The target class index for which to compute attributions.
            delta (bool): Whether to compute and return the convergence delta. Default is False.
        Returns:
            None (stores attributions and optionally delta in the class instance)
        """
        if hasattr(self.ig, 'attribute'):
            if delta:
                self.attributions, self.delta = self.ig.attribute(input_tensor, target=target, return_convergence_delta=delta)
            else:
                self.attributions = self.ig.attribute(input_tensor, target=target)
        else:
            raise ValueError("IntegratedGradients object does not support attribute method.")

    def plot_top_importance(self, top_n=20):
        """
        Plots the top N feature importances as a horizontal bar graph, with feature names on the y-axis and importance values on the x-axis. The plot is saved to the output directory specified in the class initialization.
        Args:
            top_n (int): The number of top features to display in the plot. Default is
        Returns:
            None (displays and saves the plot)
        """
        outname = 'top_feature_importance.png'
        if self.attributions is None:
            raise ValueError("Feature importances have not been computed yet.")
        
        # Get indices of top N features
        indices = np.argsort(self.attributions)[::-1][:top_n]
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.title("Top Feature Importances")
        plt.bar(range(top_n), self.attributions[indices], color='b', align='center')
        plt.xticks(range(top_n), [self.feature_names[i] for i in indices], rotation=90)
        plt.xlim([-1, top_n])
        plt.tight_layout()
        if self.logging: plt.savefig(self.outdir + outname)
        ##plt.show()
    
    def plot_attributions(self):
        """
        Plots the average feature attributions across all test samples. The x-axis represents the MinHash feature index, and the y-axis represents the average attribution value. The plot is saved to the output directory specified in the class initialization.
        Args:
            None (relies on initialized attributions)
        Returns:
            None (displays and saves the plot)
        """
        outname = 'average_feature_attr.png'
        avg_attributions = self.attributions.mean(0).cpu().detach().numpy()

        fig, ax = plt.subplots(figsize=(8,5))
        plt.plot(avg_attributions, label='Average Attribution values')
        plt.xlabel('MinHash Feature Index')
        plt.ylabel('Attribution Value')
        plt.title('Average Feature Attributions for Test Samples')
        plt.legend()
        if self.logging: plt.savefig(self.outdir+outname)
        ##plt.show()

    def _prep_PCA(self):
        """Prepares the feature importance attributions for PCA analysis by converting them to a numpy array, extracting labels for coloring, and performing PCA to compute the scores and loadings."""
        self.attr_np = self.attributions.detach().cpu().numpy() #Convert torch tensor to numpy array

        # labels for bact genus groupings
        self.blabels = []
        for bact_sample in self.metadata_test[:,0]:
            self.blabels.append(self.id_lookup_bact[self.id_lookup_bact['Bacterium_Name'] == bact_sample]['Species'].values[0])

        # labels for phage genus groupings
        self.plabels = [phage_sample for phage_sample in self.metadata_test[:,1]]

        # labels for bact samples that has at least one phage interaction
        self.leastonelabels = []
        for bact_sample in self.metadata_test[:,0]: #Iterate through the bacterium samples in the test metadata
            if bact_sample in self.host_range_data.keys() and any(self.host_range_data[bact_sample].values()):
                self.leastonelabels.append(True)
            else:
                self.leastonelabels.append(False)

        if self.TS:
            print(Counter(self.leastonelabels))

        self.pca = PCA(n_components=2)
        self.embedding = self.pca.fit_transform(self.attr_np)
        self.loadings = self.pca.components_.T

        # We'll take top 10 features by magnitude (sqrt(pc1^2 + pc2^2))
        magnitude = np.sqrt(self.loadings[:, 0]**2 + self.loadings[:, 1]**2)
        self.top_indices = np.argsort(magnitude)[-10:] 

        # We scale the loadings so they are visible on the same scale as the scores
        self.scale_factor = np.max(np.abs(self.embedding)) / np.max(np.abs(self.loadings)) * 0.8
        self.pca_prepped = True

    def plot_PCA(self, color_samples_by : str = 'bacteria', biplot : bool = True):
        """
        Plots a PCA biplot of the feature importances, coloring the samples by either bacteria species, phage species, or whether the bacterium has at least one known phage interaction.
        Args:
            color_samples_by (str):['bacteria', 'phage', 'interaction'] Determines how to color the samples in the PCA plot.
        Returns:
            None (displays and saves the PCA biplot)
        """
        if color_samples_by not in ['bacteria', 'phage', 'interaction']:
            raise ValueError("color_samples_by must be one of: 'bacteria', 'phage', 'interaction'")

        outname = f'feature_importance_PCA_{color_samples_by}.png'
        if not self.pca_prepped:
            self._prep_PCA()

        if color_samples_by == 'bacteria':
            label = self.blabels
            title_appendix = "Bacteria Strains"    
            pal = "tab20"

        elif color_samples_by == 'phage':
            label = self.plabels
            title_appendix = "Phage Species"
            pal = "tab20"

        elif color_samples_by == 'interaction':
            label = self.leastonelabels
            title_appendix = "Bacteria Interacting"
            pal = "Set2"
        
        if len(set(label)) > 10:
            pal = "tab20"
        else:
            pal = "tab10"
        
        plt.figure(figsize=(10, 7))

        # Scores
        sns.scatterplot(x=self.embedding[:,0], y=self.embedding[:,1], hue=label,
                        palette=pal, s=40, alpha=0.6, edgecolor='w')

        # Loadings (feature vectors)
        if biplot:
            for i in self.top_indices:
                plt.arrow(0, 0, self.loadings[i, 0]*self.scale_factor, self.loadings[i, 1]*self.scale_factor, 
                        color='black', alpha=0.7, head_width=0.05)
                plt.text(self.loadings[i, 0]*self.scale_factor*1.05, self.loadings[i, 1]*self.scale_factor*1.05, 
                        f'F{i}', color='black', fontsize=9)

        plt.title(f'Biplot of Sample Attributions - Colored by {title_appendix}')
        plt.xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.grid(True, linestyle='--', alpha=0.5)

        # Decrease legend box: smaller font, smaller markers, tighter spacing
        leg = plt.legend(title=title_appendix.split(" ")[1], loc='best',
                        fontsize='small', title_fontsize='small',
                        markerscale=0.6, borderpad=0.3, labelspacing=0.5,
                        handletextpad=0.4)
        if self.logging: plt.savefig(self.outdir + outname)
    
    def plot_attributions_PCA_clusters(self, n_clusters : int = 4, color_samples_by : str = 'bacteria'):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(self.embedding)
        labels = kmeans.labels_

        if color_samples_by == 'bacteria':
            label = self.blabels    
        elif color_samples_by == 'phage':
            label = self.plabels
        elif color_samples_by == 'interaction':
            label = self.leastonelabels

        # Relate majority blabels (bacterium species) to each cluster
        cluster_majority = {}
        for i in range(n_clusters):
            idx = np.where(labels == i)[0]
            if idx.size == 0:
                print(f"Cluster {i}: EMPTY")
                continue
            species_in_cluster = [label[j] for j in idx]
            cnt = Counter(species_in_cluster)
            top3 = cnt.most_common(3)
            top_label, top_count = top3[0]
            pct = top_count / idx.size * 100
            cluster_majority[i] = top_label
            print(f"Cluster {i}: size={idx.size} | majority={top_label} ({top_count} samples, {pct:.1f}%) | top3={top3}")

        # 2. Calculate mean attribution per cluster
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True)
        axes = axes.flatten()
        fig.suptitle("Mean Feature Attribution by PCA Cluster")
        colors = plt.cm.tab10(np.arange(n_clusters) % 10)

        # Gather cluster means first to compute global y-limits
        cluster_means = {}
        for i in range(n_clusters):
            cluster_mask = (labels == i)
            if not cluster_mask.any():
                continue
            cluster_means[i] = self.attr_np[cluster_mask].mean(axis=0)

        if cluster_means:
            all_vals = np.concatenate(list(cluster_means.values()))
            y_min, y_max = all_vals.min(), all_vals.max()
        else:
            y_min, y_max = 0, 1

        for i in range(n_clusters):
            ax = axes[i]
            if i in cluster_means:
                color = colors[i % len(colors)]
                ax.plot(cluster_means[i], label=f'{cluster_majority.get(i,"N/A")}', color=color, alpha=0.8)
            ax.set_title(f"Cluster {i+1} Mean Feature Attribution")
            ax.set_ylim(y_min, y_max)
            ax.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if self.logging: 
            outname = 'cluster_span_attr_bcolor.png'
            plt.savefig(self.outdir+outname)
        ##plt.show()
    
    def regain_kmers_fa(self, k : int, sourmash : bool, top_n : int = 20, idx_to_minhash : dict = None, mapping_func=None, mapping_args=None):
        """
        Wrapper inside the class that calls the standalone regain_kmers function.
        """
        self.k = k
        # Call standalone function and store results in instance state
        self.top_idx, self.top_vals, self.top10_decoded = regain_kmers(
            k=k, 
            sourmash=sourmash, 
            top_n=top_n, 
            idx_to_minhash=idx_to_minhash, 
            mapping_func=mapping_func, 
            mapping_args=mapping_args, 
            attributions=self.attributions, 
            TS=self.TS, 
            logging=self.logging, 
            logfile=self.logfile
        )
        return self.top10_decoded

    def plot_top_kmers(self, sourmash : bool, top_n : int = 20):
        """
        Plots the top N k-mer features based on their importance scores, with bars colored by the sign of the attribution (positive or negative). The plot is saved to the output directory specified in the class initialization.
        Args:
            top_n (int): The number of top k-mer features to display in the plot. Default is 20.
        Returns:
            None (displays and saves the plot)
        """
        if not sourmash:
            # Plot top10_decoded with thier attribution values
            fig, ax = plt.subplots(figsize=(10, 8))
            bars = ax.bar(range(len(self.top_idx)), self.top_vals, color='#1f77b4')
            ax.set_xticks(range(len(self.top_idx)))
            ax.set_xticklabels(self.top10_decoded, rotation=45, ha='right')
            ax.set_ylabel('Mean Attribution Value')
            ax.set_title('Top 10 Kmers by Mean Attribution Value')
            plt.tight_layout()

            if self.logging: 
                outname = f'top10_{self.k}mer_attr.png'
                plt.savefig(self.outdir+outname)
            ##plt.show()
        
        else:
            print("Sourmash-based model does not support k-mer decoding or plotting.")

    def run_shap_analysis(self, X_test, X_test_tensor):
        """
        Performs SHAP analysis. One for first sample attributions, waterfall plot, then one plot for global attributions 
        Args:
            X_test (numpy.ndarray): the test array used for evaluation
            X_test_tensor (torch.Tensor): the tensor used for evaluation.
        Returns:
            None (displays and saves the SHAP plots)
        """
        print(f"Starting SHAP analysis on given samples...")
        
        explainer = shap.GradientExplainer(self.model, X_test_tensor)
        raw_shap_values = explainer.shap_values(X_test_tensor)
        shap_vals_array = raw_shap_values[0] if isinstance(raw_shap_values, list) else raw_shap_values

        # Manually calculate the 1D Expected Value - GradientExplainer doesn't have explain_value attribution
        self.model.eval()
        with torch.no_grad():
            bg_preds = self.model(X_test_tensor)
            # Ensure this is a single scalar number
            expected_value_scalar = bg_preds.mean().item()
        
        ###### 1. Visualization: First sample space #######
        # Manually create shap explanation with metadata, for plotting
        exp = shap.Explanation(
            values=shap_vals_array[0].flatten(),  # Select first sample and flatten to 1D
            base_values=expected_value_scalar,    # The scalar starting point
            data=X_test[0].flatten(),            # The raw feature values for sample 0
            feature_names=[f"Feat_{i}" for i in range(X_test.shape[1])])

        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(exp, max_display=30)
        
        if self.logging:
            plt.savefig(self.outdir + "shap_first_wf.png", bbox_inches='tight')
            print(f"SHAP first sample waterfall saved to {self.outdir}")
        
        ##plt.show()

        ###### 2. Visualization: Global importance ######
        # Manually create shap explanation with metadata, for plotting
        shap_values_batch = raw_shap_values[0] if isinstance(raw_shap_values, list) else raw_shap_values

        global_exp = shap.Explanation(
            values=shap_values_batch.squeeze(), # Shape should be (100, 42270)
            data=X_test[:100],
            feature_names=[f"Feat_{i}" for i in range(X_test.shape[1])])

        plt.figure(figsize=(10, 8))
        shap.plots.bar(global_exp, max_display=20)
        
        if self.logging:
            plt.savefig(self.outdir + "shap_glob_attr.png", bbox_inches='tight')
            print(f"SHAP Global Attributions saved to {self.outdir}")
        
        ##plt.show()

class GeneAnalysis():
    def __init__(self, logfile, logging : bool, outdir : str):
        self.root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
        self.raw_data_path = os.path.join(self.root, "raw_data/")
        self.data_prod_path = os.path.join(self.root, "data_prod/")
        self.path_to_nn_runs = os.path.join(self.root, "nn_runs/")
        self.logfile = logfile
        self.logging = logging
        self.outdir = outdir

        # Load kmer annotations from CSV into a dictionary for quick lookup
        self.local_kmer_db = self.data_prod_path + "kmer_annotations.csv"
    
    def _clean_kmer_line(self, kmer_line):
        """Clean the line containing kmers, from a messy string with noise, to a list with only decoded kmers"""
        kmers_string = kmer_line.split(":")[-1].strip()
        return kmers_string.strip("[]").replace("'", "").split(", ")

    def _save_kmer_annotations(self, kmer_annot_df):
        """
        Saves the k-mer annotations dataframe to a CSV file for future reference. This allows for building a local database of k-mer annotations that can be quickly accessed in subsequent analyses without needing to re-query NCBI for the same kmers.
        First load the existing annotations, then append new ones, and save the combined dataframe back to the CSV file.
        Args:
            kmer_annot_df (pd.DataFrame): A dataframe containing k-mer sequences and their annotated functions to be saved.
        Returns:
            None (saves the dataframe to a CSV file)
        """
        # Filter away rows only containing "N/A" in "Gene" & "Function" columns
        kmer_annot_df = kmer_annot_df[~((kmer_annot_df['Gene'] == "N/A") & (kmer_annot_df['Function'] == "N/A"))]
        
        try:
            existing_annot_df = pd.read_csv(self.local_kmer_db)
            combined_df = pd.concat([existing_annot_df, kmer_annot_df], ignore_index=True).drop_duplicates(subset=['Kmer'])
        except FileNotFoundError:
            combined_df = kmer_annot_df
        
        combined_df.to_csv(self.local_kmer_db, index=False)

    def _load_kmer_annotations(self, kmer_list):
        """
        Performs a lookup in the CSV file containing the annotations for all kmers from previous runs, and returns a dataframe with existing kmer and their annotations. This allows for quick retrieval of functional information for any given kmer.
        Args:
            kmer_list (list): A list of k-mer sequences for which to try to retrieve annotations.
        Returns:
            pd.DataFrame: A dataframe containing the kmer sequences and their annotated functions.
        """
        kmer_annot_df = pd.DataFrame(columns=['Kmer', 'Gene', 'Function'])  # Initialize empty DataFrame with expected columns
        try:
            kmer_annot_df = pd.read_csv(self.local_kmer_db)   
            # Filter away rows only containing "N/A" in "Gene" & "Function" columns
            kmer_annot_df = kmer_annot_df[~((kmer_annot_df['Gene'] == "N/A") & (kmer_annot_df['Function'] == "N/A"))]
            # Filter df to only include kmers from kmer_list (case-insensitive)
            kmer_annot_df = kmer_annot_df[kmer_annot_df['Kmer'].isin(kmer_list)]
        
            # Check for missing kmers
            found_kmers_lower = set(kmer_annot_df['Kmer'])
            missing_kmers = set(kmer_list) - found_kmers_lower
            
            if missing_kmers and self.logging:
                print(f"Note: {len(missing_kmers)} kmers not found in annotation database")
                print(f"Missing kmers in db: {missing_kmers}")
        
        except FileNotFoundError:
            print(f"Annotation file {self.local_kmer_db} not found.")

        except Exception as e:
            print(f"Error loading kmer annotations: {e}")

        return kmer_annot_df

    def extract_kmer_list(self, file_path):
        """
        Extracts the list of top decoded kmers from a log file generated during a feature importance analysis. 
        The function reads the log file, searches for the line containing the top decoded kmers, and returns a cleaned list of those kmers.
        
        Args:
            file_path (str): The path to the log file containing the feature importance analysis results.
        
        Returns:
            list: A list of the top decoded kmers extracted from the log file.
        """
        # Check if filepath exists
        try:
            os.path.exists(file_path)
        except FileExistsError as e:
            print("File path doesn't exist", e)
            return []
        
        with open(file_path, "r") as logfile:
            for line in logfile:
                if "Top 10 decoded kmers:" in line:
                    return self._clean_kmer_line(line)

    def search_and_annotate_kmers(self, kmer_list, organism : str = "Unknown", summarise_by: str = None, outfile:str = None, acc_num:int = 3, tax_origin:str  = "txid38018[orgn]", ncbi_program:str = "blastn", ncbi_db:str = "core_nt", expect:int = 1000):
        """
        Blasts each of the kmers against NCBI, for related species (accessions), then searches its genes for the kmer along with possible functionalities.
        
        Args:            
            kmer_list (list | pd.DataFrame): A list or DataFrame of k-mer sequences to search for.
            organism (str): The organism given (related in output file names)
            summarise_by (str): Whether to summarise the kmer results by gene or function.
            outfile (str): The path to the output file where results will be logged. If None, defaults to "logs/NCBI_gene_search.txt" in the root directory.
            acc_num (int): The number of top BLAST hits to consider for gene annotation.
            tax_origin (str): The Entrez query to restrict BLAST search to a specific taxonomic group (default is "txid38018[orgn]" for phages, "txid91347[orgn]" for Enterobacterales).
            ncbi_program (str): The BLAST program to use (default is "blastn" for nucleotide BLAST).
            ncbi_db (str): The NCBI database to search against (default is "core_nt" for the core nucleotide database).
        
        Returns:
            pd.DataFrame: A DataFrame containing the k-mers, the genes they were found in, and the annotated functions of those genes based on the BLAST hits.
        """
        Entrez.email = "s215045@student.dtu.dk"
        if outfile is None:
            outfile = self.root + f"logs/{organism}_NCBI_gene_search.csv"
        
        if self.logging:
            ncbi_logs = self.root + f"logs/{organism}_NCBI_search_logs.txt"
            print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Starting gene annotation for {len(kmer_list)} kmers. Output will be saved to {outfile}. NCBI logs will be saved to {ncbi_logs}', file=self.logfile)

        if isinstance(kmer_list, pd.DataFrame):
            kmer_list = kmer_list['decoded_kmer'].tolist() # Assuming the DataFrame has a column named 'kmer'

        print(f"Checking for existing annotations for {len(kmer_list)} kmers in local database...")
        existing_annot_df = self._load_kmer_annotations(kmer_list)
        annotated_kmers = set(existing_annot_df['Kmer'])
        if len(annotated_kmers) == 0:
            print("No existing annotations found.")
            kmers_to_annotate = kmer_list
        else:
            kmers_to_annotate = [k for k in kmer_list if k not in annotated_kmers]
        
        print(f"Found {len(existing_annot_df)} existing annotations. {len(kmers_to_annotate)} kmers will be annotated through BLAST search.")
        if len(kmers_to_annotate) > 0:
            print(f"Starting BLAST for {len(kmers_to_annotate)} kmers against Database {ncbi_db} using {ncbi_program}...")
            
            # We combine kmers into one FASTA-style string to save API calls
            fasta_query = "\n".join([f">kmer_{i}\n{k}" for i, k in enumerate(kmers_to_annotate)])
            
            try:
                # qblast parameters for short sequences:
                # - program: blastn
                # - database: nt (nucleotide)
                # - entrez_query: Restrict to Viruses
                # - word_size: 7 (minimum for blastn)
                # - expect: 1000 (higher to catch short hits)
                result_handle = NCBIWWW.qblast(
                    program=ncbi_program, 
                    database=ncbi_db, 
                    sequence=fasta_query,
                    entrez_query=tax_origin,
                    word_size=10,
                    expect=expect,
                    short_query=True,
                    hitlist_size=acc_num
                )
            except Exception as e:
                print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")}   [ERROR]: BLAST search failed: {e}', file=self.logfile)
                return pd.DataFrame() # Return empty DataFrame on failure
            
            try:
                #Read fully to check for completion
                blast_results_raw = result_handle.read()
                result_handle.close()

                with open(ncbi_logs, "w") as logf:
                    logf.write(f"BLAST search parameters:\nProgram: {ncbi_program}\nDatabase: {ncbi_db}\nEntrez Query: {tax_origin}\nWord Size: 10\nExpect Threshold: {expect}\nShort Query: True\nHitlist Size: {acc_num}\n\n")
                    logf.write("Raw BLAST results:\n")
                    logf.write(blast_results_raw)

                    if "</BlastOutput>" not in blast_results_raw:
                        if self.logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} ERROR: NCBI returned incomplete XML (truncated). Try a smaller kmer batch.', file=self.logfile)
                        return pd.DataFrame() # Or handle as needed

                    from io import StringIO
                    blast_records = list(NCBIXML.parse(StringIO(blast_results_raw)))
                    
                    # Use enumerate to match records back to the original kmer_list
                    results = []
                    logf.write(f"\nParsed {len(blast_records)} BLAST records.\n")
                    for i, record in enumerate(tqdm(blast_records, desc="Processing BLAST records")):
                        logf.write(f"\n--- Processing Record {i+1}/{len(blast_records)} for Kmer: {kmer_list[i]} ---\n")
                        logf.write(f"Record has {len(record.alignments)} alignments.\n")
                        
                        # Safety check: ensure we match the right kmer
                        # NCBI sometimes skips records if NO hits are found at all
                        kmer_seq = kmer_list[i] 
                        #print(f"\n--- Results for Kmer: {kmer_seq} ---")
                        
                        if not record.alignments:
                            logf.write(f"{record.description}\n")
                            logf.write("  [INFO] No significant hits found for this kmer.\n")
                            if self.logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} No significant hits found.', file=self.logfile)
                            continue
                        
                        output = []
                        gene_found = []
                        function_found = []
                        results_inner = []
                        for alignment in record.alignments:
                            logf.write(f"\nProcessing alignment: {alignment.accession} | {alignment.title}\n")
                            accession = alignment.accession
                            hit_def = alignment.title
                            output.append(f"Checking Gene in Hit: {accession}")
                            
                            # We use a try-block for Entrez in case one specific ID fails
                            try:
                                # 4. FETCH: Use small sleep to avoid 429 Too Many Requests
                                sleep(0.5) 
                                handle = Entrez.efetch(db="nucleotide", id=accession, rettype="gb", retmode="text")
                                genbank_rec = SeqIO.read(handle, "genbank")
                                handle.close()
                                
                                hsp = alignment.hsps[0]
                                start, end = min(hsp.sbjct_start, hsp.sbjct_end), max(hsp.sbjct_start, hsp.sbjct_end)
                                
                                found_gene = False
                                for feature in genbank_rec.features:
                                    if feature.type == "CDS":
                                        if start >= feature.location.start and end <= feature.location.end:
                                            product = feature.qualifiers.get('product', ['Unknown'])[0]
                                            gene = feature.qualifiers.get('gene', ['N/A'])[0]
                                            #print(f"  [MATCH] Found in Gene: {gene} | Function: {product}")
                                            #gene_found.append(gene)
                                            #function_found.append(product)
                                            found_gene = True
                                            # Populate results_inner 
                                            results_inner.append({"Kmer": kmer_seq, "Gene": gene, "Function": product})
                                            break
                                if not found_gene:
                                    results_inner.append({"Kmer": kmer_seq, "Gene": "N/A", "Function": "N/A"})
                                #     output.append("  [INFO] Hit is in an intergenic/non-coding region.")
                                    
                            except Exception as e:
                                #output.append(f"  [ERROR] Could not fetch details for {accession}: {e}")
                                if self.logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")}   [ERROR] Could not fetch details for {accession}: {e}', file=self.logfile)
                                results_inner.append({"Kmer": kmer_seq, "Gene": "Error Fetching", "Function": "Error Fetching"})

                        results.extend(results_inner)

            except Exception as e:
                if self.logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")}   [ERROR]: BLAST search failed: {e}', file=self.logfile)
                return pd.DataFrame()

            # terminate if results is empty
            if not results:
                if self.logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")}   [INFO]: No results to save after BLAST search.', file=self.logfile)
                return pd.DataFrame() 

            try:  # Try to convert results to DataFrame and save raw results to CSV for record-keeping
                results_df = pd.DataFrame(results)
                results_df.to_csv(outfile, index=False)
                if self.logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")}   [INFO]: Results saved to {outfile}', file=self.logfile)
            except Exception as e:
                if self.logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")}   [ERROR]: Failed to save results to CSV: {e}', file=self.logfile)
                return pd.DataFrame()

            try:    
                if summarise_by == "gene":
                    # Group by Kmer and find the most common gene
                    results_df = results_df.groupby('Kmer')['Gene'].agg(lambda x: x.value_counts().idxmax()).reset_index()
                elif summarise_by == "function":
                    # Group by Kmer and find the most common function
                    results_df = results_df.groupby('Kmer')['Function'].agg(lambda x: x.value_counts().idxmax()).reset_index()
            except Exception as e:
                if self.logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")}   [ERROR]: Error during summarization: {e}', file=self.logfile)

            try:
                self._save_kmer_annotations(results_df)
                if self.logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")}   [INFO]: Kmer annotations saved to local database.', file=self.logfile)
            except Exception as e:
                if self.logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")}   [ERROR]: Failed to save kmer annotations: {e}', file=self.logfile)

            # Merge existing annotations with new results if input was a DataFrame, and return the final annotated DataFrame
            results_df = pd.merge(existing_annot_df, results_df, on ="Kmer", how='left')
            if 'Kmer' in results_df.columns:
                results_df = results_df.drop(columns=['Kmer'])
            if 'feature_index' in results_df.columns:
                feature_index = results_df.pop('feature_index')
                results_df.insert(0, 'feature_index', feature_index)
        
        else:
            if self.logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")}   [INFO]: All kmers already have annotations in local database. No BLAST search needed.', file=self.logfile)
            results_df = existing_annot_df

        return results_df

    def plot_annotated_kmer_statistics(self, blast_results):
        if self.logging: print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Starting plotting of annotated kmer statistics...', file=self.logfile)
        organisms = sorted(blast_results["organism"].dropna().unique())
        n_orgs = len(organisms)

        fig, axes = plt.subplots(2, n_orgs, figsize=(7 * n_orgs, 12), sharey="row")


        # Ensure axes is always 2D: [row][col]
        if n_orgs == 1:
            axes = [[axes[0]], [axes[1]]]

        for i, org in enumerate(organisms):
        # Row 1: Function
            ax_func = axes[0][i]
            subset_func = blast_results[(blast_results["organism"] == org) & (blast_results["Function"].notna())]
            order_func = subset_func["Function"].value_counts().index

            if subset_func.empty:
                ax_func.text(0.5, 0.5, "No Function data", ha="center", va="center", transform=ax_func.transAxes)
                ax_func.set_xticks([])
            else:
                sns.countplot(data=subset_func, x="Function", order=order_func, ax=ax_func)
                ax_func.tick_params(axis="x", rotation=45)
                for lbl in ax_func.get_xticklabels():
                    lbl.set_ha("right")

            ax_func.set_title(f"{org} count of annotated Kmer Functions")
            ax_func.set_xlabel("Function")
            ax_func.set_ylabel("Count")

        # Row 2: Gene
            ax_gene = axes[1][i]
            subset_gene = blast_results[(blast_results["organism"] == org) & (blast_results["Gene"].notna())]
            order_gene = subset_gene["Gene"].value_counts().index

            if subset_gene.empty:
                ax_gene.text(0.5, 0.5, "No Gene data", ha="center", va="center", transform=ax_gene.transAxes)
                ax_gene.set_xticks([])
            else:
                sns.countplot(data=subset_gene, x="Gene", order=order_gene, ax=ax_gene)
                ax_gene.tick_params(axis="x", rotation=45)
                for lbl in ax_gene.get_xticklabels():
                    lbl.set_ha("right")

            ax_gene.set_title(f"{org} count of annotated Kmer Genes")
            ax_gene.set_xlabel("Gene")
            ax_gene.set_ylabel("Count")

        plt.tight_layout()
        if self.logging: 
            outname = 'annotated_kmer_stats.png'
            plt.savefig(self.outdir+outname)
            print(f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} Finished plotting of annotated kmer statistics. Plot saved to {self.outdir+outname}', file=self.logfile)


    def assign_gene_clusters(self, rank_df):
        """
        Assign kmers to gene clusters based on functional annotations obtained from the BLAST search. 
        """
        pass