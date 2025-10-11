########################
#####  Analysis.py  ####
########################
# Contains functions for analysis and plotting

##### Imports -----------
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

##### Paths -------------
raw_data_path = "../raw_data/"
data_prod_path = "../data_prod/"

def perform_pca(data: pd.DataFrame, n_components=2) -> (PCA, np.array, np.array):
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
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Perform PCA
    pca = PCA(n_components=n_components)
    score = pca.fit_transform(data_scaled)
    coeff = pca.components_.T
    return pca, score, coeff


def pca_biplot(score, coeff, PCA = None, data = None, labels=None, n_features_to_plot=10, hide_labels=[False, False]) -> None:
    """
    Generates a PCA biplot with dynamically scaled arrows for loadings and 
    uses adjust_text to prevent label overlap.

    Args:
        score (np.array): PCA scores (data points/samples).
        coeff (np.array): PCA components (eigenvectors/feature directions).
        labels (list): Feature labels (e.g., k-mers).
        n_features_to_plot (int): Maximum number of features (arrows) to display.
        hide_labels (list of bool): [hide_score_labels, hide_loading_labels]. If True, hides the respective labels.
    
    Returns:
        None (displays the plot)
    """
    if data is None and PCA is None:
        raise ValueError("Either 'data' or 'PCA' must be provided.")
    
    if data is not None and PCA is None: #construct PCA from data
        pca, score, coeff = perform_pca(data)
        components_df = pd.DataFrame(score, columns=[f'PC{i+1}' for i in range(score.shape[1])], index=data.index)
    
    elif data is None and PCA is not None: #use provided PCA
        pca = PCA
        components_df = pd.DataFrame(score, columns=[f'PC{i+1}' for i in range(score.shape[1])])

    xs = score[:, 0]
    ys = score[:, 1]
    
    # 1. Calculate a dynamic scaling factor for the loading arrows
    # This factor ensures the longest arrow is proportional to the max spread of the scores.
    max_coeff_length = np.max(np.sqrt(coeff[:, 0]**2 + coeff[:, 1]**2))
    max_score_extent = np.max(np.abs(score))
    
    # Aim for the longest arrow to reach about 80% of the maximum score axis extent.
    scale_factor = (max_score_extent * 0.8) / max_coeff_length

    plt.figure(figsize=(12, 10))
    
    # 2. Plot the scores (Phage samples)
    plt.scatter(xs, ys, c=np.arange(len(xs)), s=50)
    if not hide_labels[0]:
         # Annotate Points with Sequence IDs
        texts = [
            plt.text(
                components_df.PC1[i] + 1.5,
                components_df.PC2[i] + 2.5,
                components_df.index[i].split("_")[-1],
                fontsize=9
            )
            for i in range(components_df.shape[0])
        ]
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'))

    # Prepare list for adjust_text
    texts = []

    # 3. Plot the loadings (K-mers)
    for i in range(min(len(coeff), n_features_to_plot)):
        # Apply the scaling factor
        x_arrow = coeff[i, 0] * scale_factor
        y_arrow = coeff[i, 1] * scale_factor
        
        # Draw the arrow
        plt.arrow(0, 0, x_arrow, y_arrow, 
                  color='black', 
                  alpha=0.3, 
                  linewidth=1.25,
                  head_width=2,  # Adjust size of the arrow head
                  head_length=2.5,
                  overhang=0.25)
        
        if not hide_labels[1]:
            # Add the text label and collect it for adjust_text
            texts.append(plt.text(x_arrow * 1.05, y_arrow * 1.05, 
                                labels[i], 
                                color='black', 
                                fontsize=6))

    # Use adjust_text to automatically position the labels without overlap
    if texts:
        adjust_text(texts, 
                    arrowprops=dict(arrowstyle="-", color='k', lw=0.5, alpha=0.6))

    # Set axes and labels
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}% Variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}% Variance)')
    plt.title("PCA Biplot with Scaled Loadings")
    plt.grid(True, linestyle='--')
    plt.axhline(0, color='gray', linewidth=0.5, linestyle='-')
    plt.axvline(0, color='gray', linewidth=0.5, linestyle='-')
    plt.show()
