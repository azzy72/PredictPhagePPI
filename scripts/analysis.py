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
from sklearn.metrics import roc_curve, roc_auc_score


##### Paths -------------
raw_data_path = "../raw_data/"
data_prod_path = "../data_prod/"

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
    plt.show()

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

    plt.show()

def plot_losses(train_losses, valid_losses, n_epochs, title=None):
    # Plotting the losses 
    fig,ax = plt.subplots(1,1, figsize=(9,5))
    ax.plot(range(n_epochs), train_losses, label='Train loss', c='b')
    ax.plot(range(n_epochs), valid_losses, label='Valid loss', c='m')
    ax.legend()
    if title is not None:
        fig.suptitle(title)
    fig.show()