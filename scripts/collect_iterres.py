#!/net/domus/home/people/s215045/miniconda3/bin/python

import os
import re
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from paths import data_prod_path, path_to_nn_runs
outdir_default = data_prod_path + "iterExclClus/"
print(path_to_nn_runs)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Metric Extraction Script")
    parser.add_argument("--base_dir", type=str, default=path_to_nn_runs, 
                        help="Base directory containing run folders")
    parser.add_argument("--outdir", type=str, default=outdir_default,
                        help="Directory to save output graphs and CSV")
    return parser.parse_args()

def extract_metrics_from_log(file_path):
    """Parses a single log file for key performance metrics."""
    with open(file_path, 'r') as f:
        content = f.read()

    metrics = {
        'file_path': file_path,
        'test_accuracy': None,
        'test_balanced_accuracy': None,
        'precision': None,
        'recall': None,
        'f1': None,
        'TN': None, 'FN': None, 'FP': None, 'TP': None
    }
    
    # Extract Params dict from logfile
    params_match = re.search(r"^Params:\s*(\{.*\})\s*$", content, re.MULTILINE)
    if not params_match:
        print(f"Could not find Params line in {file_path}")
        return None

    params_str = params_match.group(1)
    try:
        params_dict = ast.literal_eval(params_str)

        # Preferred format: 'nk': [n, k]
        nk = params_dict.get('nk')
        if isinstance(nk, (list, tuple)) and len(nk) >= 2:
            metrics['n'] = nk[0]
            metrics['k'] = nk[1]
        else:
            # Backward compatibility if logs contain direct keys
            metrics['n'] = params_dict.get('n')
            metrics['k'] = params_dict.get('k')

        if metrics['n'] is None or metrics['k'] is None:
            print(f"Could not extract n/k from Params in {file_path}")
            return None

    except Exception as e:
        print(f"Error parsing Params in {file_path}: {e}")
        return None

    # Extract Accuracy 
    acc_match = re.search(r"test accuracy:\s+([\d.]+)", content)
    if acc_match:
        metrics['test_accuracy'] = float(acc_match.group(1))

    # Extract Balanced Accuracy
    ba_match = re.search(r"test balanced accuracy:\s+([\d.]+)", content)
    if ba_match:
        metrics['test_balanced_accuracy'] = float(ba_match.group(1))

    # Extract Baseline Metrics (Precision, Recall, F1) [cite: 29]
    base_metrics = re.search(r"Baseline .* Precision:\s+([\d.]+),\s+Recall:\s+([\d.]+),\s+F1:\s+([\d.]+)", content)
    if base_metrics:
        metrics['precision'] = float(base_metrics.group(1))
        metrics['recall'] = float(base_metrics.group(2))
        metrics['f1'] = float(base_metrics.group(3))

    # Extract Confusion Matrix components [cite: 31]
    # Log format: [TN FN] \n [FP TP]
    cm_pattern = r"Confusion matrix:.*?\[\s*(\d+)\s+(\d+)\s*\].*?\[\s*(\d+)\s+(\d+)\s*\]"
    cm_match = re.search(cm_pattern, content, re.DOTALL)
    
    if cm_match:
        metrics['TN'] = int(cm_match.group(1))
        metrics['FN'] = int(cm_match.group(2))
        metrics['FP'] = int(cm_match.group(3))
        metrics['TP'] = int(cm_match.group(4))

    return metrics

def _plot_cm_bars(df, outdir, title_suffix=""):
    # 1. Prepare the Confusion Matrix Data
    # We melt the dataframe so 'TN', 'FN', 'FP', 'TP' become categories in one column
    cm_cols = ['TN', 'FN', 'FP', 'TP']
    cm_df = df[cm_cols + ['folder']].copy()
    cm_melted = cm_df.melt(id_vars='folder', var_name='Metric', value_name='Count')

    # 2. Setup the Plot
    plt.figure(figsize=(16, 7)) # Wider figure for many bars
    sns.set_style("whitegrid")
    
    # 3. Create the Grouped Barplot
    # x='folder' creates the groups, hue='Metric' creates the individual bars per group
    ax = sns.barplot(
        data=cm_melted, 
        x='folder', 
        y='Count', 
        hue='Metric', 
        palette='muted', # 'muted' or 'Set2' works well here
        edgecolor='black'
    )

    # 4. Add labels on top of bars (similar to your reference image)
    for p in ax.patches:
        if p.get_height() > 0: # Only label bars with a value
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        fontsize=9, color='black', 
                        xytext=(0, 7), 
                        textcoords='offset points')

    # 5. Formatting
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Confusion Matrix Components by Run {title_suffix}', fontsize=16, weight='bold', pad=20)
    plt.ylabel('Count (Number of Samples)', fontsize=12)
    plt.xlabel('Run / Configuration', fontsize=12)
    
    # Place legend outside to the right
    plt.legend(title='Metrics', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    
    # 6. Save
    plt.savefig(outdir + 'confusion_matrix_by_run.png', dpi=300)
    plt.close() # Close to free up memory
    print("Saved: confusion_matrix_by_run.png")

def _plot_bars(df, x_col, y_col, hue_col=None, title="", ylabel="", outpath=""):
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    if hue_col:
        ax = sns.barplot(
            data=df, 
            x=x_col, 
            y=y_col, 
            hue=hue_col, 
            palette='viridis',
            edgecolor='black'
        )
    else:
        ax = sns.barplot(
            data=df, 
            x=x_col, 
            y=y_col, 
            edgecolor='black'
        )
        plt.xticks(rotation=45, ha='right')

    plt.title(title, fontsize=15, pad=15)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel(x_col.capitalize(), fontsize=12)
    if hue_col: plt.legend(title=hue_col.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(outpath)
    print(f"Saved: {outpath}")

def plot_graphs(df, outdir=outdir_default):
    # Ensure all columns are integers for proper sorting
    df['n'] = pd.to_numeric(df['n'])
    df['k'] = pd.to_numeric(df['k'])
    df['test_accuracy'] = pd.to_numeric(df['test_accuracy'])
    df['test_balanced_accuracy'] = pd.to_numeric(df['test_balanced_accuracy'])
    df['precision'] = pd.to_numeric(df['precision'])
    df['recall'] = pd.to_numeric(df['recall'])
    df['f1'] = pd.to_numeric(df['f1'])

    if len(set(df['n'])) == 1 and len(set(df['k'])) == 1:
        print("All runs have the same 'n' & 'k' value. Results will be plotted based on rows and not grouped by 'n' and 'k'.")
        singular_nk = True
    else:
        df = df.sort_values(['n', 'k'])
        singular_nk = False

    title_suffix = "by N and K" if not singular_nk else "for single N and K"

    print("Recognized metrics:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    print(df[['n', 'k', 'test_accuracy', 'precision', 'recall', 'f1']])

    # --- Graph 1: Grouped Accuracy Bar Chart (X=n, Hue=k) ---
    _plot_bars(
        df=df, 
        x_col='n', 
        y_col='test_accuracy', 
        hue_col='k' if not singular_nk else None, 
        title=f'FFNN Test Accuracy {title_suffix}', 
        ylabel='Test Accuracy',
        outpath=outdir + 'accuracy_by_nk.png'
    )

    # --- Graph 2: Grouped Balanced Accuracy Bar Chart (X=n, Hue=k) ---
    _plot_bars(
        df=df, 
        x_col='n', 
        y_col='test_balanced_accuracy', 
        hue_col='k' if not singular_nk else None, 
        title=f'FFNN Test Balanced Accuracy {title_suffix}', 
        ylabel='Test Balanced Accuracy',
        outpath=outdir + 'balanced_accuracy_by_nk.png'
    )

    # --- Graph 3: Grouped F1 Bar Chart (X=n, Hue=k) ---
    _plot_bars(
        df=df, 
        x_col='n', 
        y_col='f1', 
        hue_col='k' if not singular_nk else None, 
        title=f'FFNN F1 score {title_suffix}', 
        ylabel='Test F1',
        outpath=outdir + 'f1_by_nk.png'
    )

    # --- Graph 4: Confusion Matrix as bars ---
    _plot_cm_bars(df, outdir, title_suffix=title_suffix)

    # --- Graph 5: Averaged Confusion Matrix ---
    avg_cm = df[['TN', 'FN', 'FP', 'TP']].mean()
    # Reshape into 2x2 matrix: [[TN, FN], [FP, TP]]
    cm_data = np.array([[avg_cm['TN'], avg_cm['FN']], 
                        [avg_cm['FP'], avg_cm['TP']]])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_data, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title(f'Global Averaged Confusion Matrix\n(Across {len(df)} runs)', fontsize=15, pad=15)
    plt.tight_layout()
    plt.savefig(outdir + 'averaged_confusion_matrix.png')
    print("Saved: averaged_confusion_matrix.png")

def main(base_dir=path_to_nn_runs, outdir=outdir_default):
    all_data = []
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found.")
        return
    
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        print(f"Created output directory: {outdir}")

    # Iterate through all folders in nn_runs
    for folder_name in os.listdir(base_dir):          
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            # Search for log files in this specific run folder
            for file in os.listdir(folder_path):
                if file.endswith(".txt") or file.endswith(".log"):
                    log_path = os.path.join(folder_path, file)
                    metrics = extract_metrics_from_log(log_path)
                    if metrics and metrics['test_accuracy'] is not None:
                        metrics['folder'] = folder_name
                        all_data.append(metrics)

    if all_data:
        df = pd.DataFrame(all_data)
        plot_graphs(df, outdir=outdir)
        # Optional: save the raw data for inspection
        df.to_csv(outdir +'all_runs_summary.csv', index=False)
        print("Summary CSV saved as all_runs_summary.csv")
    else:
        print("No valid data found to plot.")

if __name__ == "__main__":
    if parse_arguments().base_dir:
        main(base_dir=parse_arguments().base_dir, outdir=parse_arguments().outdir)
    else:
        main()