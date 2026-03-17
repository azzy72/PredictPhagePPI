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
outdir = data_prod_path + "iterNK/"
print(path_to_nn_runs)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Metric Extraction Script")
    parser.add_argument("--base_dir", type=str, default=path_to_nn_runs, 
                        help="Base directory containing run folders")
    return parser.parse_args()

def extract_metrics_from_log(file_path):
    """Parses a single log file for key performance metrics."""
    with open(file_path, 'r') as f:
        content = f.read()

    metrics = {
        'file_path': file_path,
        'test_accuracy': None,
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

def plot_graphs(df):
    # Ensure all columns are integers for proper sorting
    df['n'] = pd.to_numeric(df['n'])
    df['k'] = pd.to_numeric(df['k'])
    df['test_accuracy'] = pd.to_numeric(df['test_accuracy'])
    df['precision'] = pd.to_numeric(df['precision'])
    df['recall'] = pd.to_numeric(df['recall'])
    df['f1'] = pd.to_numeric(df['f1'])
    df = df.sort_values(['n', 'k'])

    print("Recognized metrics:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    print(df[['n', 'k', 'test_accuracy', 'precision', 'recall', 'f1']])

    # --- Graph 1: Grouped Accuracy Bar Chart (X=n, Hue=k) ---
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    ax = sns.barplot(
        data=df, 
        x='n', 
        y='test_accuracy', 
        hue='k', 
        palette='viridis',
        edgecolor='black'
    )
    
    plt.title('FFNN Test Accuracy by N and K', fontsize=15, pad=15)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.xlabel('n (Number of Features/Samples)', fontsize=12)
    plt.legend(title='k values', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0.8, 1.0) # Adjust based on your performance range
    plt.tight_layout()
    plt.savefig(outdir + 'accuracy_by_nk.png')
    print("Saved: accuracy_by_nk.png")

    # --- Graph 2: Grouped F1 Bar Chart (X=n, Hue=k) ---
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    ax = sns.barplot(
        data=df, 
        x='n', 
        y='f1', 
        hue='k', 
        palette='viridis',
        edgecolor='black'
    )
    
    for i, bar in enumerate(ax.patches):
        # Find corresponding row in df
        n = bar.get_x() + bar.get_width() / 2
        height = bar.get_height()
        # Get index for the bar (barplot arranges bars sequentially)
        idx = i % len(df)
        recall = df.iloc[idx]['recall']
        precision = df.iloc[idx]['precision']
        label = f'R:{recall:.2f}\nP:{precision:.2f}'
        ax.text(
            bar.get_x() + bar.get_width() / 2, height + 0.01,
            label,
            ha='center', va='bottom', fontsize=6, color='black'
        )
    
    plt.title('FFNN F1 score by N and K', fontsize=15, pad=15)
    plt.ylabel('Test F1', fontsize=12)
    plt.xlabel('n (Number of Features/Samples)', fontsize=12)
    plt.legend(title='k values', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0.2, 1.0)
    plt.tight_layout()
    plt.savefig(outdir + 'f1_by_nk.png')
    print("Saved: f1_by_nk.png")

    # --- Graph 3: Averaged Confusion Matrix ---
    avg_cm = df[['TN', 'FN', 'FP', 'TP']].mean()
    # Reshape into 2x2 matrix: [[TN, FN], [FP, TP]]
    cm_data = np.array([[avg_cm['TN'], avg_cm['FN']], 
                        [avg_cm['FP'], avg_cm['TP']]])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_data, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title(f'Global Averaged Confusion Matrix\n(Across {len(df)} runs)')
    plt.tight_layout()
    plt.savefig(outdir + 'averaged_confusion_matrix.png')
    print("Saved: averaged_confusion_matrix.png")

def main(base_dir=path_to_nn_runs):
    all_data = []
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found.")
        return

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
        plot_graphs(df)
        # Optional: save the raw data for inspection
        df.to_csv(outdir +'all_runs_summary.csv', index=False)
        print("Summary CSV saved as all_runs_summary.csv")
    else:
        print("No valid data found to plot.")

if __name__ == "__main__":
    if parse_arguments().base_dir:
        main(base_dir=parse_arguments().base_dir)
    else:
        main()