#!/net/domus/home/people/s215045/miniconda3/bin/python

import os
import re
import ast
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import numpy as np
from tqdm import tqdm
import argparse
from decimal import Decimal
from time import time, sleep
from datetime import datetime
from paths import data_prod_path, path_to_nn_runs
from analysis import GeneAnalysis, PFI_Lookup
import json
outdir_default = data_prod_path + "iterExclClus/"

# Global logger for capturing both stdout and file output
class DualLogger:
    def __init__(self, logfile=None):
        self.logfile = logfile
    
    def log(self, message="", end="\n"):
        """Write message to both stdout and logfile."""
        print(message, end=end)
        if self.logfile:
            self.logfile.write(message + end)
            self.logfile.flush()
    
    def set_logfile(self, logfile):
        self.logfile = logfile

logger = DualLogger()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Metric Extraction Script")
    parser.add_argument("--base_dir", type=str, default=path_to_nn_runs, 
                        help="Base directory containing run folders")
    parser.add_argument("--out_dir", type=str, default=outdir_default,
                        help="Directory to save output graphs and CSV")
    parser.add_argument("--hk_lookup_path", type=str, default=None,
                        help="Path to the HK lookup table")
    
    #Flags
    parser.add_argument("--weight_pfi", action='store_true', 
                        help="Whether to weight the PFI scores by the corresponding test (balanced) accuracy of each run")
    parser.add_argument("--top_kmers", type=int, default=500,
                        help="Number of top k-mers to extract and analyze")
    
    ## Optional grouping arguments for more flexible plotting 
    parser.add_argument("--show_cm_bar_percentage", action='store_true',
                        help="Whether to display percentages on confusion matrix bars")
    parser.add_argument("--x_col", type=str, default=None,
                        help="Column to use for x-axis in plots")
    parser.add_argument("--hue_col", type=str, default=None,
                        help="Column to use for color coding in plots")
    parser.add_argument("--group_x_col", type=str, default=None,
                        help="Column to use for grouping x-axis in plots")
    parser.add_argument("--group_hue_col", type=str, default=None,
                        help="Column to use for grouping hue in plots")
    return parser.parse_args()

def correct_deci_number(value):
    """
    In some cases the decimal values may be extracted as 568 instead of 0.568. This function checks if the value is greater than 1 and if so, divides it by the appropriate power of 10 to correct it. 
    For example, if the value is 568, it will be divided by 1000 to become 0.568. If the value is already a proper decimal (e.g., 0.568), it will be returned unchanged.
    """
    try:
        num = float(value)
        if num > 1:
            # Determine the number of digits to divide by
            digits = len(str(int(num)))
            corrected_value = num / (10 ** digits)
            return corrected_value
        else:
            return num
    
    except ValueError:
        logger.log(f"Warning: Unable to convert '{value}' to a float. Returning original value.")
        return value
    #return Decimal(value)  # Validate if it's a number

def calculate_unified_score(metrics_dict):
    """
    Unified Performance Score (UPS) Calculation:
    Calculates a single performance score from a dictionary of NN metrics.
    Weights can be adjusted based on project priorities.
    """
    # 1. Define Weights (Total = 1.0)
    # We prioritize Balanced Accuracy and Unseen Performance
    weights = {
        'test_balanced_accuracy': 0.30,
        'f1': 0.25,
        'unseen_test_balanced_accuracy': 0.45 
    }
    
    # 2. Extract values (with defaults to prevent crashes)
    b_acc = metrics_dict.get('test_balanced_accuracy', 0)
    f1 = metrics_dict.get('f1', 0)
    unseen_b_acc = metrics_dict.get('unseen_test_balanced_accuracy', 0)
    
    # 3. Calculate Weighted Score
    final_score = (
        (b_acc * weights['test_balanced_accuracy']) +
        (f1 * weights['f1']) +
        (unseen_b_acc * weights['unseen_test_balanced_accuracy'])
    )
    
    return round(final_score, 4)


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

    run_info = {
        'use_encoded' : None,
        'data2': None
    }
    
    # Extract Params from logfile
    params_dict = None

    # Preferred (legacy) format: single-line dict
    params_match = re.search(r"^Params:\s*(\{.*\})\s*$", content, re.MULTILINE)
    if params_match:
        params_str = params_match.group(1)
        try:
            params_dict = ast.literal_eval(params_str)
        except Exception as e:
            logger.log(f"Error parsing single-line Params in {file_path}: {e}")
            return None
    else:
        # New format: logger-prefixed multiline key-value block after "Params:"
        # Example lines:
        # 2026-05-12 ... - root - INFO - Params:
        # 2026-05-12 ... - root - INFO -   nk: [500, 12]
        params_dict = {}
        lines = content.splitlines()
        params_start = None

        for i, line in enumerate(lines):
            if re.search(r"\bParams:\s*$", line):
                params_start = i
                break

        if params_start is None:
            logger.log(f"Could not find Params section in {file_path}")
            return None

        param_line_re = re.compile(
            r"^(?:.*?\s-\sINFO\s-\s*)?(?P<key>[A-Za-z0-9_]+):\s*(?P<value>.*)$"
        )

        for line in lines[params_start + 1:]:
            stripped = line.strip()

            # Stop at the first non-parameter line after the Params block begins.
            if not stripped:
                if params_dict:
                    break
                continue

            match = param_line_re.match(line)
            if not match:
                if params_dict:
                    break
                continue

            key = match.group('key').strip()
            value_str = match.group('value').strip()

            # Some logger formats may leave the key/value in the tail of the line
            # after the final " - INFO - " segment, so try that fallback too.
            if not key or key.lower() == 'info':
                tail = line.split(' - INFO - ', 1)[-1].strip()
                tail_match = re.match(r"(?P<key>[A-Za-z0-9_]+):\s*(?P<value>.*)$", tail)
                if tail_match:
                    key = tail_match.group('key').strip()
                    value_str = tail_match.group('value').strip()

            if value_str == "":
                params_dict[key] = None
                continue

            try:
                params_dict[key] = ast.literal_eval(value_str)
            except Exception:
                # Keep plain (unquoted) strings as-is
                params_dict[key] = value_str

        if not params_dict:
            print(f"Could not parse any Params entries in {file_path}")
            return None

    try:
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
            logger.log(f"Could not extract n/k from Params in {file_path}")
            return None

        run_info['use_encoded'] = params_dict.get('use_encoded')
        run_info['data2'] = params_dict.get('data2')

    except Exception as e:
        logger.log(f"Error parsing Params in {file_path}: {e}")
        return None

    # Extract Accuracy 
    acc_match = re.search(r"Standard test accuracy:\s+([\d.]+)", content)
    if acc_match:
        metrics['test_accuracy'] = correct_deci_number((acc_match.group(1)))

    unseen_acc_match = re.search(r"truly unseen test accuracy:\s+([\d.]+)", content)
    if unseen_acc_match:
        metrics['unseen_test_accuracy'] = correct_deci_number(unseen_acc_match.group(1))

    # Extract Balanced Accuracy
    ba_match = re.search(r"Standard test balanced accuracy:\s+([\d.]+)", content)
    if ba_match:
        #metrics['test_balanced_accuracy'] = correct_deci_number(ba_match.group(1))
        metrics['test_balanced_accuracy'] = correct_deci_number(ba_match.group(1))
    unseen_ba_match = re.search(r"truly unseen test balanced accuracy:\s+([\d.]+)", content)
    if unseen_ba_match:
        metrics['unseen_test_balanced_accuracy'] = correct_deci_number(unseen_ba_match.group(1))

    # Extract Baseline Metrics (Precision, Recall, F1) [cite: 29]
    # Try Baseline format first: "Baseline (threshold=X) -> Precision: Y, Recall: Z, F1: W"
    base_metrics = re.search(
        r"Baseline\s*\([^)]*\)\s*(?:->|:)\s*Precision:\s*([\d.]+)\s*,\s*Recall:\s*([\d.]+)\s*,\s*F1:\s*([\d.]+)",
        content
    )
    if base_metrics:
        metrics['precision'] = correct_deci_number(base_metrics.group(1))
        metrics['recall'] = correct_deci_number(base_metrics.group(2))
        metrics['f1'] = correct_deci_number(base_metrics.group(3))
    else:
        # Try "Best threshold by F1" format: "Best threshold by F1 -> threshold=X, Precision=Y, Recall=Z, F1=W"
        best_threshold = re.search(
            r"Best\s+threshold\s+by\s+F1\s*(?:->|:)\s*(?:threshold=[^,]+,\s*)?Precision\s*=\s*([\d.]+)\s*,\s*Recall\s*=\s*([\d.]+)\s*,\s*F1\s*=\s*([\d.]+)",
            content,
            re.IGNORECASE
        )
        if best_threshold:
            metrics['precision'] = correct_deci_number(best_threshold.group(1))
            metrics['recall'] = correct_deci_number(best_threshold.group(2))
            metrics['f1'] = correct_deci_number(best_threshold.group(3))

    # Extract Confusion Matrix components [cite: 31]
    # New format: --- Confusion Matrix --- with [[TN FP] [FN TP]]
    cm_array_match = re.search(
        r"--- Confusion Matrix ---\s*\[\[\s*(\d+)\s+(\d+)\s*\]\s*\[\s*(\d+)\s+(\d+)\s*\]\s*\]",
        content,
        re.DOTALL
    )
    if cm_array_match:
        metrics['TN'] = int(cm_array_match.group(1))
        metrics['FP'] = int(cm_array_match.group(2))
        metrics['FN'] = int(cm_array_match.group(3))
        metrics['TP'] = int(cm_array_match.group(4))
    else:
        # Try legacy format: [TN FN] \n [FP TP]
        cm_pattern = r"Confusion matrix:.*?\[\s*(\d+)\s+(\d+)\s*\].*?\[\s*(\d+)\s+(\d+)\s*\]"
        cm_match = re.search(cm_pattern, content, re.DOTALL)
        if cm_match:
            metrics['TN'] = int(cm_match.group(1))
            metrics['FN'] = int(cm_match.group(2))
            metrics['FP'] = int(cm_match.group(3))
            metrics['TP'] = int(cm_match.group(4))
    
    if "INFO - Process completed in" in content:
        metrics['status'] = True
    else:
        metrics['status'] = False

    return metrics, run_info

class GAPlottingUtils:
    def __init__(self, df, outdir):
        self.df = df
        self.outdir = outdir
    
    def plot_top_genes(self, df: pd.DataFrame, entity_type : str, title_suffix: str = ""):
        """
        Plot the annotated genes found in the entity specific dataframe (df)
        """
        if entity_type == "bacterium":
            gene_counts = df['gene'].value_counts()
        elif entity_type == "phage":
            gene_counts = df['product'].value_counts()

        plt.figure(figsize=(10, 6))
        sns.barplot(x=gene_counts.index, y=gene_counts.values, palette='viridis')
        plt.title(f'Top {title_suffix} {entity_type.capitalize()} Kmers Annotated Genes')
        plt.xlabel('Gene')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.outdir + f'top_genes_{entity_type}.png')
        plt.close()
    
    def plot_kmer_distribution(self, df : pd.DataFrame, entity_type : str, title_suffix: str = ""):
        """
        Plot the distribution of k-mers across different genes for the given entity type
        """
        gene_kmer_counts = df['kmer_in_seq'].value_counts()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=gene_kmer_counts.index, y=gene_kmer_counts.values, palette='magma')
        plt.title(f'Distribution of Top {title_suffix} {entity_type.capitalize()} Kmers')
        plt.xlabel('Kmers')
        plt.ylabel('Kmer Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.outdir + f'kmer_distribution_{entity_type}.png')
        plt.close()
    
    def plot_kmer_against_ups_or_pfi(self, df: pd.DataFrame, entity_type: str, sort_by = 'UPS'):
        """
        Plot the relationship between k-mer counts and the Unified Performance Score (UPS) for the given entity type
        """
        if sort_by not in df.columns:
            logger.log("Column not found in dataframe. Cannot plot k-mer against UPS or PFI.")
            return
        
        plt.figure(figsize=(10, 6))
        title_part = "Unified Performance Score (UPS)" if sort_by == 'UPS' else "PFI Score"
        sns.scatterplot(x='kmer_in_seq', y=sort_by, data=df, hue='gene' if entity_type == 'bacterium' else 'product', palette='coolwarm')
        plt.title(f'Kmer Count vs {title_part} for {entity_type.capitalize()} Kmers')
        plt.xlabel('Kmer Count')
        plt.ylabel(title_part)
        plt.legend(title='Gene' if entity_type == 'bacterium' else 'Product', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.outdir + f'kmer_vs_{sort_by.lower()}_{entity_type}.png')
        plt.close()

class MetricPlottingUtils:
    def __init__(self, df, outdir, x_col = None, hue_col = None, x_col_by_cluster = False, x_col_by_phage = False):
        # Ensure all columns are integers for proper sorting
        df['n'] = pd.to_numeric(df['n'])
        df['k'] = pd.to_numeric(df['k'])
        df['test_accuracy'] = pd.to_numeric(df['test_accuracy'])
        df['test_balanced_accuracy'] = pd.to_numeric(df['test_balanced_accuracy'])
        df['precision'] = pd.to_numeric(df['precision'])
        df['recall'] = pd.to_numeric(df['recall'])
        df['f1'] = pd.to_numeric(df['f1'])
        df['folder'] = df['folder'].astype(str).str.replace(r'_run\d+$', '', regex=True)
        df['status'] = df['status'].apply(lambda x: 'passed' if x else 'failed')
        
        try:
            df['b_value'] = df['folder'].str.extract(r"b(\d+)")[0]
            df['p_value'] = df['folder'].str.extract(r"p(\d+)")[0]
        except Exception as e:
            print(f"Error extracting b_value and p_value from folder names: {e}")
            df['b_value'] = None
            df['p_value'] = None

        self.outdir = outdir
        self.failed_runs_flag = False

        # Automatically detect if all runs have the same 'n' and 'k' values
        self.singular_n = False
        self.singular_k = False
        if len(set(df['n'])) == 1 and len(set(df['k'])) == 1:
            print("All runs have the same 'n' & 'k' value. Results will be plotted based on rows and not grouped by 'n' and 'k'.")
            self.singular_n = True
            self.singular_k = True
        elif len(set(df['n'])) == 1:
            df = df.sort_values(['n', 'k'])
            self.singular_n = True
            self.singular_k = False
        elif len(set(df['k'])) == 1:
            df = df.sort_values(['n', 'k'])
            self.singular_n = False
            self.singular_k = True
        else:
            df = df.sort_values(['n', 'k'])
            self.singular_n = False
            self.singular_k = False
        
        self.title_suffix = "by N and K" if not (self.singular_n and self.singular_k) else "for single N and K"

        # Set x and hue columns based on the presence of singular n or k
        if x_col is None:
            self.x_col = 'folder' if self.singular_n else 'n'
        else:
            self.x_col = x_col
        if hue_col is None:
            self.hue_col = None if self.singular_k else 'k'
        else:
            self.hue_col = hue_col
        #self.x_col = x_col if not None else ('n' if not self.singular_n else 'folder')
        #self.hue_col = hue_col if not None else ('k' if not self.singular_k else None)
        print(f"Using x_col: {self.x_col}, hue_col: {self.hue_col}")

        if x_col_by_cluster or x_col_by_phage: 
            parts = df["folder"].str.extract(r"^cluster_(\d+)_([A-Z].+)$")
            df["cluster_id"] = pd.to_numeric(parts[0], errors="coerce")
            df["phage_name"] = parts[1]
            df["cluster_group"] = "cluster=" + parts[0]
            if x_col_by_phage:
                df["phage_group"] = "phage=" + parts[1]
                self.x_col = "phage_group" if x_col_by_phage else self.x_col
                # sort by phage_name for better visualization
                df = df.sort_values("phage_name")
            elif x_col_by_cluster:
                self.x_col = "cluster_group" if x_col_by_cluster else self.x_col
                # sort by cluster_id for better visualization
                df = df.sort_values("cluster_id")
        
        self.df = df

        if "failed" in self.df['status'].values:
            print("Warning: Some runs have failed! Count: ", (self.df['status'] == 'failed').sum())
            self.df_all = self.df.copy()
            if "passed" in self.df['status'].values:
                self.df = self.df[self.df['status'] == 'passed']
            else:
                raise ValueError("All runs have failed. No data to plot.")
            self.failed_runs_flag = True

        print(f"Dataframe length: {len(self.df)}. Recognized metrics:")
        for col in self.df.columns:
            print(f"  {col}: {self.df[col].dtype}")
        print(self.df)

    def _plot_cm_bars(self, title_suffix="", show_percentage=False):
        # 1. Prepare the Confusion Matrix Data
        # We melt the dataframe so 'TN', 'FN', 'FP', 'TP' become categories in one column
        cm_cols = ['TN', 'FN', 'FP', 'TP']
        cm_df = self.df[cm_cols + [self.x_col]].copy()
        cm_melted = cm_df.melt(id_vars=self.x_col, var_name='Metric', value_name='Count')
        cm_melted['Metric'] = cm_melted['folder'] + '_' + cm_melted['Metric'] # Combine folder and metric for unique bars
        
        # Calculate percentages if requested
        if show_percentage:
            # Calculate total for each x_col group
            cm_melted['Total'] = cm_melted.groupby(self.x_col)['Count'].transform('sum')
            cm_melted['Percentage'] = (cm_melted['Count'] / cm_melted['Total']) * 100
            plot_value = 'Percentage'
            y_label = 'Percentage (%)'
        else:
            plot_value = 'Count'
            y_label = 'Count (Number of Samples)'
        
        cm_melted = cm_melted.sort_values(by=[self.x_col, 'Metric']) # sort by folder and then by metric for consistent ordering
        print("Prepared confusion matrix data for bar plot:")
        print(cm_melted)

        # 2. Setup the Plot
        plt.figure(figsize=(16, 7)) # Wider figure for many bars
        sns.set_style("whitegrid")
        
        # 3. Create the Grouped Barplot
        # Extract the metric suffixes (TN, FN, FP, TP) and create a color mapping
        cm_melted['MetricType'] = cm_melted['Metric'].str.extract(r'_(TN|FN|FP|TP)$')
        metric_colors = {'TN': '#FF6B6B', 'FN': '#4ECDC4', 'FP': '#45B7D1', 'TP': '#B7FF78'}
        cm_melted['Color'] = cm_melted['MetricType'].map(metric_colors)
        
        ax = sns.barplot(
            data=cm_melted, 
            x=self.x_col, 
            y=plot_value, 
            hue='MetricType', 
            palette=metric_colors, 
            edgecolor='black',
        )
        plt.legend(title='Confusion Matrix Metric', bbox_to_anchor=(1.05, 1), loc='upper left')

        # 4. Add labels on top of bars (similar to your reference image)
        if len(cm_melted) < 50:  # Only add labels if there aren't too many bars to avoid clutter
            for p in ax.patches:
                height = p.get_height()
                if show_percentage:
                    label = f'{height:.1f}%'
                else:
                    label = f'{int(height)}'
                ax.annotate(label, 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='center', 
                            fontsize=4, color='black', 
                            xytext=(0, 7),
                            textcoords='offset points')

        # 5. Formatting
        #ax.tick_params(axis='x', labelsize=8)
        plt.xticks(rotation=90, ha='right')
        plt.title(f'Confusion Matrix Components by Run {title_suffix}', fontsize=14, weight='bold', pad=20)
        plt.ylabel(y_label, fontsize=10)
        plt.xlabel('Run / Configuration', fontsize=6)
        
        # Place legend outside to the right
        plt.legend(title='Metrics', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        
        plt.tight_layout()
        
        # 6. Save
        filename = 'confusion_matrix_by_run_percentage.png' if show_percentage else 'confusion_matrix_by_run.png'
        plt.savefig(self.outdir + filename, dpi=300)
        plt.close() # Close to free up memory
        print(f"Saved: {filename}")

    def _plot_bars(self, x_col, y_col, hue_col=None, title="", ylabel="", outpath=""):
        plt.figure(figsize=(12, 7))
        sns.set_style("whitegrid")
        
        if hue_col:
            ax = sns.barplot(
                data=self.df, 
                x=x_col, 
                y=y_col, 
                hue=hue_col, 
                palette='viridis',
                edgecolor='black'
            )
        else:
            ax = sns.barplot(
                data=self.df, 
                x=x_col, 
                y=y_col, 
                edgecolor='black'
            )
            plt.xticks(rotation=45, ha='right')

        plt.title(title, fontsize=15, pad=15)
        plt.ylabel(ylabel, fontsize=12)
        plt.xlabel(x_col.capitalize(), fontsize=6)
        if hue_col: plt.legend(title=hue_col.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(outpath)
        print(f"Saved: {outpath}")

    def _plot_bp_heatmap(self, title_suffix=""):
        # Calculate success rate for each combination
        heatmap_data = self.df_all.groupby(['b_value', 'p_value'])['status'].value_counts(normalize=True).unstack().fillna(0)
        # We only care about the 'passed' percentage
        passed_rate = heatmap_data['passed'].unstack()

        plt.figure(figsize=(10, 8))
        cmap = ListedColormap(["#D4840DE8", "#2E78E0D9"])  # muted orange, muted blue
        norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

        sns.heatmap(
            passed_rate,
            annot=True,
            fmt=".0f",
            cmap=cmap,
            norm=norm,
            cbar_kws={"label": "Success Rate", "ticks": [0, 1]}
        )

        plt.title("Success Rate by Phage and Bacterial Clusters")
        plt.xlabel("Phage Clusters")
        plt.ylabel("Bacterial Clusters")
        plt.tight_layout()
        plt.savefig(self.outdir + 'bp_cluster_heatmap.png', dpi=300)
        plt.close() # Close to free up memory
        print("Saved: bp_cluster_heatmap.png")

    def plot_graphs(self):
        # --- Graph 1: Grouped Accuracy Bar Chart (X=n, Hue=k) ---
        self._plot_bars(
            x_col=self.x_col,
            y_col='test_accuracy', 
            hue_col=self.hue_col, 
            title=f'FFNN Test Accuracy {self.title_suffix}', 
            ylabel='Test Accuracy',
            outpath=self.outdir + 'accuracy_by_nk.png'
        )

        # --- Graph 2: Grouped Balanced Accuracy Bar Chart (X=n, Hue=k) ---
        self._plot_bars(
            x_col=self.x_col, 
            y_col='test_balanced_accuracy', 
            hue_col=self.hue_col, 
            title=f'FFNN Test Balanced Accuracy {self.title_suffix}', 
            ylabel='Test Balanced Accuracy',
            outpath=self.outdir + 'balanced_accuracy_by_nk.png'
        )

        # --- Graph 3: Grouped Truly Unseen Accuracy Bar Chart (X=n, Hue=k) ---
        self._plot_bars(
            x_col=self.x_col, 
            y_col='unseen_test_accuracy', 
            hue_col=self.hue_col, 
            title=f'FFNN Truly Unseen Test Accuracy {self.title_suffix}', 
            ylabel='Truly Unseen Test Accuracy',
            outpath=self.outdir + 'unseen_accuracy_by_nk.png'
        )

        # --- Graph 4: Grouped Truly Unseen Balanced Accuracy Bar Chart (X=n, Hue=k) ---
        self._plot_bars(
            x_col=self.x_col, 
            y_col='unseen_test_balanced_accuracy', 
            hue_col=self.hue_col, 
            title=f'FFNN Truly Unseen Test Balanced Accuracy {self.title_suffix}', 
            ylabel='Truly Unseen Test Balanced Accuracy',
            outpath=self.outdir + 'unseen_balanced_accuracy_by_nk.png'
        )

        # --- Graph 5: Grouped F1 Bar Chart (X=n, Hue=k) ---
        self._plot_bars(
            x_col=self.x_col, 
            y_col='f1', 
            hue_col=self.hue_col, 
            title=f'FFNN F1 score {self.title_suffix}', 
            ylabel='Test F1',
            outpath=self.outdir + 'f1_by_nk.png'
        )

        # --- Graph 6: Confusion Matrix as bars ---
        self._plot_cm_bars(title_suffix=self.title_suffix, show_percentage=args.show_cm_bar_percentage)

        # --- Graph 7: Averaged Confusion Matrix ---
        avg_cm = self.df[['TN', 'FN', 'FP', 'TP']].mean()
        # Reshape into 2x2 matrix: [[TN, FN], [FP, TP]]
        cm_data = np.array([[avg_cm['TN'], avg_cm['FN']], 
                            [avg_cm['FP'], avg_cm['TP']]])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_data, annot=True, fmt='.1f', cmap='Blues', 
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        plt.title(f'Global Averaged Confusion Matrix\n(Across {len(self.df)} runs)', fontsize=15, pad=15)
        plt.tight_layout()
        plt.savefig(self.outdir + 'averaged_confusion_matrix.png')
        print("Saved: averaged_confusion_matrix.png")

        # --- Graph 8: Heatmap of success rate by bacterial and phage clusters ---
        if self.failed_runs_flag:
            self._plot_bp_heatmap(title_suffix=self.title_suffix)

def open_hk_lookup(hk_lookup_path, reverse=True):
    if hk_lookup_path and os.path.exists(hk_lookup_path):
        with open(hk_lookup_path, 'r') as f:
            hk_lookup_dict = json.load(f)
        logger.log(f"Loaded hk_lookup JSON file from {hk_lookup_path}")
        if reverse:
            # reverse key values in hk_lookup_dict to create a mapping from kmer to gene
            kmer_to_gene = {v: k for k, v in hk_lookup_dict.items()}
            return kmer_to_gene
        else:
            return hk_lookup_dict
    else:
        logger.log(f"hk_lookup JSON file not found at {hk_lookup_path}\tProceeding without it.")
        return None

def main(base_dir=path_to_nn_runs, outdir=outdir_default, x_col=None, hue_col=None, group_x_col=None, group_hue_col=None):
    all_data = []
    top_kmers_df = pd.DataFrame() # Placeholder top_kmers_csv file
    
    if not os.path.exists(base_dir):
        logger.log(f"Directory {base_dir} not found.")
        return
    else:
        logger.log(f"Scanning directory: {base_dir}")
    
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        logger.log(f"Created output directory: {outdir}")
    else:
        logger.log(f"Output directory already exists: {outdir}")
    
    logfile_path = os.path.join(outdir, "collect_iterres_log.txt")
    logfile = open(logfile_path, 'w')
    logger.set_logfile(logfile)
    logger.log(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S] ')} collect_iterres started. Scanning {base_dir} for log files.")

    # Try opening hk_lookup & pfi_lookup
    if args.hk_lookup_path:
        kmer_to_gene = open_hk_lookup(args.hk_lookup_path, reverse=True)
    else:
        logger.log("Note: HK lookup path not provided. Will attempt to deduce from log files.")
        kmer_to_gene = None

    # Iterate through all folders in nn_runs
    for folder_name in tqdm(os.listdir(base_dir), desc="Processing folders"):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            pfi_success = False
            top_int_kmer_success = False
            
            # Search for log files in this specific run folder
            for file in os.listdir(folder_path):
                # Find failed runs and sort them

                # Extract metrics from log files 
                if file.endswith(".txt") and "log_run" in file.lower():
                    log_path = os.path.join(folder_path, file)
                    metrics, run_info = extract_metrics_from_log(log_path)
                    metrics['folder'] = folder_name
                    all_data.append(metrics)
                    #print("run_info", run_info)

                    if args.hk_lookup_path is None:
                        #print("hk_lookup not provided. Attempting to deduce it from log info for encoded data2 run.")
                        # If hk_lookup is not provided, try to deduce it from the log info
                        try:
                            dir = "encoded_sketches" if run_info['use_encoded'] else "sketches_sketches"
                            if run_info['data2']:
                                dir += "_data2"

                            hk_path = os.path.join(data_prod_path, dir, f"hk_lookup_n{metrics['n']}_k{metrics['k']}.json")
                            kmer_to_gene = open_hk_lookup(hk_path, reverse=True)
                            print(f"Deduced hk_lookup path: {hk_path} for folder: {folder_name}")
                        except Exception as e:
                            print(f"Error deducing hk_lookup from log info in {log_path}: {e}")

                # Extract top kmers from pair_kmers.csv files
                elif file.endswith("pair_kmers.csv"):
                    logger.log(f"Found top kmers file: {file} in folder: {folder_name}")
                    top_kmers_path = os.path.join(folder_path, file)
                    try:
                        df_kmers = pd.read_csv(top_kmers_path)
                        df_kmers['folder'] = folder_name
                        df_kmers["UPS"] = calculate_unified_score(metrics)
                        top_int_kmer_success = True
                    except Exception as e:
                        logger.log(f"Error reading {top_kmers_path}: {e}")
                
                # Extract pfi lookup
                elif file.startswith("pfi_") and file.endswith(".txt"):
                    pfi_file_path = os.path.join(folder_path, file)
                    logger.log(f"Found PFI file: {file} in folder: {folder_name}")
                    pfi_success = True
            
            if kmer_to_gene is not None and top_int_kmer_success and pfi_success:
                pfi_lookup = pd.read_csv(pfi_file_path, sep="\t")
                logger.log(f"Processing PFI lookup for {folder_name}...")
                pfi_class = PFI_Lookup(kmer_to_gene, pfi_lookup, TS=True)
                df_kmers = pfi_class.append_pfi_values(df_kmers, kmer_col="decoded_kmer")
            else:
                logger.log(f"Skipping PFI calculation for {folder_name}. Reason: top_kmers={top_int_kmer_success}, pfi_file={pfi_success}, hk_lookup={kmer_to_gene is not None}")
            
            if top_int_kmer_success:
                top_kmers_df = pd.concat([top_kmers_df, df_kmers], ignore_index=True)


    ### Sorting top_kmers_df by weighted PFI score (if weight_pfi flag is set)
    if not top_kmers_df.empty and args.weight_pfi:
        if "UPS" in top_kmers_df.columns:
            top_kmers_df = top_kmers_df.sort_values(by="UPS", ascending=False)
            logger.log("Sorted top_kmers_df by Unified Performance Score (UPS).")
        else:
            logger.log("Warning: 'UPS' column not found in top_kmers_df. Skipping sorting by UPS.")

    ### Metrics Extraction Summary and Plotting ###
    if all_data:
        df = pd.DataFrame(all_data)
        logger.log(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Extracted metrics from {len(df)} log files.")

        #check if any values of the cols in below_one_cols are above 1, if so, apply the correct_deci_number function to the entire column
        try:
            below_one_cols = ['test_accuracy', 'test_balanced_accuracy', 'unseen_test_accuracy', 'unseen_test_balanced_accuracy', 'precision', 'recall', 'f1']
            for col in below_one_cols:
                if col in df.columns:
                    if (df[col] > 1).any():
                        logger.log(f"Column '{col}' contains values greater than 1. Applying correction to entire column.")
                        df[col] = df[col].apply(correct_deci_number)
                    else:
                        continue
                else:
                    logger.log(f"Column '{col}' not found in dataframe. Skipping correction for this column.")
        except Exception as e:
            logger.log(f"Error during decimal correction: {e}")

        try:
            group_x_col = group_x_col.lower()
        except Exception as e:
            logger.log(f"Unable to process group_x_col: {group_x_col}, error: {e}. Defaulting to no grouping.")

        # Marking TP = 0 runs as failed runs for better visualization in the confusion matrix bar plot and heatmap
        if 'TP' in df.columns:
            df['status'] = df.apply(lambda row: False if row['TP'] == 0 or row['TP'] is None else row['status'], axis=1)
            
        
        # Subsetting df to only include successful runs
        if False in df["status"].values:
            logger.log(f"⚠ WARNING: Some runs have failed! Count: {(df['status'] == False).sum()}")
            
            #Get the list of failed runs folder names            
            failed_runs = df[df['status'] == False]['folder'].tolist()
            df_all = df.copy()
            if (df['status'] == True).any():
                df = df[df['status'] == True]
            else:
                raise ValueError("All runs have failed. No data to plot.")
            
            #Subset the top_kmers_df to only include the successful runs as well
            logger.log(top_kmers_df.head())
            top_kmers_df = top_kmers_df[~top_kmers_df['folder'].isin(failed_runs)]
            logger.log(f"Subsetted dataframe to {len(df)} successful runs for plotting. Also subsetted top_kmers_df to {len(top_kmers_df)} entries corresponding to successful runs.")

            # Obtain b_value and p_value from each failed run
            try: 
                failed_runs_info = []
                for folder in failed_runs:
                    b_value = None
                    p_value = None
                    try:
                        b_match = re.search(r"b(\d+)", folder)
                        p_match = re.search(r"p(\d+)", folder)
                        if b_match:
                            b_value = b_match.group(1)
                        if p_match:
                            p_value = p_match.group(1)
                    except Exception as e:
                        logger.log(f"Error extracting b_value and p_value from folder name '{folder}': {e}")
                    failed_runs_info.append((folder, b_value, p_value))
                logger.log("Failed runs and their corresponding b_value and p_value:")
                for folder, b_value, p_value in failed_runs_info:
                    logger.log(f"  {folder}: b={b_value}, p={p_value}")
            except Exception as e:
                logger.log(f"Error processing failed runs for b_value and p_value extraction: {e}")
        
        plotting = MetricPlottingUtils(df=df, outdir=str(outdir), x_col=x_col, hue_col=hue_col, x_col_by_cluster=(group_x_col == 'cluster'), x_col_by_phage=(group_x_col == 'phage'))
        plotting.plot_graphs()
        # Optional: save the raw data for inspection
        df.to_csv(outdir +'all_runs_summary.csv', index=False)
        logger.log("✓ Summary CSV saved as all_runs_summary.csv")
    else:
        logger.log("No valid data found for Metrics Plotting.")
    
    ### Top Kmers Annotation Summary and Plotting ###
    if not top_kmers_df.empty:
        logger.log(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Extracted top k-mers from {len(top_kmers_df['folder'].unique())} files.")
        logger.log(f"Value counts for 'organism' column:\n{top_kmers_df['organism'].value_counts()}")

        # Split by "entity" column 
        bact_kmers_df = top_kmers_df[top_kmers_df['organism'] == 'bacterium']
        phage_kmers_df = top_kmers_df[top_kmers_df['organism'] == 'phage']
        bact_len_before = len(bact_kmers_df)
        phage_len_before = len(phage_kmers_df)
        logger.log(f"Bacterium k-mers sample:\n{bact_kmers_df.head()}")
        logger.log(f"Phage k-mers sample:\n{phage_kmers_df.head()}")

        sort_by = 'UPS' if not args.weight_pfi else 'PFI'

        # Keep only args.top_kmers number of kkmers per entity per folder based on UPS score
        if not bact_kmers_df.empty:
            bact_kmers_df = bact_kmers_df.sort_values(by=sort_by, ascending=False).groupby(['folder', 'entity']).head(args.top_kmers)
            logger.log(f"Top k-mers with {sort_by} scores - bacterium:")
            logger.log(f"{bact_kmers_df[['entity', 'decoded_kmer', sort_by]].head()}")
            bact_len_after = len(bact_kmers_df)

        else:
            logger.log(f"No valid bacterium k-mers data found for {sort_by} sorting.")
            bact_len_after = 0

        if not phage_kmers_df.empty:
            phage_kmers_df = phage_kmers_df.sort_values(by=sort_by, ascending=False).groupby(['folder', 'entity']).head(args.top_kmers)
            logger.log(f"Top k-mers with {sort_by} scores - phage:")
            logger.log(f"{phage_kmers_df[['entity', 'decoded_kmer', sort_by]].head()}")
            phage_len_after = len(phage_kmers_df)
        else:
            logger.log(f"No valid phage k-mers data found for {sort_by} sorting.")
            phage_len_after = 0

        # Obtain pfi scores for kmers and add them to the dataframes if weight_pfi flag is set, then sort by pfi scores instead of UPS scores
        if args.weight_pfi:
            pass
        
        logger.log(f"Reduced bacterium k-mers from {bact_len_before} to {bact_len_after} based on top_kmers and sorting criteria.")
        logger.log(f"Reduced phage k-mers from {phage_len_before} to {phage_len_after} based on top_kmers and sorting criteria.")
        if bact_len_after > 0:
            logger.log(f"Final bacterium sample:\n{bact_kmers_df.head()}")
        if phage_len_after > 0:
            logger.log(f"Final phage sample:\n{phage_kmers_df.head()}")


        #return # for testing purposes, to check the outputs up to this point before proceeding with annotation and plotting
        # Gene analysis
        try:
            GA = GeneAnalysis()
            if not bact_kmers_df.empty:
                bact_annot_df = GA.batch_bact_annotate(bkmers=bact_kmers_df['decoded_kmer'].tolist(), bact_names=bact_kmers_df['entity'].tolist(), data_prod_path=data_prod_path)
            else:
                logger.log("No valid bacterium k-mers data found for annotation.")

            if not phage_kmers_df.empty:
                phage_annot_df = GA.batch_phage_annotate(pkmers=phage_kmers_df['decoded_kmer'].tolist(), phage_names=phage_kmers_df['entity'].tolist(), data_prod_path=data_prod_path)
            else:
                logger.log("No valid phage k-mers data found for annotation.")
        except Exception as e:
            raise ValueError(f"Error during gene annotation: {e}")

        # Gene Annot Plotting
        try:
            title_suffix = "(PFI)" if args.weight_pfi else "(UPS)"
            plotting_utils = GAPlottingUtils(df=top_kmers_df, outdir=str(outdir))
            if not bact_kmers_df.empty:
                plotting_utils.plot_top_genes(bact_annot_df, entity_type="bacterium", title_suffix=title_suffix)
                plotting_utils.plot_kmer_distribution(bact_annot_df, entity_type="bacterium", title_suffix=title_suffix)
                if args.weight_pfi:
                    plotting_utils.plot_kmer_against_ups_or_pfi(bact_annot_df, entity_type="bacterium", sort_by=sort_by)

            if not phage_kmers_df.empty:
                plotting_utils.plot_top_genes(phage_annot_df, entity_type="phage", title_suffix=title_suffix)
                plotting_utils.plot_kmer_distribution(phage_annot_df, entity_type="phage", title_suffix=title_suffix)
                if args.weight_pfi:
                    plotting_utils.plot_kmer_against_ups_or_pfi(phage_annot_df, entity_type="phage", sort_by=sort_by)
                    
        except Exception as e:
            raise ValueError(f"Error during gene annotation plotting: {e}")

        # Concatenate annotation results and save
        try: 
            if not bact_kmers_df.empty and not phage_kmers_df.empty:
                combined_annot_df = pd.concat([bact_annot_df, phage_annot_df], ignore_index=True)
                combined_annot_df.to_csv(outdir + 'top_kmers_annotations.csv', index=False)
                logger.log("Top Kmers Annotations CSV saved as top_kmers_annotations.csv")
            
            elif not bact_kmers_df.empty:
                bact_annot_df.to_csv(outdir + 'top_kmers_annotations.csv', index=False)
                logger.log("Bacterium Kmers Annotations CSV saved as top_kmers_annotations.csv")
            elif not phage_kmers_df.empty:
                phage_annot_df.to_csv(outdir + 'top_kmers_annotations.csv', index=False)
                logger.log("Phage Kmers Annotations CSV saved as top_kmers_annotations.csv")
        except Exception as e:
            raise ValueError(f"Error saving top k-mers annotations: {e}")

    else:
        logger.log("No valid top k-mers data found.")

    logger.log(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Script execution completed.")
    logfile.close()

if __name__ == "__main__":
    if parse_arguments().base_dir:
        args = parse_arguments()
        main(base_dir=args.base_dir.strip(" "), outdir=args.out_dir.strip(" "), x_col=args.x_col, hue_col=args.hue_col, group_x_col=args.group_x_col, group_hue_col=args.group_hue_col)
    else:
        main()