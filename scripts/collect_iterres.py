#!/net/domus/home/people/s215045/miniconda3/bin/python

import os
import re
import ast
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm, Normalize, LinearSegmentedColormap
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator 
import networkx as nx
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
import joblib
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
    parser.add_argument("--filter_harsh", action='store_true',
                        help="Whether to apply harsh filtering criteria on the success of runs (only include runs that have; test accuracy above 0.5 and precision and recall above 0.5)")
    parser.add_argument("--ignore_failed_runs", action='store_true',
                        help="Whether to ignore failed runs in the analysis")

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
    parser.add_argument("--highlight_multi", action='store_true',
                        help="Highlight kmers that map to multiple genes in gene/kmer plots")
    parser.add_argument("--network_top_kmers", type=int, default=50,
                        help="Number of top kmers to include in the kmer-gene network plot")
    parser.add_argument("--hostrange_excel", type=str, default=None,
                        help="Path to the hostrange/EOP Excel file for hostrange heatmap plots")
    parser.add_argument("--hostrange_sheet", type=str, default="Sheet1",
                        help="Sheet name in the hostrange Excel file (default: Sheet1)")
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

def calculate_unified_score(metrics_dict, no_unseen=False):
    """
    Unified Performance Score (UPS) Calculation:
    Calculates a single performance score from a dictionary of NN metrics.
    Weights can be adjusted based on project priorities.
    """
    # 1. Define Weights (Total = 1.0)
    # We prioritize Balanced Accuracy and Unseen Performance
    if no_unseen:
        weights = {
            'test_balanced_accuracy': 0.60,
            'f1': 0.40
        }
    else:
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

def _normalize_for_combine(df, species_col, entity_col, organism_label):
    """Rename species/entity-label columns to a shared schema, add organism tag."""
    out = df.copy()
    out['species'] = out[species_col]
    out['entity_label'] = out[entity_col]   # gene name for bact, product for phage
    out['organism'] = organism_label
    return out

def _unwrap_braced_entity_values(df, columns):
    """Strip brace-wrapped entity values like {'Host 3'} down to Host 3."""
    out = df.copy()
    for column in columns:
        if column in out.columns:
            cleaned = out[column].astype("string").str.extract(
                r"^\{\s*['\"]?(.*?)['\"]?\s*\}$",
                expand=False,
            )
            out[column] = cleaned.where(cleaned.notna(), out[column])
    return out

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
        'TN': None, 'FN': None, 'FP': None, 'TP': None,
        'train_pairs': None, 'test_pairs': None, 'test_train_ratio': None,
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
        metrics['FP'] = int(cm_array_match.group(3))
        metrics['FN'] = int(cm_array_match.group(2))
        metrics['TP'] = int(cm_array_match.group(4))
    else:
        # Try legacy format: [TN FN] \n [FP TP]
        cm_pattern = r"Confusion matrix:.*?\[\s*(\d+)\s+(\d+)\s*\].*?\[\s*(\d+)\s+(\d+)\s*\]"
        cm_match = re.search(cm_pattern, content, re.DOTALL)
        if cm_match:
            metrics['TN'] = int(cm_match.group(1))
            metrics['FN'] = int(cm_match.group(3))
            metrics['FP'] = int(cm_match.group(2))
            metrics['TP'] = int(cm_match.group(4))
    
    # Extract Dataset Pairs: "Built dataset with X pairs and excluded Y pairs"
    # X = train_pairs, Y = test_pairs (excluded is test)
    pairs_match = re.search(r"Built dataset with (\d+) pairs and excluded (\d+) pairs", content)
    if pairs_match:
        metrics['train_pairs'] = int(pairs_match.group(1))
        metrics['test_pairs'] = int(pairs_match.group(2))
        if metrics['train_pairs'] is not None and metrics['test_pairs'] is not None and metrics['train_pairs'] > 0:
            metrics['test_train_ratio'] = round(metrics['test_pairs'] / metrics['train_pairs'], 4)
    
    if "INFO - Process completed in" in content:
        metrics['status'] = True
    else:
        metrics['status'] = False

    return metrics, run_info

class GAPlottingUtils:
    def __init__(self, df, outdir, sort_by="PFI"):
        self.df = df
        self.outdir = outdir
        self.sort_by = sort_by

    def _limit_series_top(self, series: pd.Series, max_items: int = 40) -> pd.Series:
        """
        Return the top `max_items` entries of a value-count Series. If the series
        length is already <= max_items it's returned unchanged.
        """
        try:
            if len(series) <= max_items:
                return series
            # Ensure highest counts first
            return series.sort_values(ascending=False).head(max_items)
        except Exception:
            return series
    
    def plot_top_genes(self, df: pd.DataFrame, entity_type : str, title_suffix: str = ""):
        """
        Plot the annotated genes found in the entity specific dataframe (df)
        """
        # Support highlighting kmers that map to multiple genes by producing stacked bars
        key_col = 'gene' if entity_type == 'bacterium' else 'product'
        # Determine kmer column
        kmer_col = 'decoded_kmer' if 'decoded_kmer' in df.columns else ('kmer_in_seq' if 'kmer_in_seq' in df.columns else None)
        if kmer_col is None:
            logger.log(f"Warning: no kmer column found for plotting top genes for {entity_type}.")
            return

        # Determine per unique kmer how many distinct genes it maps to (for multi-mapping flag).
        # This is done on unique kmer sequences, not on individual rows.
        km_to_num_genes = (
            df.groupby(kmer_col)[key_col]
            .agg(lambda x: len(set(str(v) for v in x if pd.notna(v))))
            .rename('num_genes')
        )

        # Work on a row-level copy so we count actual occurrences per gene, not unique kmers.
        df_work = df[[kmer_col, key_col]].dropna(subset=[key_col]).copy()
        df_work['num_genes'] = df_work[kmer_col].map(km_to_num_genes).fillna(1)
        df_work['is_multi']  = df_work['num_genes'] > 1

        # Count rows per gene (reflects true frequency across all bacteria/runs)
        gene_multi = df_work[df_work['is_multi']].groupby(key_col).size().rename('multi_count')
        gene_total = df_work.groupby(key_col).size().rename('total_count')
        gene_counts_df = pd.concat([gene_total, gene_multi], axis=1).fillna(0)
        gene_counts_df['single_count'] = gene_counts_df['total_count'] - gene_counts_df['multi_count']

        # Limit to top genes by total_count
        gene_counts_df = gene_counts_df.sort_values('total_count', ascending=False).head(40)

        # Plot stacked bars (single vs multi)
        plt.figure(figsize=(12, 7))
        ind = range(len(gene_counts_df))
        plt.bar(ind, gene_counts_df['single_count'], color='tab:blue', label='single-mapped kmers')
        plt.bar(ind, gene_counts_df['multi_count'], bottom=gene_counts_df['single_count'], color='tab:red', label='multi-mapped kmers')
        plt.xticks(ind, gene_counts_df.index, rotation=90, ha='right')
        plt.ylabel('Kmer Count')
        plt.xlabel('Gene')
        plt.title(f'Top {title_suffix} {entity_type.capitalize()} Kmers Annotated Genes (single vs multi-mapped)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.outdir + f'top_genes_{entity_type}.png')
        plt.close()
    
    def plot_kmer_distribution(self, df: pd.DataFrame, entity_type: str, title_suffix: str = ""):
        """
        Plot the distribution of kmers across different genes for the given entity type.
        Bars are colored by the number of distinct genes the kmer maps to.
        """
        kmer_col = 'decoded_kmer' if 'decoded_kmer' in df.columns else (
            'kmer_in_seq' if 'kmer_in_seq' in df.columns else None)
        if kmer_col is None:
            logger.log(f"Warning: no kmer column found for plotting kmer distribution for {entity_type}.")
            return

        map_col = 'gene' if entity_type == 'bacterium' else 'product'

        gene_kmer_counts = df[kmer_col].value_counts()

        # Build kmer → number of distinct genes it maps to
        kmer_to_num_genes = (
            df.groupby(kmer_col)[map_col]
            .agg(lambda x: len({str(v) for v in x if pd.notna(v)}))
            .rename('num_genes')
        )

        # Limit x-axis to avoid label overlap
        gene_kmer_counts = self._limit_series_top(gene_kmer_counts, max_items=40)

        # 1. Get your counts
        num_genes_per_bar = [kmer_to_num_genes.get(k, 1) for k in gene_kmer_counts.index]
        unique_counts = sorted(set(num_genes_per_bar))

        # 2. Create a custom 3-color gradient (Low -> Mid -> High)
        # Using high-contrast hex codes for the polar ends (e.g., Deep Blue to Bright Red)
        custom_colors = ['#0055ff', '#e6e6e6', '#ff0044'] 
        cmap = LinearSegmentedColormap.from_list('tri_gradient', custom_colors)

        # 3. Choose your normalization
        # If zero-mapped kmers are present, LogNorm would exclude them, so we
        # fall back to a linear scale that keeps 0 inside the color range.
        if 0 in unique_counts:
            norm = Normalize(vmin=0, vmax=unique_counts[-1])
        elif unique_counts[-1] > 10:
            # LogNorm requires vmin > 0, so we use the smallest positive count.
            norm = LogNorm(vmin=unique_counts[0], vmax=unique_counts[-1])
        else:
            norm = Normalize(vmin=unique_counts[0], vmax=unique_counts[-1])

        # 4. Map colors
        color_lookup = {n: cmap(norm(n)) for n in unique_counts}
        colors = [color_lookup[n] for n in num_genes_per_bar]

        # 5. Handle the Legend cleanly
        # Quick tip: If you have dozens of unique counts, a traditional legend item 
        # for every single number will overflow your plot. Let's sample 3-5 distinct points instead.
        if len(unique_counts) > 5:
            # Sample min, midpoints, and max for a clean display
            legend_counts = [
                unique_counts[0], 
                unique_counts[len(unique_counts)//4],
                unique_counts[len(unique_counts)//2], 
                unique_counts[-1]
            ]
            # Remove duplicates if any overlap
            legend_counts = sorted(list(set(legend_counts)))
        else:
            legend_counts = unique_counts

        legend_patches = [
            Patch(color=color_lookup[n], label=f'{n} gene{"s" if n != 1 else ""}')
            for n in legend_counts
        ]

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(gene_kmer_counts)), gene_kmer_counts.values, color=colors)
        plt.xticks(range(len(gene_kmer_counts)), gene_kmer_counts.index, rotation=90, ha='right')
        plt.title(f'Distribution of Top {title_suffix} {entity_type.capitalize()} Kmers ')
        plt.xlabel('Kmers')
        plt.ylabel('Kmer Count')
        plt.legend(handles=legend_patches, title='Genes mapped to')
        plt.tight_layout()
        plt.savefig(self.outdir + f'kmer_distribution_{entity_type}.png')
        plt.close()

    def plot_kmer_gene_network(self, df: pd.DataFrame, entity_type: str, top_kmers: int = 200):
        """
        Plot a bipartite network showing kmers (left) connected to gene/product nodes (right).
        Uses nx.bipartite_layout to enforce the two-column structure.
        Limits to top_kmers by frequency to keep the plot readable.
        """
        kmer_col = 'decoded_kmer' if 'decoded_kmer' in df.columns else (
            'kmer_in_seq' if 'kmer_in_seq' in df.columns else None
        )
        key_col = 'gene' if entity_type == 'bacterium' else 'product'
        if kmer_col is None or key_col not in df.columns:
            logger.log(f"Cannot create network: missing columns for {entity_type}.")
            return

        # 1. Prepare Data and Subsample -------------------------------------------------
        # Select top kmers by occurrence
        top_k = df[kmer_col].value_counts().head(top_kmers).index.tolist()
        top_k_set = set(top_k)
        sub = df[df[kmer_col].isin(top_k)][[kmer_col, key_col]].dropna().copy()

        if len(sub) == 0:
            logger.log(f"No kmer-{key_col} pairs remain after filtering. Plotting aborted.")
            return

        # Node sizes should reflect how often each item appears in the plotted data.
        kmer_counts = sub[kmer_col].value_counts()
        gene_counts = sub[key_col].value_counts()
        all_counts = pd.concat([kmer_counts, gene_counts], axis=0)
        all_counts = pd.to_numeric(all_counts, errors='coerce').dropna()
        if all_counts.empty:
            logger.log("No finite node counts available for sizing. Plotting aborted.")
            return
        count_min = float(all_counts.min())
        count_max = float(all_counts.max())

        def scale_node_sizes(count_values, min_size=800, max_size=3000):
            if len(count_values) == 0:
                return np.array([])
            count_values = np.asarray(count_values, dtype=float)
            count_values = np.nan_to_num(count_values, nan=count_min)
            if count_max > count_min:
                normalized = (count_values - count_min) / (count_max - count_min)
                sizes = min_size + normalized * (max_size - min_size)
            else:
                sizes = np.full(len(count_values), (min_size + max_size) / 2.0)
            return np.clip(np.nan_to_num(sizes, nan=min_size), min_size, max_size)

        # Unique node lists — order kmers by their self.sort_by values so the layout
        # --- Pick the scoring column, if any ---
        if self.sort_by in df.columns:
            # Aggregate one score per kmer. .mean() is a sensible default;
            # swap for .max() or .median() if that matches the metric's semantics better.
            kmer_scores = (
                df[df[kmer_col].isin(top_k_set)]
                .groupby(kmer_col)[self.sort_by]
                .mean()
            )
            # Order: highest score first; tie-break by first appearance in df for stability.
            appearance_order = {k: i for i, k in enumerate(pd.unique(df[kmer_col]))}
            ordered_kmers = sorted(
                (k for k in kmer_scores.index if k in top_k_set),
                key=lambda k: (-kmer_scores.loc[k], appearance_order.get(k, 0)),
            )
            # Any top-k kmers with no score (e.g. all-NaN UPS/PFI rows) go to the end,
            # preserving their original appearance order.
            scored = set(ordered_kmers)
            ordered_kmers += [k for k in pd.unique(df[kmer_col])
                            if k in top_k_set and k not in scored]
            logger.log(f"Ordering kmers by {self.sort_by} (descending, aggregated by mean).")
        else:
            logger.log(f"No valid sort column found for {self.sort_by}. Using original kmer order.")
            ordered_kmers = [k for k in pd.unique(df[kmer_col]) if k in top_k_set]

        kmers = [f'k:{k}' for k in ordered_kmers]
        ordered_kmers = [k for k in kmer_scores.index if k in top_k_set]
        kmers = [f'k:{k}' for k in ordered_kmers]
        genes = [f'g:{g}' for g in sub[key_col].unique()]

        # 2. Build the Graph ------------------------------------------------------------
        G = nx.Graph()
        G.add_nodes_from(kmers, bipartite=0, label='Kmer')
        G.add_nodes_from(genes, bipartite=1, label='Gene' if entity_type == 'bacterium' else 'Product')

        # Track edge weight (co-occurrence count)
        edge_counts = sub.groupby([kmer_col, key_col]).size().reset_index(name='weight')
        for _, r in edge_counts.iterrows():
            G.add_edge(f'k:{r[kmer_col]}', f'g:{r[key_col]}', weight=r['weight'])

        # Store readable labels on the nodes
        for k in ordered_kmers:
            G.nodes[f'k:{k}']['display'] = k
        for g in sub[key_col].unique():
            G.nodes[f'g:{g}']['display'] = g
        
        isolates = list(nx.isolates(G))
        if isolates:
            G.remove_nodes_from(isolates)
            logger.log(f"Removed {len(isolates)} isolated nodes from the network.")

        # Rebuild partition lists from what's left in G, preserving the order we set.
        kmers = [n for n in kmers if n in G]
        genes = [n for n in genes if n in G]

        if not kmers or not genes:
            logger.log("Graph has no edges after pruning isolates. Plotting aborted.")
            return

        # 3. Node Sizing (scaled by dataset occurrence count) --------------------------
        k_sizes = scale_node_sizes([kmer_counts.get(k.replace('k:', ''), 0) for k in kmers])
        g_sizes = scale_node_sizes([gene_counts.get(g.replace('g:', ''), 0) for g in genes])

        # 4. Edge Widths and Colors ----------------------------------------------------
        edges = list(G.edges(data='weight', default=1))
        edge_list = [(u, v) for u, v, _ in edges]
        edge_weights = np.array([w for _, _, w in edges])
        w_min, w_max = edge_weights.min(), edge_weights.max()
        if w_max > w_min:
            edge_widths = 1.2 + (edge_weights - w_min) / (w_max - w_min) * 2.3
        else:
            edge_widths = np.full(len(edge_weights), 1.5)

        # 5. Bipartite Layout (kmers on left, genes on right) --------------------------
        # bipartite_layout places `nodes` on the left and the rest on the right by default.
        pos = nx.bipartite_layout(G, kmers, align='vertical', scale=2.0, aspect_ratio=0.6)

        # 6. Draw the Graph ------------------------------------------------------------
        # Scale the canvas with the filtered subgraph size so dense networks get more room.
        size_scale = np.clip(np.sqrt(len(sub) / 50.0), 1.0, 1.2)
        fig, ax = plt.subplots(figsize=(12 * size_scale, 16 * size_scale))  # taller figure suits a vertical bipartite layout
        ax.set_title(
            f'Kmer → {"Gene" if entity_type == "bacterium" else "Product"} '
            f'Bipartite Network ({entity_type}) — top {len(kmers)} kmers',
            fontsize=14, fontweight='bold', pad=15
        )

        if w_max > w_min:
            c_vmin, c_vmax = w_min, w_max
        else:
            # All edges share the same weight — pick a tight integer band around it
            # so the single value sits at the centre of the bar.
            c_vmin, c_vmax = w_min - 1, w_min + 1

        nx.draw_networkx_edges(
            G, pos,
            edgelist=edge_list,
            width=edge_widths,
            edge_color=edge_weights,
            edge_cmap=plt.cm.viridis_r,
            edge_vmin=c_vmin,       
            edge_vmax=c_vmax,        
            alpha=1,
            ax=ax
        )

        # Kmer nodes (left column)
        nx.draw_networkx_nodes(
            G, pos, nodelist=kmers,
            node_color='#2ecc71',
            node_size=k_sizes,
            alpha=0.88,
            linewidths=0.8,
            edgecolors='#1a8a4a',
            ax=ax
        )
        # Gene/product nodes (right column)
        nx.draw_networkx_nodes(
            G, pos, nodelist=genes,
            node_color='#3498db',
            node_size=g_sizes,
            alpha=0.90,
            linewidths=0.8,
            edgecolors='#1a5276',
            ax=ax
        )

        # 7. Labels --------------------------------------------------------------------
        all_labels = {n: G.nodes[n].get('display', n) for n in G.nodes()}
        nx.draw_networkx_labels(
            G, pos, labels=all_labels,
            font_size=14,
            font_color='#111111',
            bbox=dict(boxstyle='round,pad=0.22', fc='white', alpha=0.72, lw=0),
            ax=ax
        )

        # 8. Legend --------------------------------------------------------------------
        gene_label = 'genes' if entity_type == 'bacterium' else 'products'
        legend_handles = [
            mpatches.Patch(facecolor='#2ecc71', edgecolor='#1a8a4a',
                        linewidth=1.2, label='kmers'),
            mpatches.Patch(facecolor='#3498db', edgecolor='#1a5276',
                        linewidth=1.2, label=f'{gene_label}'),
        ]
        color_legend = ax.legend(
            handles=legend_handles,
            loc='upper left',
            fontsize=12,
            frameon=True,
            framealpha=0.9,
            edgecolor='#cccccc',
            handlelength=1.5,
            handleheight=1.5,
        )
        ax.add_artist(color_legend)

        # Size legend: show how marker area maps to raw occurrence counts.
        size_example_counts = np.unique(np.linspace(count_min, count_max, num=3, dtype=int))
        size_handles = []
        for count in size_example_counts:
            size_value = scale_node_sizes(np.array([count]))[0]
            size_handles.append(
                Line2D(
                    [0], [0],
                    marker='o',
                    linestyle='None',
                    markerfacecolor='#666666',
                    markeredgecolor='#333333',
                    alpha=0.65,
                    markersize=np.sqrt(size_value) / 1.7,
                    label=f'{count} occurrence' if count == 1 else f'{count} occurrences',
                )
            )
        ax.legend(
            handles=size_handles,
            title='Node size scale\n(occurrences)',
            loc='lower left',
            fontsize=12,
            title_fontsize=12,
            frameon=True,
            framealpha=0.9,
            edgecolor='#cccccc',
        )

        # 9. Colorbar for edge weights -------------------------------------------------
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.viridis_r,
            norm=plt.Normalize(vmin=c_vmin, vmax=c_vmax),
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.45, pad=0.01)
        cbar.locator = MaxNLocator(integer=True)
        cbar.update_ticks()
        cbar.set_label('Edge weight (occurrences)', fontsize=10)

        ax.axis('off')
        plt.tight_layout()
        plt.savefig(self.outdir + f'kmer_gene_network_{entity_type}.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_kmer_against_ups_or_pfi(self, df: pd.DataFrame, entity_type: str):
        """
        Plot the relationship between k-mer counts and the Unified Performance Score (UPS), Pairwise Feature Interaction Score (PFI) 
        or Weighted PFI Score (WPFI) for the given entity type
        """
        if self.sort_by not in df.columns:
            logger.log("Column not found in dataframe. Cannot plot k-mer against UPS or PFI.")
            return
        
        plt.figure(figsize=(11, 6.5))
        title_part = "Unified Performance Score (UPS)" if self.sort_by == 'UPS' else "PFI Score" if self.sort_by == 'PFI' else "WPFI Score"
        hue_col = 'gene' if entity_type == 'bacterium' else 'product'
        ax = sns.scatterplot(
            x='kmer_in_seq',
            y=self.sort_by,
            data=df,
            hue=hue_col,
            palette='coolwarm',
            legend=False,
        )
        plt.title(f'Kmer Count vs {title_part} for {entity_type.capitalize()} Kmers')
        plt.xlabel('Kmer Count')
        plt.ylabel(title_part)

        # Build a compact legend instead of letting seaborn list every category.
        hue_counts = df[hue_col].astype(str).value_counts(dropna=True)
        max_legend_items = 12
        if len(hue_counts) > max_legend_items:
            legend_labels = list(hue_counts.head(max_legend_items - 1).index)
            omitted_count = len(hue_counts) - len(legend_labels)
        else:
            legend_labels = list(hue_counts.index)
            omitted_count = 0

        palette = sns.color_palette('coolwarm', n_colors=max(1, len(hue_counts)))
        color_map = {label: palette[i % len(palette)] for i, label in enumerate(hue_counts.index)}
        legend_handles = [
            Line2D(
                [0], [0],
                marker='o',
                linestyle='None',
                markersize=6,
                markerfacecolor=color_map[label],
                markeredgecolor='black',
                label=label,
            )
            for label in legend_labels
        ]
        if omitted_count > 0:
            legend_handles.append(
                Line2D(
                    [0], [0],
                    marker='o',
                    linestyle='None',
                    markersize=6,
                    markerfacecolor='#999999',
                    markeredgecolor='black',
                    label=f'Other ({omitted_count} more)',
                )
            )

        plt.legend(
            handles=legend_handles,
            title='Gene' if entity_type == 'bacterium' else 'phage',
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            fontsize=8,
            title_fontsize=9,
            frameon=True,
            borderaxespad=0.0,
        )

        # Reduce number of x-axis tick labels to at most 50 to avoid overlap
        try:
            unique_x = np.unique(df['kmer_in_seq'].values)
            if len(unique_x) > 40:
                idx = np.linspace(0, len(unique_x) - 1, num=40, dtype=int)
                tick_vals = unique_x[idx]
            else:
                tick_vals = unique_x
            ax.set_xticks(tick_vals)
            ax.set_xticklabels([str(v) for v in tick_vals], rotation=90, ha='right')
        except Exception:
            pass

        plt.tight_layout()
        plt.savefig(self.outdir + f'kmer_vs_{self.sort_by.lower()}_{entity_type}.png')
        plt.close()

    def plot_species_distribution_grid(self, collected_df: pd.DataFrame,
                                    value_col: str,
                                    kmer_col: str = 'decoded_kmer',
                                    top_n: int = 30,
                                    figsize=(16, 10),
                                    filename_suffix: str = ''):
        """
        Heatmap of `value_col` (e.g. 'decoded_kmer' or 'entity_label') against `species`.

        Cell value = number of rows in `collected_df` where that species and that
        k-mer/gene co-occur. `top_n` caps the column count to the most frequent
        values overall, so the grid stays readable.
        """
        if collected_df.empty or value_col not in collected_df.columns \
                or 'species' not in collected_df.columns:
            logger.log(f"plot_species_distribution_grid: missing data for {value_col}.")
            return

        df = collected_df[['species', 'organism', value_col]].dropna()
        if df.empty:
            return

        # Keep only the top_n most frequent values overall, so the grid is readable
        top_values = df[value_col].value_counts().head(top_n).index
        df = df[df[value_col].isin(top_values)]

        pivot = (df.groupby(['species', value_col]).size()
                .unstack(fill_value=0)
                .reindex(columns=top_values))     # preserve frequency order

        # Sort species rows by total count desc; group by organism if both present
        if 'organism' in collected_df.columns:
            species_organism = (collected_df[['species', 'organism']]
                                .dropna().drop_duplicates()
                                .set_index('species')['organism'])
            pivot = pivot.assign(_organism=species_organism.reindex(pivot.index),
                                _total=pivot.sum(axis=1))
            pivot = pivot.sort_values(['_organism', '_total'],
                                    ascending=[True, False])
            pivot = pivot.drop(columns=['_organism', '_total'])

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            pivot, ax=ax,
            cmap='viridis_r', cbar_kws={'label': 'Co-occurrence count'},
            linewidths=0.3, linecolor='#ffffff',
            annot=pivot.values if pivot.size <= 400 else False,  # annotate only if small
            fmt='d',
        )
        ax.set_title(f'Distribution of top {len(top_values)} {value_col} across species',
                    fontsize=13, fontweight='bold')
        ax.set_xlabel(value_col)
        ax.set_ylabel('Species')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.outdir + f'species_distribution_{value_col}{filename_suffix}.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

    def plot_kmer_occurrence_per_partition(self):
        """
        Bar plot of kmer occurrence per partition (folder), one panel per organism.
        Uses self.df (top_kmers_df) directly.

        X-axis : partitions ordered by (b_value, p_value) ascending.
        Y-axis : number of kmer rows (pair entries) for that partition.
        """
        df = self.df

        # ── Sort partitions by (b, p) numerically ─────────────────────────────
        def bp_sort_key(folder):
            b = re.search(r'b(\d+)', str(folder))
            p = re.search(r'p(\d+)', str(folder))
            return (int(b.group(1)) if b else 0, int(p.group(1)) if p else 0)

        # Readable tick label: strip leading "cluster_" and trailing "_run<n>"
        def partition_label(folder):
            return re.sub(r'^cluster_', '', re.sub(r'_run\d+$', '', str(folder)))

        sorted_folders = sorted(df['folder'].unique(), key=bp_sort_key)
        labels = [partition_label(f) for f in sorted_folders]

        # ── One panel per organism ─────────────────────────────────────────────
        organisms = [
            ('bacterium', 'bact_decoded_kmer',  '#4C72B0'),
            ('phage',     'phage_decoded_kmer', '#DD8452'),
        ]
        # Keep only organisms whose kmer column exists in df
        organisms = [(org, col, color) for org, col, color in organisms if col in df.columns]

        if not organisms:
            logger.log("plot_kmer_occurrence_per_partition: neither bact_decoded_kmer "
                    "nor phage_decoded_kmer found in df. Skipping.")
            return

        fig, axes = plt.subplots(
            len(organisms), 1,
            figsize=(max(12, len(sorted_folders) * 0.6), 4 * len(organisms)),
            sharex=True,
        )
        if len(organisms) == 1:
            axes = [axes]

        for ax, (organism, kmer_col, color) in zip(axes, organisms):
            counts = (
                df.groupby('folder')[kmer_col]
                .count()                          # non-null rows = kmer occurrences
                .reindex(sorted_folders, fill_value=0)
            )

            ax.bar(range(len(sorted_folders)), counts.values,
                color=color, edgecolor='white', linewidth=0.5)
            ax.set_title(f'{organism.capitalize()} — Kmer Occurrences per Partition',
                        fontsize=12, fontweight='bold')
            ax.set_ylabel('Kmer Count', fontsize=10)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            sns.despine(ax=ax)

        axes[-1].set_xticks(range(len(sorted_folders)))
        axes[-1].set_xticklabels(labels, rotation=90, ha='right', fontsize=8)
        axes[-1].set_xlabel('Partition  (b = bacterial cluster,  p = phage cluster)',
                            fontsize=10)

        plt.tight_layout()
        plt.savefig(self.outdir + 'kmer_occurrence_per_partition.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        logger.log("Saved: kmer_occurrence_per_partition.png")

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
            df['b_value'] = pd.to_numeric(df['folder'].str.extract(r"b(\d+)")[0], errors='coerce')
            df['p_value'] = pd.to_numeric(df['folder'].str.extract(r"p(\d+)")[0], errors='coerce')
        except Exception as e:
            print(f"Error extracting b_value and p_value from folder names: {e}")
            df['b_value'] = None
            df['p_value'] = None

        self.outdir = outdir
        if not self.outdir.endswith('/'):
            self.outdir += '/'
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
        self.x_label = x_col if x_col is not None else 'Run / Configuration'

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
            df["partition_group"] = "partition_b" + df["b_value"].astype(str) + "_p" + df["p_value"].astype(str)
            if x_col_by_phage:
                df["phage_group"] = "phage=" + parts[1]
                self.x_col = "phage_group" if x_col_by_phage else self.x_col
                self.x_label = "Phage"
                # sort by phage_name for better visualization
                df = df.sort_values("phage_name")
            elif x_col_by_cluster:
                self.x_col = "partition_group" if x_col_by_cluster else self.x_col
                self.x_label = "Partition"
                # sort by partition coordinates for better visualization
                df = df.sort_values(["b_value", "p_value"], kind="mergesort")
        
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
        x_order = cm_melted[self.x_col].drop_duplicates().tolist()
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
            order=x_order,
        )

        if show_percentage:
            ax.set_ylim(0, 100)

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
        plt.xlabel(self.x_label, fontsize=6)
        
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
        x_order = self.df[x_col].drop_duplicates().tolist()
        
        if hue_col:
            ax = sns.barplot(
                data=self.df, 
                x=x_col, 
                y=y_col, 
                hue=hue_col, 
                palette='viridis',
                edgecolor='black',
                order=x_order,
            )
        else:
            ax = sns.barplot(
                data=self.df, 
                x=x_col, 
                y=y_col, 
                edgecolor='black',
                order=x_order,
            )
            plt.xticks(rotation=90, ha='right')
        
        ax.set_ylim(0, 1)

        plt.title(title, fontsize=15, pad=15)
        plt.ylabel(ylabel, fontsize=12)
        plt.xlabel(self.x_label, fontsize=6)
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

        plt.title("Success Rate by Phage and Bacterial Partitions")
        plt.xlabel("Phage partition")
        plt.ylabel("Bacterial partition")
        plt.tight_layout()
        plt.savefig(self.outdir + 'bp_partition_heatmap.png', dpi=300)
        plt.close() # Close to free up memory
        print("Saved: bp_partition_heatmap.png")

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
        if 'unseen_test_accuracy' in self.df.columns:
            self._plot_bars(
                x_col=self.x_col, 
                y_col='unseen_test_accuracy', 
                hue_col=self.hue_col, 
                title=f'FFNN Truly Unseen Test Accuracy {self.title_suffix}', 
                ylabel='Truly Unseen Test Accuracy',
                outpath=self.outdir + 'unseen_accuracy_by_nk.png'
            )

        # --- Graph 4: Grouped Truly Unseen Balanced Accuracy Bar Chart (X=n, Hue=k) ---
        if 'unseen_test_balanced_accuracy' in self.df.columns:
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

def balanced_top_k(df: pd.DataFrame, group_cols, sort_col: str, total_k: int) -> pd.DataFrame:
    if df.empty or total_k <= 0:
        return df.head(0)

    if sort_col not in df.columns:
        raise ValueError(f"Column '{sort_col}' not found in dataframe.")

    sorted_df = df.sort_values(by=sort_col, ascending=False).copy()

    if total_k >= len(sorted_df):
        return sorted_df

    n_groups = sorted_df.groupby(group_cols, sort=False).ngroups
    if n_groups == 0:
        return sorted_df.head(0)

    base_quota = total_k // n_groups
    remainder = total_k % n_groups

    sorted_df['_group_rank'] = sorted_df.groupby(group_cols, sort=False).cumcount() + 1

    if base_quota > 0:
        selected = sorted_df[sorted_df['_group_rank'] <= base_quota].copy()
    else:
        selected = sorted_df.head(0).copy()

    if remainder > 0:
        extra_candidates = sorted_df[sorted_df['_group_rank'] == (base_quota + 1)]
        extra = extra_candidates.nlargest(remainder, sort_col)
        selected = pd.concat([selected, extra], ignore_index=False)

    if len(selected) < total_k:
        selected_idx = set(selected.index.tolist())
        fill = sorted_df.loc[~sorted_df.index.isin(selected_idx)].nlargest(total_k - len(selected), sort_col)
        selected = pd.concat([selected, fill], ignore_index=False)

    selected = selected.sort_values(by=sort_col, ascending=False).head(total_k)
    return selected.drop(columns=['_group_rank'], errors='ignore')


def main(base_dir=path_to_nn_runs, outdir=outdir_default, x_col=None, hue_col=None, group_x_col=None, group_hue_col=None, ignore_failed_runs=False):
    all_data = []
    data2 = True if "data2" in base_dir else False
    top_kmers_df = pd.DataFrame() # Placeholder top_kmers_csv file

    if not os.path.exists(base_dir):
        logger.log(f"Directory {base_dir} not found.")
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

    # Iterate through all folders in nn_runs
    c = 0
    for folder_name in tqdm(os.listdir(base_dir), desc="Processing folders"):
        logger.log(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing folder: {folder_name}")
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
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

                    if True:
                        #print("hk_lookup not provided. Attempting to deduce it from log info for encoded data2 run.")
                        # If hk_lookup is not provided, try to deduce it from the log info
                        try:
                            dir = "encoded_sketches" if run_info['use_encoded'] else "SM_sketches"
                            if run_info['data2']:
                                dir += "_data2"

                            hk_path = os.path.join(data_prod_path, dir, f"hk_lookup_n{metrics['n']}_k{metrics['k']}.json")
                            kmer_to_gene = open_hk_lookup(hk_path, reverse=True)
                            if kmer_to_gene is not None:
                                logger.log(f"Deduced hk_lookup from log info for {folder_name} using path: {hk_path}")
                        except Exception as e:
                            print(f"Error deducing hk_lookup from log info in {log_path}: {e}")

                # Extract top kmers from pair_kmers.csv files
                elif file.endswith("normalized_interaction.csv"):
                    logger.log(f"Found normalized interactions file: {file} in folder: {folder_name}")
                    norm_int_rate_path = os.path.join(folder_path, file)
                    try:
                        df_kmers = pd.read_csv(norm_int_rate_path)
                        df_kmers = _unwrap_braced_entity_values(df_kmers, ["bact_entity", "phage_entity"])
                        df_kmers['folder'] = folder_name
                        top_int_kmer_success = True
                    except Exception as e:
                        logger.log(f"Error reading {norm_int_rate_path}: {e}")
            
            if top_int_kmer_success:
                df_kmers["UPS"] = calculate_unified_score(metrics)
                df_kmers["test_accuracy"] = metrics.get("test_accuracy", None)
                df_kmers["folder"] = metrics.get("file_path", None)
                top_kmers_df = pd.concat([top_kmers_df, df_kmers], ignore_index=True)
            
            else:
                logger.log(f"Skipping PFI calculation for {folder_name}. Reason: top_kmers={top_int_kmer_success}, hk_lookup={kmer_to_gene is not None}")

        logger.log(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Folder processed: {folder_name}")
        logger.log("#" * 50)
        # c += 1
        # if c > 4:
        #     break

    ### Sorting top_kmers_df by weighted PFI score (if weight_pfi flag is set)
    sort_by = "UPS"
    title_suffix = "(UPS)"
    if "norm_int_rate_score" in top_kmers_df.columns:
        top_kmers_df = top_kmers_df.rename(columns={"norm_int_rate_score": "PFI"})

    if not top_kmers_df.empty:
        top_kmers_go = True
        if "PFI" in top_kmers_df.columns:
            if args.weight_pfi:
                #scale PFI by run test accuracy then sort
                if "test_accuracy" in top_kmers_df.columns:
                    top_kmers_df["WPFI"] = top_kmers_df["PFI"] * top_kmers_df["test_accuracy"]
                    top_kmers_df = top_kmers_df.sort_values(by="WPFI", ascending=False)
                    logger.log("Sorted top_kmers_df by weighted PFI score (expected interaction score scaled by test accuracy).")
                    sort_by = "WPFI"
                    title_suffix = "(WPFI)"

                else:
                    logger.log("Warning: 'test_accuracy' column not found in top_kmers_df. Cannot weight PFI score by test accuracy. Sorting by PFI instead.")
                    top_kmers_df = top_kmers_df.sort_values(by="PFI", ascending=False)
                    logger.log("Sorted top_kmers_df by PFI score (expected interaction score) without weighting.")
                    sort_by = "PFI"
                    title_suffix = "(PFI)"
            else:
                #sort by PFI without weighting
                top_kmers_df = top_kmers_df.sort_values(by="PFI", ascending=False)
                logger.log("Sorted top_kmers_df by weighted PFI score (expected interaction score).")
                sort_by = "PFI"
                title_suffix = "(PFI)"

        elif "UPS" in top_kmers_df.columns:
            top_kmers_df = top_kmers_df.sort_values(by="UPS", ascending=False)
            logger.log("Sorted top_kmers_df by Unified Performance Score (UPS).")
            sort_by = "UPS"
            title_suffix = "(UPS)"

        else:
            logger.log("Warning: 'UPS' column not found in top_kmers_df. Skipping sorting by UPS.")
    else:
        logger.log("top_kmers_df is empty. No k-mer data to process or plot.")
        top_kmers_go = False


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
        if 'TP' in df.columns and not ignore_failed_runs:
            df['status'] = df.apply(lambda row: False if row['TP'] == 0 or row['TP'] is None else row['status'], axis=1)
        
        if args.filter_harsh:
            prec_recall_threshold = 0.5
            if len(df["precision"].dropna()) > 0:
                if (df["precision"] > 0.5).any() and (df["recall"] > prec_recall_threshold).any():
                    df['status'] = df.apply(lambda row: False if row['precision'] <= prec_recall_threshold or row['precision'] is None else row['status'], axis=1)
                    df['status'] = df.apply(lambda row: False if row['recall'] <= prec_recall_threshold or row['recall'] is None else row['status'], axis=1)
            if len(df["test_accuracy"].dropna()) > 0:
                if (df["test_accuracy"] > 0.5).any():
                    df['status'] = df.apply(lambda row: False if row['test_accuracy'] <= 0.5 or row['test_accuracy'] is None else row['status'], axis=1)
            
        
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
            if top_kmers_go:
                logger.log(top_kmers_df.head().to_string())
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
        # Save the raw data for inspection
        df.to_csv(outdir +'all_runs_summary.csv', index=False)
        if top_kmers_go:
            top_kmers_df.to_csv(outdir + 'top_kmers_summary.csv', index=False)
        logger.log("✓ Summary CSVs saved as all_runs_summary.csv and top_kmers_summary.csv")
    else:
        logger.log("No valid data found for Metrics Plotting.")


    ### Top Kmers Annotation Summary and Plotting ###
    if top_kmers_go:
        logger.log(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Extracted top k-mers from {len(top_kmers_df['folder'].unique())} files.")
        #logger.log(f"Value counts for 'organism' column:\n{top_kmers_df['organism'].value_counts()}")

        # Split by "entity" column 
        bact_kmers_df = top_kmers_df[["bact_entity", "bact_organism", "bact_decoded_kmer", "PFI", "UPS", "test_accuracy", "folder"]].copy()
        bact_kmers_df = bact_kmers_df.rename(columns={
            "bact_entity": "entity",
            "bact_organism": "organism",
            "bact_decoded_kmer": "decoded_kmer"
        })
        bact_kmers_df.insert(0, 'hash', top_kmers_df['pair'].str.extract(r'np\.int64\((\d+)\)').astype('int64'))
        if "WPFI" in top_kmers_df.columns:
            bact_kmers_df["WPFI"] = top_kmers_df["WPFI"]

        phage_kmers_df = top_kmers_df[["phage_entity", "phage_organism", "phage_decoded_kmer", "PFI", "UPS", "test_accuracy", "folder"]].copy()
        phage_kmers_df = phage_kmers_df.rename(columns={
            "phage_entity": "entity",
            "phage_organism": "organism",
            "phage_decoded_kmer": "decoded_kmer"
        })
        phage_kmers_df.insert(0, 'hash', top_kmers_df['pair'].str.extract(r'np\.int64\(\d+\).*?np\.int64\((\d+)\)').astype('int64'))
        if "WPFI" in top_kmers_df.columns:
            phage_kmers_df["WPFI"] = top_kmers_df["WPFI"]

        bact_len_before = len(bact_kmers_df)
        phage_len_before = len(phage_kmers_df)
        logger.log(f"Bacterium k-mers sample:\n{bact_kmers_df.head()}")
        logger.log(f"Phage k-mers sample:\n{phage_kmers_df.head()}")

        # Keep only top_kmers number of kkmers per entity per folder based on sort_by score
        if not bact_kmers_df.empty:
            bact_kmers_df = balanced_top_k(
                df=bact_kmers_df,
                group_cols=['folder', 'entity'],
                sort_col=sort_by,
                total_k=args.top_kmers
            )
            logger.log(f"Top k-mers with {sort_by} scores - bacterium:")
            logger.log(f"{bact_kmers_df[['entity', 'decoded_kmer', sort_by]].head()}")
            bact_len_after = len(bact_kmers_df)

        else:
            logger.log(f"No valid bacterium k-mers data found for {sort_by} sorting.")
            bact_len_after = 0

        if not phage_kmers_df.empty:
            phage_kmers_df = balanced_top_k(
                df=phage_kmers_df,
                group_cols=['folder', 'entity'],
                sort_col=sort_by,
                total_k=args.top_kmers
            )
            logger.log(f"Top k-mers with {sort_by} scores - phage:")
            logger.log(f"{phage_kmers_df[['entity', 'decoded_kmer', sort_by]].head()}")
            phage_len_after = len(phage_kmers_df)
        else:
            logger.log(f"No valid phage k-mers data found for {sort_by} sorting.")
            phage_len_after = 0
        
        logger.log(f"Reduced bacterium k-mers from {bact_len_before} to {bact_len_after} based on top_kmers and sorting criteria.")
        logger.log(f"Reduced phage k-mers from {phage_len_before} to {phage_len_after} based on top_kmers and sorting criteria.")
        if bact_len_after > 0:
            logger.log(f"Final bacterium sample:\n{bact_kmers_df.head()}")
        if phage_len_after > 0:
            logger.log(f"Final phage sample:\n{phage_kmers_df.head()}")

        # Gene analysis
        try:
            GA = GeneAnalysis()
            if not bact_kmers_df.empty:
                #bact_kmers_df = bact_kmers_df.reset_index()
                bact_annot_df = GA.batch_bact_annotate(bact_df=bact_kmers_df, kmer_col='decoded_kmer', entity_col='entity', data_prod_path=data_prod_path)
                #bact_annot_df = GA.batch_bact_annotate(bkmers=bact_kmers_df['decoded_kmer'].tolist(), bact_names=bact_kmers_df['entity'].tolist(), data_prod_path=data_prod_path)
            else:
                logger.log("No valid bacterium k-mers data found for annotation.")

            if not phage_kmers_df.empty:
                phage_annot_df = GA.batch_phage_annotate(phage_df=phage_kmers_df, kmer_col='decoded_kmer', entity_col='entity', data_prod_path=data_prod_path)
                #phage_annot_df = GA.batch_phage_annotate(pkmers=phage_kmers_df['decoded_kmer'].tolist(), phage_names=phage_kmers_df['entity'].tolist(), data_prod_path=data_prod_path)
            else:
                logger.log("No valid phage k-mers data found for annotation.")
        except Exception as e:
            raise ValueError(f"Error during gene annotation: {e}")

        # Log annotation results for inspection
        if not bact_kmers_df.empty and 'bact_annot_df' in locals():
            logger.log(f"Bacterium annotation sample:\n{bact_annot_df.head()}")
            logger.log(f"Succes/Total rate of kmer in bact df: {len(bact_annot_df)}/{len(bact_kmers_df)}")
            # save to csv for inspection
            bact_kmers_df.to_csv(outdir + 'bacterium_kmers.csv', index=False)
            bact_annot_df.to_csv(outdir + 'bacterium_kmers_annotation.csv', index=False)
        if not phage_kmers_df.empty and 'phage_annot_df' in locals():
            logger.log(f"Phage annotation sample:\n{phage_annot_df.head()}")
            logger.log(f"Succes/Total rate of kmer in phage df: {len(phage_annot_df)}/{len(phage_kmers_df)}")
            # save to csv for inspection
            phage_kmers_df.to_csv(outdir + 'phage_kmers.csv', index=False)
            phage_annot_df.to_csv(outdir + 'phage_kmers_annotation.csv', index=False)
            
        # Gene Annot Plotting
        try:
            title_suffix = f"({sort_by})" if sort_by in top_kmers_df.columns else ""
            plotting_utils = GAPlottingUtils(df=top_kmers_df, outdir=str(outdir), sort_by=sort_by)
            if not bact_annot_df.empty:
                plotting_utils.plot_top_genes(bact_annot_df, entity_type="bacterium", title_suffix=title_suffix)
                plotting_utils.plot_kmer_distribution(bact_annot_df, entity_type="bacterium", title_suffix=title_suffix)
                plotting_utils.plot_kmer_gene_network(bact_annot_df, entity_type="bacterium", top_kmers=args.network_top_kmers)
                plotting_utils.plot_kmer_against_ups_or_pfi(bact_annot_df, entity_type="bacterium")

            if not phage_annot_df.empty:
                plotting_utils.plot_top_genes(phage_annot_df, entity_type="phage", title_suffix=title_suffix)
                plotting_utils.plot_kmer_distribution(phage_annot_df, entity_type="phage", title_suffix=title_suffix)
                plotting_utils.plot_kmer_gene_network(phage_annot_df, entity_type="phage", top_kmers=args.network_top_kmers)
                plotting_utils.plot_kmer_against_ups_or_pfi(phage_annot_df, entity_type="phage")

        except Exception as e:
            logger.log(f"Error during gene annotation plotting: {e}")

        try:
            plotting_utils.plot_kmer_occurrence_per_partition()
        except Exception as e:
            logger.log(f"Error during k-mer occurrence plotting: {e}")

        try:
            title_suffix = f"({sort_by})" if sort_by in top_kmers_df.columns else ""
            plotting_utils = GAPlottingUtils(df=top_kmers_df, outdir=str(outdir), sort_by=sort_by)
            if not bact_annot_df.empty:
                GA.plot_annotation_pca(
                    annot_df=bact_annot_df,
                    entity_type="bacterium",
                    entity_col="bact",
                    gene_col="gene",
                    score_col=sort_by if sort_by in bact_annot_df.columns else None,
                    outdir=str(outdir),
                    title_suffix=title_suffix
                )

            if not phage_annot_df.empty:
                GA.plot_annotation_pca(
                    annot_df=phage_annot_df,
                    entity_type="phage",
                    entity_col="entity",
                    gene_col="product",
                    score_col=sort_by if sort_by in phage_annot_df.columns else None,
                    outdir=str(outdir),
                    title_suffix=title_suffix,
                )

        except Exception as e:
            logger.log(f"Error during PCA plotting: {e}")
        
        if args.hostrange_excel:
            try:
                # Hostrange heatmaps (requires --hostrange_excel)
                GA.plot_gene_hostrange_heatmaps(
                    bact_annot_df=bact_annot_df if not bact_annot_df.empty else pd.DataFrame(),
                    phage_annot_df=phage_annot_df if not phage_annot_df.empty else pd.DataFrame(),
                    input_excel=args.hostrange_excel,
                    sheet_name=args.hostrange_sheet,
                    outdir=str(outdir),
                    top_n=2)

            except Exception as e:
                logger.log(f"Error during hostrange heatmap plotting: {e}")

        ### Combined plotting
        frames = []
        if not bact_annot_df.empty:
            frames.append(_normalize_for_combine(
                bact_annot_df, species_col='bact', entity_col='gene',
                organism_label='bacterium'))
        if not phage_annot_df.empty:
            frames.append(_normalize_for_combine(
                phage_annot_df, species_col='entity', entity_col='product',
                organism_label='phage'))

        if frames:
            collected_df = pd.concat(frames, ignore_index=True, sort=False)
            collected_df.to_csv(outdir + 'top_kmers_annotations.csv', index=False)
            logger.log(f"Combined annotation frame: {len(collected_df)} rows, "
                    f"{collected_df['species'].nunique()} species, "
                    f"{collected_df['organism'].nunique()} organism types.")
        else:
            collected_df = pd.DataFrame()
            logger.log("No annotated rows to combine.")
        
        try:
            if not collected_df.empty:
                plotting_utils.plot_species_distribution_grid(
                    collected_df, value_col='decoded_kmer', top_n=30)
                plotting_utils.plot_species_distribution_grid(
                    collected_df, value_col='entity_label', top_n=30)
        except Exception as e:
            logger.log(f"Error during combined annotation plotting: {e}")

        # Concatenate annotation results and save
        try: 
            if not bact_annot_df.empty and not phage_annot_df.empty:
                combined_annot_df = pd.concat([bact_annot_df, phage_annot_df], ignore_index=True)
                combined_annot_df.to_csv(outdir + 'top_kmers_annotations.csv', index=False)
                logger.log("Top Kmers Annotations CSV saved as top_kmers_annotations.csv")
            
            elif not bact_annot_df.empty:
                bact_annot_df.to_csv(outdir + 'top_kmers_annotations.csv', index=False)
                logger.log("Bacterium Kmers Annotations CSV saved as top_kmers_annotations.csv")
            elif not phage_annot_df.empty:
                phage_annot_df.to_csv(outdir + 'top_kmers_annotations.csv', index=False)
                logger.log("Phage Kmers Annotations CSV saved as top_kmers_annotations.csv")
            
            # Additionally: build a mapping of decoded_kmer -> annotated genes/products
            try:
                mapping_frames = []
                # Bacterium mapping: 'decoded_kmer' -> 'gene'
                if not bact_annot_df.empty:
                    if 'decoded_kmer' in bact_annot_df.columns and 'gene' in bact_annot_df.columns:
                        df_bmap = bact_annot_df[['decoded_kmer', 'gene']].dropna()
                        if not df_bmap.empty:
                            df_bmap = df_bmap.groupby('decoded_kmer')['gene'].agg(lambda x: ';'.join(sorted(set(x)))).reset_index()
                            df_bmap = df_bmap.rename(columns={'gene': 'mapped_genes'})
                            df_bmap['organism'] = 'bacterium'
                            mapping_frames.append(df_bmap)

                # Phage mapping: 'decoded_kmer' -> 'product'
                if not phage_annot_df.empty:
                    if 'decoded_kmer' in phage_annot_df.columns and 'product' in phage_annot_df.columns:
                        df_pmap = phage_annot_df[['decoded_kmer', 'product']].dropna()
                        if not df_pmap.empty:
                            df_pmap = df_pmap.groupby('decoded_kmer')['product'].agg(lambda x: ';'.join(sorted(set(x)))).reset_index()
                            df_pmap = df_pmap.rename(columns={'product': 'mapped_genes'})
                            df_pmap['organism'] = 'phage'
                            mapping_frames.append(df_pmap)

                if mapping_frames:
                    kmer_map_df = pd.concat(mapping_frames, ignore_index=True, sort=False)
                    # Count how many distinct genes/products each kmer maps to
                    kmer_map_df['num_genes'] = kmer_map_df['mapped_genes'].apply(lambda s: 0 if pd.isna(s) or s == '' else len(str(s).split(';')))
                    # Save mapping CSV for user inspection
                    kmer_map_df.to_csv(outdir + 'kmer_to_genes_mapping.csv', index=False)
                    logger.log("Saved kmer-to-genes mapping CSV: kmer_to_genes_mapping.csv")

                    # Write a short human-readable summary with examples of multi-mapping kmers
                    total_kmers = len(kmer_map_df)
                    multi_map_count = int((kmer_map_df['num_genes'] > 1).sum())
                    examples = kmer_map_df[kmer_map_df['num_genes'] > 1].head(20)
                    summary_lines = [f"Total unique annotated kmers: {total_kmers}", f"Kmers mapping to multiple genes/products: {multi_map_count}", "\nExamples of kmers mapping to multiple genes/products (up to 20):"]
                    for _, r in examples.iterrows():
                        summary_lines.append(f"{r.get('decoded_kmer','<unknown kmer>')} ({r['organism']}): {r['mapped_genes']}")
                    with open(outdir + 'top_kmers_mapping_summary.txt', 'w') as summary_f:
                        summary_f.write('\n'.join(summary_lines))
                    logger.log("Saved human-readable mapping summary: top_kmers_mapping_summary.txt")

                    # Log concise counts for immediate visibility
                    logger.log(f"Kmer mapping summary: {total_kmers} unique kmers; {multi_map_count} map to multiple genes/products.")

                    # (Agent-only TODO tracking completed outside of runtime)

            except Exception as e:
                logger.log(f"Warning: Failed to build/save kmer->genes mapping: {e}")
        except Exception as e:
            raise ValueError(f"Error saving top k-mers annotations: {e}")

    else:
        logger.log("No valid top k-mers data found.")

    logger.log(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Script execution completed.")
    logfile.close()

if __name__ == "__main__":
    if parse_arguments().base_dir:
        args = parse_arguments()
        main(base_dir=args.base_dir.strip(" "), outdir=args.out_dir.strip(" "), x_col=args.x_col, hue_col=args.hue_col, group_x_col=args.group_x_col, group_hue_col=args.group_hue_col, ignore_failed_runs=args.ignore_failed_runs)
    else:
        main()