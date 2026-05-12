#!/net/domus/home/people/s215045/miniconda3/bin/python

import os
import re
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from time import time, sleep
from datetime import datetime
from paths import data_prod_path, path_to_nn_runs
from analysis import GeneAnalysis
outdir_default = data_prod_path + "iterExclClus/"
print(path_to_nn_runs)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Metric Extraction Script")
    parser.add_argument("--base_dir", type=str, default=path_to_nn_runs, 
                        help="Base directory containing run folders")
    parser.add_argument("--out_dir", type=str, default=outdir_default,
                        help="Directory to save output graphs and CSV")
    
    ## Optional grouping arguments for more flexible plotting 
    parser.add_argument("--x_col", type=str, default=None,
                        help="Column to use for x-axis in plots")
    parser.add_argument("--hue_col", type=str, default=None,
                        help="Column to use for color coding in plots")
    parser.add_argument("--group_x_col", type=str, default=None,
                        help="Column to use for grouping x-axis in plots")
    parser.add_argument("--group_hue_col", type=str, default=None,
                        help="Column to use for grouping hue in plots")
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
    
    # Extract Params from logfile
    params_dict = None

    # Preferred (legacy) format: single-line dict
    params_match = re.search(r"^Params:\s*(\{.*\})\s*$", content, re.MULTILINE)
    if params_match:
        params_str = params_match.group(1)
        try:
            params_dict = ast.literal_eval(params_str)
        except Exception as e:
            print(f"Error parsing single-line Params in {file_path}: {e}")
            return None
    else:
        # New format: multiline key-value block after "Params:"
        block_match = re.search(
            r"^Params:\s*\n(.*?)(?:^\s*#+\s*$|\n\s*\n)",
            content,
            re.MULTILINE | re.DOTALL
        )

        if not block_match:
            print(f"Could not find Params section in {file_path}")
            return None

        params_dict = {}
        for line in block_match.group(1).splitlines():
            stripped = line.strip()
            if not stripped or ':' not in stripped:
                continue

            key, value_str = stripped.split(':', 1)
            key = key.strip()
            value_str = value_str.strip()

            if value_str == "":
                params_dict[key] = None
                continue

            try:
                params_dict[key] = ast.literal_eval(value_str)
            except Exception:
                # Keep plain (unquoted) strings as-is
                params_dict[key] = value_str

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
            print(f"Could not extract n/k from Params in {file_path}")
            return None

    except Exception as e:
        print(f"Error parsing Params in {file_path}: {e}")
        return None

    # Extract Accuracy 
    acc_match = re.search(r"Standard test accuracy:\s+([\d.]+)", content)
    if acc_match:
        metrics['test_accuracy'] = float(acc_match.group(1))
    
    unseen_acc_match = re.search(r"truly unseen test accuracy:\s+([\d.]+)", content)
    if unseen_acc_match:
        metrics['unseen_test_accuracy'] = float(unseen_acc_match.group(1))

    # Extract Balanced Accuracy
    ba_match = re.search(r"Standard test balanced accuracy:\s+([\d.]+)", content)
    if ba_match:
        metrics['test_balanced_accuracy'] = float(ba_match.group(1))
    unseen_ba_match = re.search(r"truly unseen test balanced accuracy:\s+([\d.]+)", content)
    if unseen_ba_match:
        metrics['unseen_test_balanced_accuracy'] = float(unseen_ba_match.group(1))

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
        plt.title(f'Top {entity_type.capitalize()} Kmers Annotated Genes {title_suffix}')
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
        plt.title(f'Distribution of Top {entity_type.capitalize() }Kmers {title_suffix}')
        plt.xlabel('Kmers')
        plt.ylabel('Kmer Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.outdir + f'kmer_distribution_{entity_type}.png')
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

        self.outdir = outdir

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

        print(f"Dataframe length: {len(self.df)}. Recognized metrics:")
        for col in self.df.columns:
            print(f"  {col}: {self.df[col].dtype}")
        print(self.df)

    def _plot_cm_bars(self, title_suffix=""):
        # 1. Prepare the Confusion Matrix Data
        # We melt the dataframe so 'TN', 'FN', 'FP', 'TP' become categories in one column
        cm_cols = ['TN', 'FN', 'FP', 'TP']
        cm_df = self.df[cm_cols + [self.x_col]].copy()
        cm_melted = cm_df.melt(id_vars=self.x_col, var_name='Metric', value_name='Count')
        cm_melted['Metric'] = cm_melted['folder'] + '_' + cm_melted['Metric'] # Combine folder and metric for unique bars
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
            y='Count', 
            hue='MetricType', 
            palette=metric_colors, 
            edgecolor='black',
        )
        plt.legend(title='Confusion Matrix Metric', bbox_to_anchor=(1.05, 1), loc='upper left')

        # 4. Add labels on top of bars (similar to your reference image)
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        fontsize=6, color='black', 
                        xytext=(0, 7), 
                        textcoords='offset points')

        # 5. Formatting
        ax.tick_params(axis='x', labelsize=8)
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Confusion Matrix Components by Run {title_suffix}', fontsize=14, weight='bold', pad=20)
        plt.ylabel('Count (Number of Samples)', fontsize=10)
        plt.xlabel('Run / Configuration', fontsize=10)
        
        # Place legend outside to the right
        plt.legend(title='Metrics', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        
        plt.tight_layout()
        
        # 6. Save
        plt.savefig(self.outdir + 'confusion_matrix_by_run.png', dpi=300)
        plt.close() # Close to free up memory
        print("Saved: confusion_matrix_by_run.png")

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
        plt.xlabel(x_col.capitalize(), fontsize=12)
        if hue_col: plt.legend(title=hue_col.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(outpath)
        print(f"Saved: {outpath}")

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
        self._plot_cm_bars(title_suffix=self.title_suffix)

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

def main(base_dir=path_to_nn_runs, outdir=outdir_default, x_col=None, hue_col=None, group_x_col=None, group_hue_col=None):
    all_data = []
    top_kmers_df = pd.DataFrame() # Placeholder top_kmers_csv file
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found.")
        return
    else:
        print(f"Scanning directory: {base_dir}")
    
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        print(f"Created output directory: {outdir}")
    else:
        print(f"Output directory already exists: {outdir}")
    
    logfile_path = os.path.join(outdir, "collect_iterres_log.txt")
    logfile = open(logfile_path, 'w')
    print(f"{datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")} collect_iterres started. Scanning {base_dir} for log files.", file=logfile)

    # Iterate through all folders in nn_runs
    for folder_name in os.listdir(base_dir):          
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            # Search for log files in this specific run folder
            for file in os.listdir(folder_path):

                # Extract metrics from log files 
                if file.endswith(".txt") or file.endswith(".log"):
                    log_path = os.path.join(folder_path, file)
                    metrics = extract_metrics_from_log(log_path)
                    if metrics and metrics['test_accuracy'] is not None:
                        metrics['folder'] = folder_name
                        all_data.append(metrics)

                # Extract top kmers from pair_kmers.csv files
                elif file.endswith("pair_kmers.csv"):
                    top_kmers_path = os.path.join(folder_path, file)
                    try:
                        df_kmers = pd.read_csv(top_kmers_path)
                        df_kmers['folder'] = folder_name
                        top_kmers_df = pd.concat([top_kmers_df, df_kmers], ignore_index=True)
                    except Exception as e:
                        print(f"Error reading {top_kmers_path}: {e}")

    ### Metrics Extraction Summary and Plotting ###
    if all_data:
        df = pd.DataFrame(all_data)
        print(f"Extracted metrics from {len(df)} log files.")
        try:
            group_x_col = group_x_col.lower()
        except Exception:
            print(f"Unable to process group_x_col: {group_x_col}")
        plotting = MetricPlottingUtils(df=df, outdir=str(outdir), x_col=x_col, hue_col=hue_col, x_col_by_cluster=(group_x_col == 'cluster'), x_col_by_phage=(group_x_col == 'phage'))
        plotting.plot_graphs()
        # Optional: save the raw data for inspection
        df.to_csv(outdir +'all_runs_summary.csv', index=False)
        print("Summary CSV saved as all_runs_summary.csv")
    else:
        print("No valid data found to plot.")
    
    ### Top Kmers Annotation Summary and Plotting ###
    if not top_kmers_df.empty:
        print(f"Extracted top k-mers from {len(top_kmers_df['folder'].unique())} files.")

        # Split by "entity" column 
        bact_kmers_df = top_kmers_df[top_kmers_df['entity'] == 'bacterium']
        phage_kmers_df = top_kmers_df[top_kmers_df['entity'] == 'phage' or top_kmers_df['entity'] == 'bacteriophage']

        # Gene analysis
        try:
            GA = GeneAnalysis()
            if not bact_kmers_df.empty:
                bact_annot_df = GA.batch_bact_annotate(bkmers=bact_kmers_df['decoded_kmer'].tolist(), bact_names=bact_kmers_df['entity'].tolist(), data_prod_path=data_prod_path)
            else:
                print("No valid bacterium k-mers data found for annotation.")

            if not phage_kmers_df.empty:
                phage_annot_df = GA.batch_phage_annotate(pkmers=phage_kmers_df['decoded_kmer'].tolist(), phage_names=phage_kmers_df['entity'].tolist(), data_prod_path=data_prod_path)
            else:
                print("No valid phage k-mers data found for annotation.")
        except Exception as e:
            raise ValueError(f"Error during gene annotation: {e}")

        # Gene Annot Plotting
        try:
            plotting_utils = GAPlottingUtils(df=top_kmers_df, outdir=str(outdir))
            if not bact_kmers_df.empty:
                plotting_utils.plot_top_genes(bact_annot_df, entity_type="bacterium", title_suffix="(PFI)")
                plotting_utils.plot_kmer_distribution(bact_annot_df, entity_type="bacterium", title_suffix="(PFI)")
            if not phage_kmers_df.empty:
                plotting_utils.plot_top_genes(phage_annot_df, entity_type="phage", title_suffix="(PFI)")
                plotting_utils.plot_kmer_distribution(phage_annot_df, entity_type="phage", title_suffix="(PFI)")
        except Exception as e:
            raise ValueError(f"Error during gene annotation plotting: {e}")

        # Concatenate annotation results and save
        try: 
            if not bact_kmers_df.empty and not phage_kmers_df.empty:
                combined_annot_df = pd.concat([bact_annot_df, phage_annot_df], ignore_index=True)
                combined_annot_df.to_csv(outdir + 'top_kmers_annotations.csv', index=False)
                print("Top Kmers Annotations CSV saved as top_kmers_annotations.csv")
            
            elif not bact_kmers_df.empty:
                bact_annot_df.to_csv(outdir + 'top_kmers_annotations.csv', index=False)
                print("Bacterium Kmers Annotations CSV saved as top_kmers_annotations.csv")
            elif not phage_kmers_df.empty:
                phage_annot_df.to_csv(outdir + 'top_kmers_annotations.csv', index=False)
                print("Phage Kmers Annotations CSV saved as top_kmers_annotations.csv")
        except Exception as e:
            raise ValueError(f"Error saving top k-mers annotations: {e}")

    else:
        print("No valid top k-mers data found.")

    logfile.close()

if __name__ == "__main__":
    if parse_arguments().base_dir:
        args = parse_arguments()
        main(base_dir=args.base_dir.strip(" "), outdir=args.out_dir.strip(" "), x_col=args.x_col, hue_col=args.hue_col, group_x_col=args.group_x_col, group_hue_col=args.group_hue_col)
    else:
        main()