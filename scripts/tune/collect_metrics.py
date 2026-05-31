#!/usr/bin/env python3
"""
Slim sweep collector for FFNN_inner.py runs.

Reuses the metrics-extraction + metrics-plotting code path from
collect_iterres.py, but drops everything that depends on --perform_pfi
(top_kmers_df, GAPlottingUtils, GeneAnalysis, paths.py, analysis.py).

Expects a directory layout produced by the SLURM sweep:
    <base_dir>/kf{N}_ep{N}_lr{val}/_run{N}/log_run*.txt

Each "kf*_ep*_lr*" subdir is a sweep cell; each "_run*" inside it is one
training run for that cell (so a cell can hold multiple repeats).
"""

import argparse
import ast
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
class DualLogger:
    def __init__(self, logfile=None):
        self.logfile = logfile

    def log(self, message="", end="\n"):
        print(message, end=end)
        if self.logfile:
            self.logfile.write(message + end)
            self.logfile.flush()

    def set_logfile(self, logfile):
        self.logfile = logfile


logger = DualLogger()


# ---------------------------------------------------------------------------
# Helpers copied / adapted from collect_iterres.py
# ---------------------------------------------------------------------------
def correct_deci_number(value):
    try:
        num = float(value)
        if num > 1:
            digits = len(str(int(num)))
            return num / (10 ** digits)
        return num
    except ValueError:
        logger.log(f"Warning: cannot convert '{value}' to float.")
        return value


def extract_metrics_from_log(file_path):
    """Pull key metrics out of a single FFNN log file. Returns dict or None."""
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

    # --- Params block ----------------------------------------------------
    params_dict = None
    params_match = re.search(r"^Params:\s*(\{.*\})\s*$", content, re.MULTILINE)
    if params_match:
        try:
            params_dict = ast.literal_eval(params_match.group(1))
        except Exception as e:
            logger.log(f"Error parsing single-line Params in {file_path}: {e}")
            return None
    else:
        params_dict = {}
        lines = content.splitlines()
        start = next((i for i, ln in enumerate(lines)
                      if re.search(r"\bParams:\s*$", ln)), None)
        if start is None:
            logger.log(f"Could not find Params section in {file_path}")
            return None

        kv_re = re.compile(
            r"^(?:.*?\s-\sINFO\s-\s*)?(?P<key>[A-Za-z0-9_]+):\s*(?P<value>.*)$"
        )
        for line in lines[start + 1:]:
            stripped = line.strip()
            if not stripped:
                if params_dict:
                    break
                continue
            m = kv_re.match(line)
            if not m:
                if params_dict:
                    break
                continue
            key = m.group('key').strip()
            val = m.group('value').strip()
            if not key or key.lower() == 'info':
                tail = line.split(' - INFO - ', 1)[-1].strip()
                tm = re.match(r"(?P<key>[A-Za-z0-9_]+):\s*(?P<value>.*)$", tail)
                if tm:
                    key, val = tm.group('key').strip(), tm.group('value').strip()
            if val == "":
                params_dict[key] = None
                continue
            try:
                params_dict[key] = ast.literal_eval(val)
            except Exception:
                params_dict[key] = val

        if not params_dict:
            logger.log(f"Could not parse Params entries in {file_path}")
            return None

    # nk -> n, k
    nk = params_dict.get('nk')
    if isinstance(nk, (list, tuple)) and len(nk) >= 2:
        metrics['n'], metrics['k'] = nk[0], nk[1]
    else:
        metrics['n'] = params_dict.get('n')
        metrics['k'] = params_dict.get('k')
    if metrics['n'] is None or metrics['k'] is None:
        logger.log(f"Could not extract n/k from Params in {file_path}")
        return None

    # Pull the hyperparameters we're sweeping straight from the log so the
    # CSV/plots are correct even if the folder name is renamed.
    metrics['kf_n_splits']  = params_dict.get('kf_n_splits')
    metrics['n_epochs']     = params_dict.get('n_epochs')
    metrics['learning_rate'] = params_dict.get('learning_rate')
    metrics['batch_size']   = params_dict.get('batch_size')

    # --- Test accuracy ---------------------------------------------------
    m = re.search(r"Standard test accuracy:\s+([\d.]+)", content)
    if m:
        metrics['test_accuracy'] = correct_deci_number(m.group(1))
    m = re.search(r"truly unseen test accuracy:\s+([\d.]+)", content)
    if m:
        metrics['unseen_test_accuracy'] = correct_deci_number(m.group(1))

    m = re.search(r"Standard test balanced accuracy:\s+([\d.]+)", content)
    if m:
        metrics['test_balanced_accuracy'] = correct_deci_number(m.group(1))
    m = re.search(r"truly unseen test balanced accuracy:\s+([\d.]+)", content)
    if m:
        metrics['unseen_test_balanced_accuracy'] = correct_deci_number(m.group(1))

    # --- Precision / Recall / F1 ----------------------------------------
    base = re.search(
        r"Baseline\s*\([^)]*\)\s*(?:->|:)\s*Precision:\s*([\d.]+)\s*,\s*Recall:\s*([\d.]+)\s*,\s*F1:\s*([\d.]+)",
        content,
    )
    if base:
        metrics['precision'] = correct_deci_number(base.group(1))
        metrics['recall']    = correct_deci_number(base.group(2))
        metrics['f1']        = correct_deci_number(base.group(3))
    else:
        best = re.search(
            r"Best\s+threshold\s+by\s+F1\s*(?:->|:)\s*(?:threshold=[^,]+,\s*)?"
            r"Precision\s*=\s*([\d.]+)\s*,\s*Recall\s*=\s*([\d.]+)\s*,\s*F1\s*=\s*([\d.]+)",
            content, re.IGNORECASE,
        )
        if best:
            metrics['precision'] = correct_deci_number(best.group(1))
            metrics['recall']    = correct_deci_number(best.group(2))
            metrics['f1']        = correct_deci_number(best.group(3))

    # --- Confusion matrix ------------------------------------------------
    cm = re.search(
        r"--- Confusion Matrix ---\s*\[\[\s*(\d+)\s+(\d+)\s*\]\s*\[\s*(\d+)\s+(\d+)\s*\]\s*\]",
        content, re.DOTALL,
    )
    if cm:
        metrics['TN'] = int(cm.group(1))
        metrics['FP'] = int(cm.group(2))
        metrics['FN'] = int(cm.group(3))
        metrics['TP'] = int(cm.group(4))
    else:
        cm2 = re.search(
            r"Confusion matrix:.*?\[\s*(\d+)\s+(\d+)\s*\].*?\[\s*(\d+)\s+(\d+)\s*\]",
            content, re.DOTALL,
        )
        if cm2:
            metrics['TN'] = int(cm2.group(1))
            metrics['FN'] = int(cm2.group(2))
            metrics['FP'] = int(cm2.group(3))
            metrics['TP'] = int(cm2.group(4))

    # --- Dataset sizes ---------------------------------------------------
    p = re.search(r"Built dataset with (\d+) pairs and excluded (\d+) pairs", content)
    if p:
        metrics['train_pairs'] = int(p.group(1))
        metrics['test_pairs']  = int(p.group(2))
        if metrics['train_pairs'] > 0:
            metrics['test_train_ratio'] = round(
                metrics['test_pairs'] / metrics['train_pairs'], 4
            )

    metrics['status'] = "INFO - Process completed in" in content
    return metrics


def parse_tag(folder_name):
    """Extract kf, ep, lr from a sweep tag like 'kf3_ep50_lr1e-2'."""
    out = {'tag_kf': None, 'tag_ep': None, 'tag_lr': None}
    m = re.search(r"kf(\d+)", folder_name)
    if m: out['tag_kf'] = int(m.group(1))
    m = re.search(r"ep(\d+)", folder_name)
    if m: out['tag_ep'] = int(m.group(1))
    m = re.search(r"lr([0-9eE.+-]+)", folder_name)
    if m:
        try:
            out['tag_lr'] = float(m.group(1))
        except ValueError:
            pass
    return out


# ---------------------------------------------------------------------------
# Plotting (mirrors MetricPlottingUtils from collect_iterres.py, slimmed)
# ---------------------------------------------------------------------------
class SweepPlotter:
    METRIC_COLS = [
        ('test_accuracy',                'Test Accuracy'),
        ('test_balanced_accuracy',       'Test Balanced Accuracy'),
        ('unseen_test_accuracy',         'Truly Unseen Test Accuracy'),
        ('unseen_test_balanced_accuracy','Truly Unseen Balanced Accuracy'),
        ('precision',                    'Precision'),
        ('recall',                       'Recall'),
        ('f1',                           'F1'),
    ]

    def __init__(self, df, outdir, x_col='folder', hue_col=None):
        self.df = df.copy()
        self.outdir = outdir if outdir.endswith('/') else outdir + '/'
        os.makedirs(self.outdir, exist_ok=True)

        # numeric coercion
        for c, _ in self.METRIC_COLS:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors='coerce')
        self.df['status'] = self.df['status'].apply(
            lambda s: 'passed' if s else 'failed'
        )
        self.x_col = x_col
        self.hue_col = hue_col
        self.x_label = x_col

    def _plot_bars(self, x_col, y_col, hue_col, title, ylabel, outpath):
        if y_col not in self.df.columns or self.df[y_col].dropna().empty:
            logger.log(f"Skipping {y_col}: no data.")
            return
        plt.figure(figsize=(12, 6))
        sns.set_style("whitegrid")
        order = self.df[x_col].drop_duplicates().tolist()
        ax = sns.barplot(
            data=self.df, x=x_col, y=y_col,
            hue=hue_col, palette='viridis',
            edgecolor='black', order=order,
        )
        ax.set_ylim(0, 1)
        plt.xticks(rotation=75, ha='right')
        plt.title(title, fontsize=14, pad=12)
        plt.ylabel(ylabel)
        plt.xlabel(self.x_label)
        if hue_col:
            plt.legend(title=hue_col, bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close()
        logger.log(f"Saved: {outpath}")

    def _plot_cm_bars(self):
        needed = ['TN', 'FN', 'FP', 'TP', self.x_col]
        if not all(c in self.df.columns for c in needed):
            return
        cm_df = self.df[needed].copy()
        melted = cm_df.melt(id_vars=self.x_col,
                            var_name='Metric', value_name='Count')
        order = melted[self.x_col].drop_duplicates().tolist()
        plt.figure(figsize=(16, 6))
        sns.set_style("whitegrid")
        palette = {'TN': '#FF6B6B', 'FN': '#4ECDC4',
                   'FP': '#45B7D1', 'TP': '#B7FF78'}
        sns.barplot(data=melted, x=self.x_col, y='Count',
                    hue='Metric', palette=palette,
                    edgecolor='black', order=order)
        plt.xticks(rotation=75, ha='right')
        plt.title('Confusion Matrix Components by Run', fontsize=14, pad=12)
        plt.ylabel('Count')
        plt.xlabel(self.x_label)
        plt.legend(title='Metric', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        out = self.outdir + 'confusion_matrix_by_run.png'
        plt.savefig(out, dpi=150)
        plt.close()
        logger.log(f"Saved: {out}")

    def _plot_avg_cm(self):
        if not all(c in self.df.columns for c in ['TN', 'FN', 'FP', 'TP']):
            return
        avg = self.df[['TN', 'FN', 'FP', 'TP']].mean()
        mat = np.array([[avg['TN'], avg['FP']],
                        [avg['FN'], avg['TP']]])
        plt.figure(figsize=(5.5, 4.5))
        sns.heatmap(mat, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=['Pred 0', 'Pred 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        plt.title(f'Averaged Confusion Matrix\n({len(self.df)} runs)')
        plt.tight_layout()
        out = self.outdir + 'averaged_confusion_matrix.png'
        plt.savefig(out, dpi=150)
        plt.close()
        logger.log(f"Saved: {out}")

    def _plot_metric_vs_hyperparam(self, hp_col, hp_label):
        """For each metric, line plot value vs the swept hyperparameter."""
        if hp_col not in self.df.columns or self.df[hp_col].dropna().empty:
            return
        for y, ylabel in self.METRIC_COLS:
            if y not in self.df.columns or self.df[y].dropna().empty:
                continue
            plt.figure(figsize=(8, 5))
            sns.set_style("whitegrid")
            sns.pointplot(data=self.df, x=hp_col, y=y,
                          errorbar='sd', color='steelblue')
            plt.xticks(rotation=45)
            plt.title(f'{ylabel} vs {hp_label}')
            plt.xlabel(hp_label)
            plt.ylabel(ylabel)
            plt.tight_layout()
            out = self.outdir + f'{y}_vs_{hp_col}.png'
            plt.savefig(out, dpi=150)
            plt.close()
            logger.log(f"Saved: {out}")

    def plot_all(self):
        for y, ylabel in self.METRIC_COLS:
            self._plot_bars(
                x_col=self.x_col, y_col=y, hue_col=self.hue_col,
                title=f'FFNN {ylabel} by run',
                ylabel=ylabel,
                outpath=self.outdir + f'{y}_by_run.png',
            )
        self._plot_cm_bars()
        self._plot_avg_cm()
        for hp, label in [('tag_kf', 'kf_n_splits'),
                          ('tag_ep', 'n_epochs'),
                          ('tag_lr', 'learning_rate')]:
            self._plot_metric_vs_hyperparam(hp, label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Summarize FFNN sweep runs (no PFI / GA)."
    )
    ap.add_argument("--base_dir", required=True,
                    help="Directory holding the per-run subfolders.")
    ap.add_argument("--out_dir", required=True,
                    help="Where to write CSV + plots.")
    ap.add_argument("--x_col", default="folder",
                    help="Column to use on the x-axis of per-run bar plots.")
    ap.add_argument("--hue_col", default=None,
                    help="Optional column to use as hue.")
    return ap.parse_args()


def main():
    args = parse_args()
    base_dir = args.base_dir.rstrip('/')
    out_dir  = args.out_dir.rstrip('/') + '/'
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "collect_sweep_metrics.log")
    logger.set_logfile(open(log_path, 'w'))
    logger.log(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] scanning {base_dir}")

    rows = []
    for cell in tqdm(sorted(os.listdir(base_dir)), desc="Sweep cells"):
        cell_dir = os.path.join(base_dir, cell)
        if not os.path.isdir(cell_dir):
            continue

        # Each sweep cell contains one or more "_runN" subdirs (the actual
        # run output dirs). Fall back to the cell dir itself if no nested
        # run dirs exist, so the script is forgiving of layout drift.
        run_subdirs = sorted(
            d for d in os.listdir(cell_dir)
            if d.startswith("_run") and os.path.isdir(os.path.join(cell_dir, d))
        )
        if not run_subdirs:
            run_subdirs = [""]  # look directly inside the cell dir

        cell_hit = False
        for run_sub in run_subdirs:
            run_dir = os.path.join(cell_dir, run_sub) if run_sub else cell_dir
            log_files = [f for f in os.listdir(run_dir)
                         if f.endswith(".txt") and "log_run" in f.lower()]
            if not log_files:
                logger.log(f"  {cell}/{run_sub}: no log_run*.txt — skipped")
                continue
            log_files.sort(
                key=lambda f: os.path.getmtime(os.path.join(run_dir, f)),
                reverse=True,
            )
            log_path_i = os.path.join(run_dir, log_files[0])
            m = extract_metrics_from_log(log_path_i)
            if m is None:
                logger.log(f"  {cell}/{run_sub}: parse failed")
                continue
            m['folder']     = cell                # sweep cell (groups repeats)
            m['run_subdir'] = run_sub or "."      # which repeat inside the cell
            m['run_id']     = f"{cell}/{run_sub}" if run_sub else cell
            m.update(parse_tag(cell))
            rows.append(m)
            cell_hit = True

        if not cell_hit:
            logger.log(f"  {cell}: no usable runs found")

    if not rows:
        logger.log("No runs parsed. Exiting.")
        return

    df = pd.DataFrame(rows)

    # Decimal correction safety net.
    for c in ['test_accuracy', 'test_balanced_accuracy',
              'unseen_test_accuracy', 'unseen_test_balanced_accuracy',
              'precision', 'recall', 'f1']:
        if c in df.columns and (df[c] > 1).any():
            logger.log(f"Correcting decimals in {c}")
            df[c] = df[c].apply(correct_deci_number)

    # Mark TP=0 runs as failed.
    if 'TP' in df.columns:
        df.loc[df['TP'].fillna(0) == 0, 'status'] = False

    csv_out = out_dir + 'sweep_summary.csv'
    df.to_csv(csv_out, index=False)
    logger.log(f"Wrote {csv_out} ({len(df)} runs)")

    # Pick best by balanced accuracy (or test_accuracy as fallback).
    score_col = 'test_balanced_accuracy' if 'test_balanced_accuracy' in df.columns else 'test_accuracy'
    passed = df[df['status'] == True]
    if not passed.empty and score_col in passed.columns:
        best = passed.sort_values(score_col, ascending=False).head(5)
        logger.log(f"\nTop 5 individual runs by {score_col}:")
        cols = ['run_id', 'tag_kf', 'tag_ep', 'tag_lr',
                'test_accuracy', 'test_balanced_accuracy', 'f1']
        cols = [c for c in cols if c in best.columns]
        logger.log(best[cols].to_string(index=False))

        # Also report best sweep cells (mean across their _runN repeats).
        group_cols = [c for c in ['folder', 'tag_kf', 'tag_ep', 'tag_lr']
                      if c in passed.columns]
        if group_cols and passed['folder'].duplicated().any():
            agg = (passed.groupby(group_cols, dropna=False)[score_col]
                          .agg(['mean', 'std', 'count'])
                          .sort_values('mean', ascending=False)
                          .head(5))
            logger.log(f"\nTop 5 sweep cells by mean {score_col} (over repeats):")
            logger.log(agg.to_string())

    # Plot.
    df_plot = df[df['status'] == True].copy() if (df['status'] == True).any() else df
    SweepPlotter(df_plot, outdir=out_dir,
                 x_col=args.x_col, hue_col=args.hue_col).plot_all()

    logger.log(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] done")


if __name__ == "__main__":
    main()