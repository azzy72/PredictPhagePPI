#!/usr/bin/python3
"""
summarize_collect_results.py
────────────────────────────
Scans a data_prod directory for every IterExclClus_* folder produced by
collect_iterres.py and generates cross-version / cross-sketch-type summary plots.

Folder naming convention (parsed automatically):
    IterExclClus_{DOWNDIR}_n{N}_k{K}              → v1, no harsh
    IterExclClus_{DOWNDIR}_n{N}_k{K}_v2           → v2, no harsh
    IterExclClus_{DOWNDIR}_n{N}_k{K}_v3           → v3, no harsh
    IterExclClus_{DOWNDIR}_n{N}_k{K}_v3_harsh     → v3, harsh

DOWNDIR options: SM_sketches, encoded_sketches, encoded_sketches_data2, SM_sketches_data2
Encoded-type folders additionally contain gene annotation results.

Usage:
    python3 summarize_collect_results.py \\
        --data_dir /path/to/data_prod \\
        --out_dir  /path/to/summary_output       # optional; default: data_dir/summary/
"""

import os
import re
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

VERSION_ORDER   = ['v1', 'v2', 'v3']
VERSION_PALETTE = {'v1': '#4878CF', 'v2': '#6ACC65', 'v3': '#D65F5F'}
METRIC_COLS     = [
    'test_accuracy', 'test_balanced_accuracy',
    'unseen_test_accuracy', 'unseen_test_balanced_accuracy',
    'precision', 'recall', 'f1',
]
METRIC_LABELS   = {
    'test_accuracy':                    'Test Acc',
    'test_balanced_accuracy':           'Test Bal Acc',
    'unseen_test_accuracy':             'Unseen Acc',
    'unseen_test_balanced_accuracy':    'Unseen Bal Acc',
    'precision':                        'Precision',
    'recall':                           'Recall',
    'f1':                               'F1',
}

# ── Folder-name regex ─────────────────────────────────────────────────────────
#   IterExclClus_{DOWNDIR}_n{N}_k{K}[_v<n>][_harsh]
#   DOWNDIR may contain underscores (e.g. encoded_sketches_data2),
#   so we anchor on _n<digits>_k<digits> as the reliable delimiter.
FOLDER_RE = re.compile(
    r'^IterExclClus_(.+)_n(\d+)_k(\d+)((?:_v\d+)?)((?:_harsh)?)$'
)


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='Summarise collect_iterres.py results.')
    p.add_argument('--data_dir', type=str, required=True,
                   help='Root directory containing IterExclClus_* folders (data_prod)')
    p.add_argument('--out_dir', type=str, default=None,
                   help='Output directory for summary plots (default: data_dir/summary/)')
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# FOLDER DISCOVERY & METADATA PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_folder_name(name: str):
    """Return metadata dict for a valid IterExclClus_* folder, else None."""
    m = FOLDER_RE.match(name)
    if not m:
        return None
    sketch_type  = m.group(1)                          # e.g. 'encoded_sketches_data2'
    n            = int(m.group(2))
    k            = int(m.group(3))
    version      = m.group(4).lstrip('_') or 'v1'     # '' → 'v1', '_v3' → 'v3'
    harsh        = bool(m.group(5))                    # '_harsh' → True
    is_encoded   = sketch_type.startswith('encoded')
    is_data2     = sketch_type.endswith('data2')
    return dict(
        folder_name=name, sketch_type=sketch_type,
        n=n, k=k, version=version, harsh=harsh,
        is_encoded=is_encoded, is_data2=is_data2,
    )


def load_all_results(data_dir: str):
    """
    Walk data_dir, parse every IterExclClus_* subdirectory, and load:
      - all_runs_summary.csv      → metrics_df
      - top_kmers_summary.csv     → kmers_df
      - top_kmers_annotations.csv → annot_df  (encoded folders only)

    Returns (metrics_df, kmers_df, annot_df).
    """
    metric_frames, kmer_frames, annot_frames = [], [], []

    for entry in sorted(os.listdir(data_dir)):
        meta = parse_folder_name(entry)
        if meta is None:
            continue
        base = os.path.join(data_dir, entry)

        def _load(fname):
            path = os.path.join(base, fname)
            if not os.path.exists(path):
                return None
            try:
                df = pd.read_csv(path)
                for key, val in meta.items():
                    df[key] = val
                return df
            except Exception as e:
                print(f'  Warning: could not read {path}: {e}')
                return None

        df = _load('all_runs_summary.csv');      metric_frames.append(df) if df is not None else None
        df = _load('top_kmers_summary.csv');     kmer_frames.append(df)   if df is not None else None
        df = _load('top_kmers_annotations.csv'); annot_frames.append(df)  if df is not None else None

    def _concat(frames):
        frames = [f for f in frames if f is not None and not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return _concat(metric_frames), _concat(kmer_frames), _concat(annot_frames)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, outdir: str, filename: str):
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {filename}')


def _versions(df: pd.DataFrame):
    """Return VERSION_ORDER entries that are actually in df['version']."""
    present = set(df['version'].dropna().unique())
    return [v for v in VERSION_ORDER if v in present]


def _numeric_cast(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PLOT FUNCTIONS  (numbered for easy scanning of output folder)
# ══════════════════════════════════════════════════════════════════════════════

# ── 00: how many runs per bucket ─────────────────────────────────────────────

def plot_run_count(df: pd.DataFrame, outdir: str):
    """Grouped bar: run count per version × sketch_type."""
    counts = df.groupby(['version', 'sketch_type']).size().reset_index(name='count')
    versions = _versions(df)
    n_sketch = df['sketch_type'].nunique()

    fig, ax = plt.subplots(figsize=(max(8, n_sketch * 2.5), 5))
    sns.barplot(data=counts, x='sketch_type', y='count', hue='version',
                hue_order=versions, palette=VERSION_PALETTE, ax=ax)
    ax.set_title('Number of Runs per Version × Sketch Type', fontsize=13, fontweight='bold')
    ax.set_xlabel('Sketch Type')
    ax.set_ylabel('Run Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    sns.despine(ax=ax)
    plt.tight_layout()
    _save(fig, outdir, '00_run_count.png')


# ── 01: metric distributions per version (box + strip) ───────────────────────

def plot_metric_boxes(df: pd.DataFrame, outdir: str):
    """Box + strip per metric, split by version."""
    versions = _versions(df)
    palette  = {v: VERSION_PALETTE[v] for v in versions}
    metrics  = [m for m in METRIC_COLS if m in df.columns]

    ncols = 4
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax  = axes[i]
        sub = df[['version', metric]].dropna()
        sns.boxplot(data=sub, x='version', y=metric, order=versions,
                    palette=palette, width=0.5, fliersize=3, ax=ax)
        sns.stripplot(data=sub, x='version', y=metric, order=versions,
                      color='black', alpha=0.3, size=2.5, jitter=True, ax=ax)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylim(-0.02, 1.05)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        sns.despine(ax=ax)

    for j in range(len(metrics), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Metric Distributions by Version  (non-harsh only)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save(fig, outdir, '01_metric_distributions_by_version.png')


# ── 02: mean metric per version × sketch_type bars ───────────────────────────

def plot_metric_by_sketchtype(df: pd.DataFrame, outdir: str):
    """Grouped bar of mean metric per sketch_type, hue = version."""
    versions = _versions(df)
    metrics  = [m for m in METRIC_COLS if m in df.columns]

    fig, axes = plt.subplots(1, len(metrics), figsize=(len(metrics) * 3.5, 5), sharey=False)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sub = df.groupby(['sketch_type', 'version'])[metric].mean().reset_index()
        sns.barplot(data=sub, x='sketch_type', y=metric, hue='version',
                    hue_order=versions, palette=VERSION_PALETTE, ax=ax)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=9, fontweight='bold')
        ax.set_xlabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.get_legend().remove()
        sns.despine(ax=ax)

    handles = [mpatches.Patch(color=VERSION_PALETTE[v], label=v) for v in versions]
    fig.legend(handles=handles, title='Version', loc='center right', bbox_to_anchor=(1.02, 0.5))
    fig.suptitle('Mean Metrics by Sketch Type and Version', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, outdir, '02_mean_metrics_by_sketchtype.png')


# ── 03: heatmap – mean metric across (version × sketch_type) ─────────────────

def plot_metric_heatmaps(df: pd.DataFrame, outdir: str):
    """One heatmap per key metric."""
    key_metrics = [m for m in
                   ['test_balanced_accuracy', 'unseen_test_balanced_accuracy', 'f1']
                   if m in df.columns]
    for metric in key_metrics:
        pivot = (df.groupby(['version', 'sketch_type'])[metric]
                   .mean().unstack('sketch_type'))
        pivot = pivot.reindex([v for v in VERSION_ORDER if v in pivot.index])
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(max(5, len(pivot.columns) * 1.6), 3))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                    vmin=0.4, vmax=1.0, linewidths=0.5, ax=ax,
                    cbar_kws={'label': METRIC_LABELS.get(metric, metric)})
        ax.set_title(f'Mean {METRIC_LABELS.get(metric, metric)}  —  Version × Sketch Type',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Version')
        plt.tight_layout()
        safe = metric.replace('_', '')
        _save(fig, outdir, f'03_heatmap_{safe}.png')


# ── 04: version progression line plot ────────────────────────────────────────

def plot_version_progression(df: pd.DataFrame, outdir: str):
    """
    Median metric vs version, one line per sketch_type.
    Shaded band = IQR. Shows whether training improved across versions.
    """
    versions     = _versions(df)
    key_metrics  = [m for m in
                    ['test_balanced_accuracy', 'unseen_test_balanced_accuracy', 'f1']
                    if m in df.columns]
    sketch_types = sorted(df['sketch_type'].unique())
    palette      = sns.color_palette('tab10', len(sketch_types))

    fig, axes = plt.subplots(1, len(key_metrics),
                             figsize=(len(key_metrics) * 5.5, 5), sharey=False)
    if len(key_metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, key_metrics):
        for st, color in zip(sketch_types, palette):
            sub = df[df['sketch_type'] == st]
            med = sub.groupby('version')[metric].median().reindex(versions)
            q25 = sub.groupby('version')[metric].quantile(0.25).reindex(versions)
            q75 = sub.groupby('version')[metric].quantile(0.75).reindex(versions)
            xs  = list(range(len(versions)))
            ax.plot(xs, med.values, marker='o', label=st, color=color, linewidth=2)
            ax.fill_between(xs, q25.values, q75.values, alpha=0.15, color=color)
        ax.set_xticks(range(len(versions)))
        ax.set_xticklabels(versions)
        ax.set_ylim(0, 1.05)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10, fontweight='bold')
        ax.set_xlabel('Version')
        ax.yaxis.set_major_locator(MaxNLocator(5))
        sns.despine(ax=ax)

    handles = [mpatches.Patch(color=c, label=s)
               for c, s in zip(palette, sketch_types)]
    fig.legend(handles=handles, title='Sketch Type',
               loc='lower center', bbox_to_anchor=(0.5, -0.14), ncol=len(sketch_types))
    fig.suptitle('Metric Progression Across Versions  (median ± IQR)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, outdir, '04_version_progression.png')


# ── 05: test vs unseen scatter (generalisation check) ────────────────────────

def plot_test_vs_unseen(df: pd.DataFrame, outdir: str):
    """
    Scatter: test_balanced_accuracy vs unseen_test_balanced_accuracy.
    Points on the diagonal generalise perfectly; below = overfitting.
    Coloured by version, shaped by sketch_type.
    """
    if not {'test_balanced_accuracy', 'unseen_test_balanced_accuracy'}.issubset(df.columns):
        return
    versions     = _versions(df)
    sketch_types = sorted(df['sketch_type'].unique())
    markers      = ['o', 's', '^', 'D', 'P']

    fig, ax = plt.subplots(figsize=(8, 7))
    for st, marker in zip(sketch_types, markers):
        for v in versions:
            sub = df[(df['sketch_type'] == st) & (df['version'] == v)].dropna(
                subset=['test_balanced_accuracy', 'unseen_test_balanced_accuracy'])
            ax.scatter(sub['test_balanced_accuracy'],
                       sub['unseen_test_balanced_accuracy'],
                       c=VERSION_PALETTE[v], marker=marker,
                       alpha=0.55, s=45, linewidths=0)

    # Diagonal reference
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.35, linewidth=1,
            label='Test = Unseen (no overfit)')
    ax.set_xlim(0, 1.05);  ax.set_ylim(0, 1.05)
    ax.set_xlabel('Test Balanced Accuracy')
    ax.set_ylabel('Unseen Test Balanced Accuracy')
    ax.set_title('Generalisation: Test vs Unseen Balanced Accuracy',
                 fontsize=12, fontweight='bold')

    v_h  = [mpatches.Patch(color=VERSION_PALETTE[v], label=f'Version {v}')
            for v in versions]
    st_h = [plt.Line2D([0],[0], marker=m, color='grey', linestyle='None', label=s, markersize=7)
            for m, s in zip(markers, sketch_types)]
    ax.legend(handles=v_h + st_h, fontsize=8, loc='lower right')
    sns.despine(ax=ax)
    plt.tight_layout()
    _save(fig, outdir, '05_generalisation_scatter.png')


# ── 06: harsh vs normal comparison ───────────────────────────────────────────

def plot_harsh_vs_normal(df: pd.DataFrame, outdir: str):
    """
    Side-by-side boxplots for key metrics: normal vs harsh filtering,
    grouped by version.
    """
    key_metrics = [m for m in
                   ['test_balanced_accuracy', 'unseen_test_balanced_accuracy', 'f1']
                   if m in df.columns]
    versions    = _versions(df)

    fig, axes = plt.subplots(1, len(key_metrics), figsize=(len(key_metrics) * 5, 5))
    if len(key_metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, key_metrics):
        sub          = df[['version', 'harsh', metric]].dropna().copy()
        sub['label'] = sub['version'] + '\n' + sub['harsh'].map(
                           {True: 'harsh', False: 'normal'})
        order = [f'{v}\n{h}' for v in versions for h in ['normal', 'harsh']
                 if f'{v}\n{h}' in sub['label'].values]
        palette = {}
        for v in versions:
            palette[f'{v}\nnormal'] = VERSION_PALETTE[v]
            palette[f'{v}\nharsh']  = sns.desaturate(VERSION_PALETTE[v], 0.45)

        sns.boxplot(data=sub, x='label', y=metric, order=order,
                    palette=palette, width=0.55, fliersize=3, ax=ax)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylim(0, 1.05)
        sns.despine(ax=ax)

    fig.suptitle('Effect of Harsh Filtering on Key Metrics',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, outdir, '06_harsh_vs_normal.png')


# ── 07: pass / fail stacked bars ─────────────────────────────────────────────

def plot_pass_rate(df: pd.DataFrame, outdir: str):
    """Stacked bar: passed vs failed runs per version × sketch_type."""
    if 'status' not in df.columns:
        return
    df2 = df.copy()
    df2['passed'] = df2['status'].isin([True, 'True', 'passed', 1])

    grouped = (df2.groupby(['version', 'sketch_type'])['passed']
                  .agg(['sum', 'count']).reset_index()
                  .rename(columns={'sum': 'n_passed', 'count': 'n_total'}))
    grouped['n_failed'] = grouped['n_total'] - grouped['n_passed']

    versions     = _versions(df)
    sketch_types = sorted(grouped['sketch_type'].unique())

    fig, axes = plt.subplots(1, len(sketch_types),
                             figsize=(max(6, len(sketch_types) * 4), 5), sharey=True)
    if len(sketch_types) == 1:
        axes = [axes]

    for ax, st in zip(axes, sketch_types):
        sub = (grouped[grouped['sketch_type'] == st]
               .set_index('version').reindex(versions).fillna(0))
        xs  = range(len(versions))
        ax.bar(xs, sub['n_passed'], label='Passed', color='#4CAF50', edgecolor='white')
        ax.bar(xs, sub['n_failed'], bottom=sub['n_passed'],
               label='Failed', color='#F44336', edgecolor='white')
        # Percentage label inside each bar
        for x, (_, row) in zip(xs, sub.iterrows()):
            total = row['n_total']
            if total > 0:
                pct = row['n_passed'] / total * 100
                ax.text(x, row['n_passed'] / 2, f'{pct:.0f}%',
                        ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        ax.set_xticks(list(xs))
        ax.set_xticklabels(versions)
        ax.set_title(st, fontsize=10, fontweight='bold')
        ax.set_xlabel('Version')
        ax.set_ylabel('Run Count')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        sns.despine(ax=ax)

    axes[0].legend(loc='upper left')
    fig.suptitle('Pass / Fail Run Counts by Version × Sketch Type',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, outdir, '07_pass_rate.png')


# ── 08: averaged normalised confusion matrices per version ───────────────────

def plot_confusion_matrices(df: pd.DataFrame, outdir: str):
    """Normalised (%) average confusion matrix per version."""
    cm_cols = ['TN', 'FN', 'FP', 'TP']
    if not all(c in df.columns for c in cm_cols):
        return
    versions = _versions(df)

    fig, axes = plt.subplots(1, len(versions),
                             figsize=(len(versions) * 4.5, 4))
    if len(versions) == 1:
        axes = [axes]

    for ax, v in zip(axes, versions):
        avg   = df[df['version'] == v][cm_cols].dropna().mean()
        total = avg.sum()
        mat   = np.array([[avg['TN'], avg['FP']],
                           [avg['FN'], avg['TP']]]) / (total or 1)
        sns.heatmap(mat, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=['Pred 0', 'Pred 1'],
                    yticklabels=['Act 0',  'Act 1'],
                    linewidths=0.5, cbar=False, ax=ax)
        ax.set_title(f'Version {v}', fontsize=11, fontweight='bold')

    fig.suptitle('Normalised Average Confusion Matrix per Version',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, outdir, '08_confusion_matrix_by_version.png')


# ── 09: radar chart ───────────────────────────────────────────────────────────

def plot_radar(df: pd.DataFrame, outdir: str):
    """Spider/radar: mean metric profile per version."""
    metrics  = [m for m in
                ['test_balanced_accuracy', 'unseen_test_balanced_accuracy',
                 'precision', 'recall', 'f1']
                if m in df.columns]
    labels   = [METRIC_LABELS.get(m, m) for m in metrics]
    versions = _versions(df)
    N        = len(metrics)
    angles   = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles  += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for v in versions:
        vals = df[df['version'] == v][metrics].mean().tolist() + [None]
        vals[-1] = vals[0]
        ax.plot(angles, vals, 'o-', linewidth=2,
                label=f'Version {v}', color=VERSION_PALETTE[v])
        ax.fill(angles, vals, alpha=0.10, color=VERSION_PALETTE[v])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=7)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=10)
    ax.set_title('Mean Metric Radar by Version', fontsize=13,
                 fontweight='bold', pad=22)
    _save(fig, outdir, '09_radar_by_version.png')


# ── 10: parallel coordinates ─────────────────────────────────────────────────

def plot_parallel_coordinates(df: pd.DataFrame, outdir: str):
    """
    Every run is a faint line across all metrics; bold lines show per-version
    median. Good for seeing run consistency and detecting bad outliers.
    """
    metrics = [m for m in METRIC_COLS if m in df.columns]
    sub     = df[['version'] + metrics].dropna()
    if sub.empty:
        return
    versions = _versions(sub)
    xs       = range(len(metrics))

    fig, ax = plt.subplots(figsize=(12, 6))
    for _, row in sub.iterrows():
        v = row['version']
        ax.plot(xs, [row[m] for m in metrics],
                color=VERSION_PALETTE.get(v, 'grey'), alpha=0.18, linewidth=0.8)

    for v in versions:
        med = sub[sub['version'] == v][metrics].median()
        ax.plot(xs, med.values, color=VERSION_PALETTE[v],
                linewidth=2.8, label=f'{v} median', zorder=5)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metrics], fontsize=9)
    ax.set_ylim(-0.02, 1.06)
    ax.set_ylabel('Score')
    ax.set_title('Parallel Coordinates: All Runs  (bold = version median)',
                 fontsize=12, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    sns.despine(ax=ax)
    plt.tight_layout()
    _save(fig, outdir, '10_parallel_coordinates.png')


# ── 11: pairwise scatter grid ────────────────────────────────────────────────

def plot_pairwise(df: pd.DataFrame, outdir: str):
    """Pair grid for test_balanced_accuracy, unseen_balanced_accuracy, and f1."""
    metrics  = [m for m in
                ['test_balanced_accuracy', 'unseen_test_balanced_accuracy', 'f1']
                if m in df.columns]
    sub = df[['version'] + metrics].dropna()
    if len(sub) < 10:
        return
    versions = _versions(sub)
    palette  = {v: VERSION_PALETTE[v] for v in versions}

    g = sns.PairGrid(sub, vars=metrics, hue='version',
                     hue_order=versions, palette=palette, diag_sharey=False)
    g.map_diag(sns.kdeplot, fill=True, alpha=0.35)
    g.map_offdiag(sns.scatterplot, alpha=0.4, s=20)
    g.add_legend(title='Version')
    g.figure.suptitle('Pairwise Metric Relationships by Version',
                      fontsize=13, fontweight='bold', y=1.02)
    _save(g.figure, outdir, '11_pairwise_metrics.png')


# ── 12: UPS distribution (from top_kmers_summary) ────────────────────────────

def plot_ups(kmers_df: pd.DataFrame, outdir: str):
    """Violin of UPS per version (uses top_kmers_summary data)."""
    if 'UPS' not in kmers_df.columns:
        return
    versions = _versions(kmers_df)
    fig, ax  = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=kmers_df[['version', 'UPS']].dropna(),
                   x='version', y='UPS', order=versions,
                   palette=VERSION_PALETTE, inner='box', cut=0, ax=ax)
    ax.set_title('Unified Performance Score (UPS) by Version',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Version')
    ax.set_ylabel('UPS')
    sns.despine(ax=ax)
    plt.tight_layout()
    _save(fig, outdir, '12_ups_by_version.png')


# ── 13: PFI distribution (from top_kmers_summary) ────────────────────────────

def plot_pfi(kmers_df: pd.DataFrame, outdir: str):
    """Violin of PFI per version."""
    if 'PFI' not in kmers_df.columns:
        return
    versions = _versions(kmers_df)
    fig, ax  = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=kmers_df[['version', 'PFI']].dropna(),
                   x='version', y='PFI', order=versions,
                   palette=VERSION_PALETTE, inner='box', cut=0, ax=ax)
    ax.set_title('PFI Score Distribution by Version',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Version')
    ax.set_ylabel('PFI')
    sns.despine(ax=ax)
    plt.tight_layout()
    _save(fig, outdir, '13_pfi_by_version.png')


# ── 14: UPS vs PFI scatter (top_kmers_summary) ───────────────────────────────

def plot_ups_vs_pfi(kmers_df: pd.DataFrame, outdir: str):
    """Scatter UPS vs PFI, coloured by version."""
    if not {'UPS', 'PFI'}.issubset(kmers_df.columns):
        return
    versions = _versions(kmers_df)
    sub = kmers_df[['version', 'UPS', 'PFI']].dropna()

    fig, ax = plt.subplots(figsize=(7, 6))
    for v in versions:
        s = sub[sub['version'] == v]
        ax.scatter(s['UPS'], s['PFI'], c=VERSION_PALETTE[v],
                   alpha=0.35, s=15, label=f'Version {v}', linewidths=0)
    ax.set_xlabel('UPS')
    ax.set_ylabel('PFI')
    ax.set_title('UPS vs PFI per Kmer  (coloured by version)',
                 fontsize=12, fontweight='bold')
    ax.legend()
    sns.despine(ax=ax)
    plt.tight_layout()
    _save(fig, outdir, '14_ups_vs_pfi_scatter.png')


# ── 15: top annotated genes / products across versions ───────────────────────

def plot_top_genes_across_versions(annot_df: pd.DataFrame, outdir: str):
    """
    Stacked bar of the top-20 annotated genes/products across versions.
    One plot for bacterium, one for phage.
    """
    if annot_df.empty:
        return

    for entity_col, organism_label in [('gene', 'bacterium'), ('product', 'phage')]:
        if entity_col not in annot_df.columns:
            continue

        sub = (annot_df[annot_df['organism'] == organism_label]
               if 'organism' in annot_df.columns else annot_df)
        sub = sub.dropna(subset=[entity_col])
        if sub.empty:
            continue

        counts = (sub.groupby([entity_col, 'version'])
                     .size().unstack(fill_value=0))
        versions = [v for v in VERSION_ORDER if v in counts.columns]
        totals   = counts[versions].sum(axis=1).sort_values(ascending=False).head(20)
        counts   = counts.loc[totals.index, versions]

        fig, ax = plt.subplots(figsize=(12, 6))
        bottoms = np.zeros(len(counts))
        for v in versions:
            ax.bar(range(len(counts)), counts[v].values,
                   bottom=bottoms, label=f'Version {v}',
                   color=VERSION_PALETTE[v], edgecolor='white', linewidth=0.4)
            bottoms += counts[v].values

        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=90, ha='right', fontsize=8)
        ax.set_ylabel('Total Kmer Count (across all partitions)')
        ax.set_title(
            f'Top 20 Annotated {entity_col.capitalize()}s Across Versions '
            f'({organism_label.capitalize()})',
            fontsize=12, fontweight='bold')
        ax.legend(title='Version')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        sns.despine(ax=ax)
        plt.tight_layout()
        _save(fig, outdir, f'15_top_{entity_col}s_across_versions.png')


# ── 16: kmer occurrence per partition across versions ─────────────────────────

def plot_kmer_occurrence_per_partition_all_versions(kmers_df: pd.DataFrame, outdir: str):
    """
    For each organism, grouped bar showing kmer count per partition,
    with one bar group per version. Partitions sorted by (b, p).
    """
    def bp_key(f):
        b = re.search(r'b(\d+)', str(f))
        p = re.search(r'p(\d+)', str(f))
        return (int(b.group(1)) if b else 0, int(p.group(1)) if p else 0)

    def short_label(f):
        return re.sub(r'^cluster_', '', re.sub(r'_run\d+$', '', str(f)))

    for kmer_col, organism in [('bact_decoded_kmer', 'Bacterium'),
                               ('phage_decoded_kmer', 'Phage')]:
        if kmer_col not in kmers_df.columns:
            continue

        versions        = _versions(kmers_df)
        all_folders     = kmers_df['folder'].unique()
        sorted_folders  = sorted(all_folders, key=bp_key)
        labels          = [short_label(f) for f in sorted_folders]

        # Count kmer rows per (version, folder)
        counts = (kmers_df.groupby(['version', 'folder'])[kmer_col]
                           .count().unstack('folder', fill_value=0)
                           .reindex(columns=sorted_folders, fill_value=0))
        counts = counts.reindex([v for v in versions if v in counts.index])

        n_v    = len(counts)
        width  = 0.8 / max(n_v, 1)
        offsets = np.linspace(-(n_v-1)/2, (n_v-1)/2, n_v) * width

        fig, ax = plt.subplots(figsize=(max(14, len(sorted_folders) * 0.55), 5))
        for (v, row), offset in zip(counts.iterrows(), offsets):
            ax.bar(np.arange(len(sorted_folders)) + offset, row.values,
                   width=width * 0.9, label=f'Version {v}',
                   color=VERSION_PALETTE.get(v, 'grey'), edgecolor='white', linewidth=0.3)

        ax.set_xticks(range(len(sorted_folders)))
        ax.set_xticklabels(labels, rotation=90, ha='right', fontsize=7)
        ax.set_title(f'{organism} — Kmer Occurrences per Partition × Version',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Partition  (b = bacterial cluster, p = phage cluster)')
        ax.set_ylabel('Kmer Count')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(title='Version')
        sns.despine(ax=ax)
        plt.tight_layout()
        _save(fig, outdir, f'16_kmer_occurrence_per_partition_{organism.lower()}.png')


# ── 17: metric improvement delta heatmap (v1→v3) ─────────────────────────────

def plot_improvement_delta(df: pd.DataFrame, outdir: str):
    """
    Heatmap of (mean_v3 - mean_v1) per metric per sketch_type.
    Green = improved, red = regressed. Only drawn when both v1 and v3 exist.
    """
    versions = _versions(df)
    if 'v1' not in versions or len(versions) < 2:
        return

    latest  = versions[-1]   # compare oldest vs newest present
    metrics = [m for m in METRIC_COLS if m in df.columns]

    rows = []
    for st in sorted(df['sketch_type'].unique()):
        sub  = df[df['sketch_type'] == st]
        row  = {'sketch_type': st}
        for m in metrics:
            mean_old = sub[sub['version'] == 'v1'][m].mean()
            mean_new = sub[sub['version'] == latest][m].mean()
            row[m]   = mean_new - mean_old
        rows.append(row)

    delta = pd.DataFrame(rows).set_index('sketch_type')[metrics]
    max_abs = max(delta.abs().max().max(), 0.01)

    fig, ax = plt.subplots(figsize=(max(5, len(metrics)), max(3, len(rows))))
    sns.heatmap(delta, annot=True, fmt='+.3f', cmap='RdYlGn',
                center=0, vmin=-max_abs, vmax=max_abs,
                linewidths=0.5, ax=ax,
                cbar_kws={'label': f'Δ mean  ({latest} − v1)'})
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metrics],
                       rotation=30, ha='right')
    ax.set_title(f'Metric Improvement from v1 → {latest}  per Sketch Type',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    _save(fig, outdir, f'17_improvement_delta_v1_to_{latest}.png')


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def write_summary_csv(df: pd.DataFrame, outdir: str):
    """Save a grouped mean/std summary CSV for all numeric metrics."""
    metrics = [m for m in METRIC_COLS if m in df.columns]
    agg     = (df.groupby(['sketch_type', 'version', 'harsh'])[metrics]
                 .agg(['mean', 'std'])
                 .round(4))
    path = os.path.join(outdir, 'summary_table.csv')
    agg.to_csv(path)
    print(f'  Saved: summary_table.csv')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args    = parse_args()
    data_dir = args.data_dir
    out_dir  = args.out_dir or os.path.join(data_dir, 'summary')
    os.makedirs(out_dir, exist_ok=True)

    print(f'[{datetime.now():%H:%M:%S}]  Scanning: {data_dir}')
    metrics_df, kmers_df, annot_df = load_all_results(data_dir)

    if metrics_df.empty:
        print('No all_runs_summary.csv files found — nothing to plot.')
        return

    # ── Numeric cast ──────────────────────────────────────────────────────────
    cm_cols = ['TN', 'FN', 'FP', 'TP']
    metrics_df = _numeric_cast(metrics_df, METRIC_COLS + cm_cols)
    if not kmers_df.empty:
        kmers_df = _numeric_cast(kmers_df, ['UPS', 'PFI', 'WPFI'])

    # ── Report what was found ─────────────────────────────────────────────────
    n_folders = metrics_df['folder_name'].nunique()
    print(f'  {len(metrics_df)} metric rows from {n_folders} result folders')
    print(f'  Versions    : {sorted(metrics_df["version"].unique())}')
    print(f'  Sketch types: {sorted(metrics_df["sketch_type"].unique())}')
    print(f'  Harsh runs  : {metrics_df["harsh"].sum()} / {len(metrics_df)}')
    if not annot_df.empty:
        print(f'  Annotation rows (encoded): {len(annot_df)}')

    # ── Split non-harsh for most plots ────────────────────────────────────────
    df_main = metrics_df[~metrics_df['harsh']].copy()
    df_all  = metrics_df.copy()

    print(f'\n[{datetime.now():%H:%M:%S}]  Generating plots...')

    # Basic counts & overview
    plot_run_count(df_all,  out_dir)
    write_summary_csv(df_all, out_dir)

    # Metric distributions
    plot_metric_boxes(df_main,          out_dir)
    plot_metric_by_sketchtype(df_main,  out_dir)
    plot_metric_heatmaps(df_main,       out_dir)
    plot_version_progression(df_main,   out_dir)

    # Generalisation & filtering
    plot_test_vs_unseen(df_main,  out_dir)
    plot_harsh_vs_normal(df_all,  out_dir)
    plot_pass_rate(df_all,        out_dir)

    # Confusion matrix & aggregate views
    plot_confusion_matrices(df_main,      out_dir)
    plot_radar(df_main,                   out_dir)
    plot_parallel_coordinates(df_main,    out_dir)
    plot_pairwise(df_main,                out_dir)
    plot_improvement_delta(df_main,       out_dir)

    # Kmer-level (top_kmers_summary)
    if not kmers_df.empty:
        df_kmers_main = kmers_df[~kmers_df['harsh']] if 'harsh' in kmers_df.columns \
                        else kmers_df
        plot_ups(df_kmers_main,                              out_dir)
        plot_pfi(df_kmers_main,                              out_dir)
        plot_ups_vs_pfi(df_kmers_main,                       out_dir)
        plot_kmer_occurrence_per_partition_all_versions(df_kmers_main, out_dir)

    # Gene annotation (encoded only)
    if not annot_df.empty:
        plot_top_genes_across_versions(annot_df, out_dir)

    print(f'\n[{datetime.now():%H:%M:%S}]  Done.  {out_dir}')


if __name__ == '__main__':
    main()