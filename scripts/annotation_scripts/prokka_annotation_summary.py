"""
prokka_annotation_summary.py
-----------------------------
Summarise and compare Prokka annotation results across multiple genomes.

Usage
-----
# Analyse specific directories:
python prokka_annotation_summary.py \
    path/to/Genome_A_prokka_output \
    path/to/Genome_B_prokka_output

# Or use glob patterns:
python prokka_annotation_summary.py /data/prokka_results/*/

Outputs (written to --outdir, default: current directory)
---------
  prokka_summary.csv              Per-genome statistics table
  prokka_coverage.png             Genome coverage bar chart (known / unknown / non-coding)
  prokka_annotation_quality.png   CDS annotation quality stacked bar (COG+EC / COG / named / hypothetical)
  prokka_features.png             Non-CDS feature counts (tRNA, tmRNA, repeat_region)
"""

import argparse
import glob
import re
import sys
from pathlib import Path

import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


# ---------------------------------------------------------------------------
# Layout / style constants
# ---------------------------------------------------------------------------
MAX_PER_PANEL  = 8     # max items (genomes) per subplot panel before splitting
PANEL_W        = 6.0   # width  of each panel in inches  → approx square
PANEL_H        = 5.5   # height of each panel in inches

FONTSIZE_TITLE  = 15
FONTSIZE_AXIS   = 13
FONTSIZE_TICK   = 12
FONTSIZE_LEGEND = 10
FONTSIZE_LABEL  = 8    # data labels inside / above bars

# Suffixes stripped from genome/phage names in all plot tick labels
_STRIP_SUFFIXES = ("_reoriented_merged", "_reoriented", "_merged")


def _clean_name(name: str) -> str:
    """Remove common technical suffixes from display names."""
    for suffix in _STRIP_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _chunks(lst, n):
    """Yield successive sublists of at most *n* items."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _panel_path(base: Path, i: int, total: int) -> Path:
    """Return the output path for panel i. Appends _1, _2, … when total > 1."""
    if total == 1:
        return base
    return base.parent / f"{base.stem}_{i + 1}{base.suffix}"


# ---------------------------------------------------------------------------
# Annotation quality tiers (applied to CDS rows, in priority order)
# A CDS is assigned the highest tier it qualifies for.
# ---------------------------------------------------------------------------
QUALITY_TIERS = [
    "named + COG + EC",   # product named, has COG, has EC
    "named + COG",        # product named, has COG, no EC
    "named only",         # product named, no COG, no EC
    "hypothetical",       # product == "hypothetical protein"
]

QUALITY_COLORS = {
    "named + COG + EC": "#1D9E75",
    "named + COG":      "#5DCAA5",
    "named only":       "#9FE1CB",
    "hypothetical":     "#B4B2A9",
}

COVERAGE_COLORS = {
    "Known function":   "#1D9E75",
    "Unknown function": "#B4B2A9",
    "Non-coding":       "#E8E6DF",
}

FEATURE_COLORS = {
    "tRNA":          "#7F77DD",
    "tmRNA":         "#D4537E",
    "repeat_region": "#EF9F27",
    "rRNA":          "#378ADD",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_txt_stats(txt_path: Path) -> dict:
    """
    Parse the Prokka .txt summary file.
    Returns a dict with keys like 'bases', 'CDS', 'tRNA', etc.
    """
    stats = {}
    with txt_path.open() as fh:
        for line in fh:
            line = line.strip()
            if ": " in line:
                key, _, val = line.partition(": ")
                try:
                    stats[key.strip()] = int(val.strip())
                except ValueError:
                    stats[key.strip()] = val.strip()
    return stats


def find_prokka_files(prokka_dir: Path):
    """
    Locate the .tsv and .txt files inside a Prokka output directory.
    Returns (tsv_path, txt_path, genome_name).
    """
    tsvs = list(prokka_dir.glob("*.tsv"))
    txts = list(prokka_dir.glob("*.txt"))

    if not tsvs:
        raise FileNotFoundError(f"No *.tsv found in {prokka_dir}")
    if not txts:
        raise FileNotFoundError(f"No *.txt found in {prokka_dir}")

    tsv = tsvs[0]
    txt = txts[0]
    name = tsv.stem           # filename without extension
    return tsv, txt, name


def assign_quality_tier(row) -> str:
    """Return the annotation quality tier for a single CDS row."""
    if row["product"] == "hypothetical protein":
        return "hypothetical"
    has_cog = pd.notna(row["COG"]) and str(row["COG"]).strip() != ""
    has_ec  = pd.notna(row["EC_number"]) and str(row["EC_number"]).strip() != ""
    if has_cog and has_ec:
        return "named + COG + EC"
    if has_cog:
        return "named + COG"
    return "named only"


def analyse_genome(prokka_dir: Path) -> dict:
    """
    Parse one Prokka output directory and return a statistics dictionary.

    Keys
    ----
    name                : str   genome display name
    genome_len          : int   genome length in bp (from .txt 'bases')
    total_cds           : int   number of CDS features
    known_cds           : int   CDS with named product
    unknown_cds         : int   CDS with 'hypothetical protein'
    pct_known_cds       : float
    pct_unknown_cds     : float
    known_bp            : int   bp covered by named-product CDS
    unknown_bp          : int   bp covered by hypothetical CDS
    noncoding_bp        : int   genome bp not covered by any CDS
    pct_known_bp        : float fraction of genome with named annotation
    pct_unknown_bp      : float
    pct_noncoding_bp    : float
    quality_cds         : dict  {tier: CDS count}
    quality_bp          : dict  {tier: bp}
    feature_counts      : dict  {ftype: count} for non-CDS features
    has_cog_cds         : int   CDS with a COG assignment
    has_ec_cds          : int   CDS with an EC number
    has_gene_cds        : int   CDS with a gene name
    txt_stats           : dict  raw .txt summary fields
    """
    prokka_dir = Path(prokka_dir)
    tsv, txt, name = find_prokka_files(prokka_dir)
    txt_stats  = parse_txt_stats(txt)
    genome_len = txt_stats.get("bases", 0)

    df = pd.read_csv(tsv, sep="\t")

    # Split by feature type
    cds_df  = df[df["ftype"] == "CDS"].copy()
    other_df = df[df["ftype"] != "CDS"]

    total_cds    = len(cds_df)
    known_mask   = cds_df["product"] != "hypothetical protein"
    known_cds    = int(known_mask.sum())
    unknown_cds  = int((~known_mask).sum())

    known_bp     = int(cds_df.loc[known_mask,  "length_bp"].sum())
    unknown_bp   = int(cds_df.loc[~known_mask, "length_bp"].sum())
    total_cds_bp = int(cds_df["length_bp"].sum())
    noncoding_bp = max(0, genome_len - total_cds_bp)

    # Annotation quality tiers
    cds_df["quality_tier"] = cds_df.apply(assign_quality_tier, axis=1)
    quality_cds = cds_df.groupby("quality_tier")["length_bp"].agg(
        count="count", bp="sum"
    )
    quality_cds_dict = {tier: int(quality_cds.loc[tier, "count"])
                        if tier in quality_cds.index else 0
                        for tier in QUALITY_TIERS}
    quality_bp_dict  = {tier: int(quality_cds.loc[tier, "bp"])
                        if tier in quality_cds.index else 0
                        for tier in QUALITY_TIERS}

    # Non-CDS feature counts
    feature_counts = other_df["ftype"].value_counts().to_dict()

    return {
        "name":             name,
        "genome_len":       genome_len,
        "total_cds":        total_cds,
        "known_cds":        known_cds,
        "unknown_cds":      unknown_cds,
        "pct_known_cds":    round(known_cds  / total_cds * 100, 1) if total_cds else 0,
        "pct_unknown_cds":  round(unknown_cds / total_cds * 100, 1) if total_cds else 0,
        "known_bp":         known_bp,
        "unknown_bp":       unknown_bp,
        "noncoding_bp":     noncoding_bp,
        "pct_known_bp":     round(known_bp     / genome_len * 100, 1) if genome_len else 0,
        "pct_unknown_bp":   round(unknown_bp   / genome_len * 100, 1) if genome_len else 0,
        "pct_noncoding_bp": round(noncoding_bp / genome_len * 100, 1) if genome_len else 0,
        "quality_cds":      quality_cds_dict,
        "quality_bp":       quality_bp_dict,
        "feature_counts":   feature_counts,
        "has_cog_cds":      int(cds_df["COG"].notna().sum()),
        "has_ec_cds":       int(cds_df["EC_number"].notna().sum()),
        "has_gene_cds":     int(cds_df["gene"].notna().sum()),
        "txt_stats":        txt_stats,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_genome_coverage(stats_list: list, outpath: Path):
    """
    Stacked horizontal bar chart: genome coverage by annotation status.
    One file per chunk; x-axis fixed at 0–100 % for comparability across files.
    """
    chunks = list(_chunks(stats_list, MAX_PER_PANEL))

    legend_handles = [
        mpatches.Patch(color=COVERAGE_COLORS["Known function"],   label="Known function"),
        mpatches.Patch(color=COVERAGE_COLORS["Unknown function"], label="Hypothetical / unknown"),
        mpatches.Patch(color=COVERAGE_COLORS["Non-coding"],       label="Non-coding",
                       edgecolor="#cccccc", linewidth=0.5),
    ]

    for i, chunk in enumerate(chunks):
        fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))

        names         = [s["name"]            for s in chunk]
        known_pct     = [s["pct_known_bp"]     for s in chunk]
        unknown_pct   = [s["pct_unknown_bp"]   for s in chunk]
        noncoding_pct = [s["pct_noncoding_bp"] for s in chunk]

        bar_h = 0.55
        y = range(len(names))

        ax.barh(y, known_pct, height=bar_h,
                color=COVERAGE_COLORS["Known function"])
        ax.barh(y, unknown_pct, height=bar_h, left=known_pct,
                color=COVERAGE_COLORS["Unknown function"])
        ax.barh(y, noncoding_pct, height=bar_h,
                left=[k + u for k, u in zip(known_pct, unknown_pct)],
                color=COVERAGE_COLORS["Non-coding"],
                edgecolor="#cccccc", linewidth=0.5)

        for j, (k, u, nc) in enumerate(zip(known_pct, unknown_pct, noncoding_pct)):
            if k > 8:
                ax.text(k / 2, j, f"{k:.1f}%", va="center", ha="center",
                        fontsize=FONTSIZE_LABEL, color="white", fontweight="bold")
            if u > 8:
                ax.text(k + u / 2, j, f"{u:.1f}%", va="center", ha="center",
                        fontsize=FONTSIZE_LABEL, color="white", fontweight="bold")
            if nc > 8:
                ax.text(k + u + nc / 2, j, f"{nc:.1f}%", va="center", ha="center",
                        fontsize=FONTSIZE_LABEL, color="#555555", fontweight="bold")

        ax.set_yticks(list(y))
        ax.set_yticklabels([_clean_name(n) for n in names], fontsize=FONTSIZE_TICK)
        ax.set_xlabel("% of genome", fontsize=FONTSIZE_AXIS)
        ax.set_xlim(0, 100)          # fixed — same on every file
        ax.set_title("Genome coverage by annotation status", fontsize=FONTSIZE_TITLE, pad=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        fig.legend(handles=legend_handles,
                   loc="lower center", bbox_to_anchor=(0.5, 0),
                   ncol=len(legend_handles), fontsize=FONTSIZE_LEGEND,
                   framealpha=0.9, edgecolor="#cccccc")

        plt.tight_layout(rect=[0, 0.08, 1, 1])
        out = _panel_path(outpath, i, len(chunks))
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


def plot_annotation_quality(stats_list: list, outpath: Path):
    """
    Stacked bar chart: CDS annotation quality tiers per genome.
    One file per chunk; y-axis fixed to global max for comparability across files.
    """
    legend_handles = [
        mpatches.Patch(color=QUALITY_COLORS[t], label=t) for t in QUALITY_TIERS
    ]

    global_ymax    = max(s["total_cds"] for s in stats_list) * 1.12
    chunks = list(_chunks(stats_list, MAX_PER_PANEL))
    bar_w  = 0.55

    for i, chunk in enumerate(chunks):
        fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))

        x       = range(len(chunk))
        bottoms = [0] * len(chunk)

        for tier in QUALITY_TIERS:
            values = [s["quality_cds"].get(tier, 0) for s in chunk]
            color  = QUALITY_COLORS[tier]
            ax.bar(x, values, bar_w, bottom=bottoms, color=color)
            for j, (v, b) in enumerate(zip(values, bottoms)):
                if v > global_ymax * 0.03:
                    ax.text(j, b + v / 2, str(v), va="center", ha="center",
                            fontsize=FONTSIZE_LABEL,
                            color="white" if tier != "named only" else "#555",
                            fontweight="bold")
            bottoms = [b + v for b, v in zip(bottoms, values)]

        ax.set_xticks(list(x))
        ax.set_xticklabels([_clean_name(s["name"]) for s in chunk],
                           fontsize=FONTSIZE_TICK, rotation=30, ha="right")
        ax.set_ylabel("Number of CDS", fontsize=FONTSIZE_AXIS)
        ax.set_ylim(0, global_ymax)   # fixed — same on every file
        ax.set_title("CDS annotation quality", fontsize=FONTSIZE_TITLE, pad=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        fig.legend(handles=legend_handles,
                   loc="lower center", bbox_to_anchor=(0.5, 0),
                   ncol=len(legend_handles), fontsize=FONTSIZE_LEGEND,
                   framealpha=0.9, edgecolor="#cccccc",
                   title="Tier (best → worst)", title_fontsize=FONTSIZE_LEGEND)

        plt.tight_layout(rect=[0, 0.1, 1, 1])
        out = _panel_path(outpath, i, len(chunks))
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


def plot_feature_counts(stats_list: list, outpath: Path):
    """
    Grouped bar chart of non-CDS feature counts (tRNA, tmRNA, rRNA, repeat_region).
    One file per chunk; y-axis fixed to global max for comparability across files.
    """
    all_ftypes = []
    for s in stats_list:
        for ft in s["feature_counts"]:
            if ft not in ("gene",) and ft not in all_ftypes:
                all_ftypes.append(ft)

    if not all_ftypes:
        print("  No non-CDS features to plot — skipping feature chart.")
        return

    legend_handles = [
        mpatches.Patch(color=FEATURE_COLORS.get(ft, "#D3D1C7"), label=ft)
        for ft in all_ftypes
    ]

    n_ft       = len(all_ftypes)
    bar_w      = 0.7 / n_ft
    global_max = max(
        (s["feature_counts"].get(ft, 0) for s in stats_list for ft in all_ftypes),
        default=1,
    )
    global_ymax = global_max * 1.15

    chunks = list(_chunks(stats_list, MAX_PER_PANEL))

    for i, chunk in enumerate(chunks):
        fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
        x = range(len(chunk))

        for fi, ftype in enumerate(all_ftypes):
            offset = (fi - n_ft / 2 + 0.5) * bar_w
            values = [s["feature_counts"].get(ftype, 0) for s in chunk]
            color  = FEATURE_COLORS.get(ftype, "#D3D1C7")
            bars   = ax.bar([xi + offset for xi in x], values, bar_w, color=color)
            for bar, v in zip(bars, values):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + global_max * 0.01,
                            str(v), ha="center", va="bottom",
                            fontsize=FONTSIZE_LABEL)

        ax.set_xticks(list(x))
        ax.set_xticklabels([_clean_name(s["name"]) for s in chunk],
                           fontsize=FONTSIZE_TICK, rotation=30, ha="right")
        ax.set_ylabel("Count", fontsize=FONTSIZE_AXIS)
        ax.set_ylim(0, global_ymax)   # fixed — same on every file
        ax.set_title("Non-CDS genomic features", fontsize=FONTSIZE_TITLE, pad=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        fig.legend(handles=legend_handles,
                   loc="lower center", bbox_to_anchor=(0.5, 0),
                   ncol=len(legend_handles), fontsize=FONTSIZE_LEGEND,
                   framealpha=0.9, edgecolor="#cccccc")

        plt.tight_layout(rect=[0, 0.08, 1, 1])
        out = _panel_path(outpath, i, len(chunks))
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def write_summary_csv(stats_list: list, outpath: Path):
    rows = []
    for s in stats_list:
        rows.append({
            "genome":                   s["name"],
            "genome_length_bp":         s["genome_len"],
            "total_cds":                s["total_cds"],
            "known_function_cds":       s["known_cds"],
            "hypothetical_cds":         s["unknown_cds"],
            "pct_known_cds":            s["pct_known_cds"],
            "pct_hypothetical_cds":     s["pct_unknown_cds"],
            "known_function_bp":        s["known_bp"],
            "hypothetical_bp":          s["unknown_bp"],
            "noncoding_bp":             s["noncoding_bp"],
            "pct_genome_known":         s["pct_known_bp"],
            "pct_genome_hypothetical":  s["pct_unknown_bp"],
            "pct_genome_noncoding":     s["pct_noncoding_bp"],
            "cds_with_cog":             s["has_cog_cds"],
            "cds_with_ec":              s["has_ec_cds"],
            "cds_with_gene_name":       s["has_gene_cds"],
            "tier_named_cog_ec":        s["quality_cds"].get("named + COG + EC", 0),
            "tier_named_cog":           s["quality_cds"].get("named + COG", 0),
            "tier_named_only":          s["quality_cds"].get("named only", 0),
            "tier_hypothetical":        s["quality_cds"].get("hypothetical", 0),
            **{f"feature_{k}": v
               for k, v in s["feature_counts"].items()
               if k != "gene"},
        })
    df = pd.DataFrame(rows)
    df.to_csv(outpath, index=False)
    print(f"  Saved: {outpath}")
    return df


# ---------------------------------------------------------------------------
# Dataset classification
# ---------------------------------------------------------------------------

def classify_bact_dataset(prokka_dir: Path) -> str:
    """
    Return 'dataset2' if the directory name starts with 'Kp_', else 'dataset1'.
    Dataset 2 bacteria (data2_klebbacts/) are named Kp_KU1, Kp_KU2, etc.
    """
    return "dataset2" if prokka_dir.name.startswith("Kp_") else "dataset1"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_outputs(stats_list: list, label: str, outdir: Path, fmt: str, no_plots: bool):
    """Write CSV and plots for one dataset group."""
    if not stats_list:
        return
    print(f"\n  [{label}] {len(stats_list)} genome(s)")
    write_summary_csv(stats_list, outdir / f"prokka_summary_{label}.csv")
    if not no_plots:
        plot_genome_coverage(   stats_list, outdir / f"prokka_coverage_{label}.{fmt}")
        plot_annotation_quality(stats_list, outdir / f"prokka_annotation_quality_{label}.{fmt}")
        plot_feature_counts(    stats_list, outdir / f"prokka_features_{label}.{fmt}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Prokka annotation results across multiple genomes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "prokka_dirs", nargs="+", metavar="PROKKA_DIR",
        help="One or more Prokka output directories (glob patterns accepted).",
    )
    parser.add_argument(
        "--outdir", default=".", metavar="DIR",
        help="Directory to write output files (default: current directory).",
    )
    parser.add_argument(
        "--fmt", default="png", choices=["png", "pdf", "svg"],
        help="Output figure format (default: png).",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip figure generation; only write the summary CSV.",
    )
    args = parser.parse_args()

    # Expand glob patterns
    dirs = []
    for pattern in args.prokka_dirs:
        expanded = glob.glob(pattern)
        if expanded:
            dirs.extend(expanded)
        else:
            dirs.append(pattern)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Analysing {len(dirs)} Prokka output director{'y' if len(dirs)==1 else 'ies'}...")

    groups: dict[str, list] = {"dataset1": [], "dataset2": []}

    for d in sorted(dirs):
        d = Path(d)
        try:
            s = analyse_genome(d)
            dataset = classify_bact_dataset(d)
            s["dataset"] = dataset
            groups[dataset].append(s)
            print(f"  [{dataset}] {s['name']}: genome {s['genome_len']:,} bp, "
                  f"{s['total_cds']} CDS, {s['pct_known_cds']}% named")
        except Exception as exc:
            print(f"  WARNING: skipping {d}: {exc}", file=sys.stderr)

    if not any(groups.values()):
        sys.exit("No valid Prokka directories found.")

    print("\nWriting outputs...")
    for label, stats_list in groups.items():
        run_outputs(stats_list, label, outdir, args.fmt, args.no_plots)

    print("\nDone.")


if __name__ == "__main__":
    main()