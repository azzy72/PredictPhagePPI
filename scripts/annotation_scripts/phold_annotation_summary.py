"""
phold_annotation_summary.py
----------------------------
Summarise and compare phold annotation results across multiple phages.

Usage
-----
# Analyse specific directories:
python phold_annotation_summary.py \
    path/to/Phage_A_phold_output \
    path/to/Phage_B_phold_output

# Or use glob patterns:
python phold_annotation_summary.py /data/phold_results/*/

Outputs (written to --outdir, default: current directory)
---------
  phold_summary.csv           Per-phage statistics table
  phold_coverage.png          Genome coverage bar chart (known / unknown / non-coding)
  phold_functions.png         Stacked bar chart of functional categories per phage
  phold_annotation_method.png Annotation method breakdown per phage
"""

import argparse
import glob
import re
import sys
from pathlib import Path

import math
import textwrap

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


# ---------------------------------------------------------------------------
# Layout / style constants
# ---------------------------------------------------------------------------
MAX_PER_PANEL  = 8     # max items (phages) per subplot panel before splitting
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
# Functional category colours (consistent across all plots)
# ---------------------------------------------------------------------------
CATEGORY_COLORS = {
    "head and packaging":                           "#378ADD",
    "tail":                                         "#7F77DD",
    "DNA, RNA and nucleotide metabolism":           "#1D9E75",
    "lysis":                                        "#E24B4A",
    "moron, auxiliary metabolic gene and host takeover": "#D4537E",
    "connector":                                    "#5DCAA5",
    "transcription regulation":                     "#EF9F27",
    "integration and excision":                     "#D85A30",
    "other":                                        "#BA7517",
    "unknown function":                             "#B4B2A9",
}

COVERAGE_COLORS = {
    "Known function":    "#1D9E75",
    "Unknown function":  "#B4B2A9",
    "Non-coding":        "#E8E6DF",
}

METHOD_COLORS = {
    "foldseek": "#378ADD",
    "pharokka":  "#7F77DD",
    "none":      "#B4B2A9",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_genome_length(gbk_path: Path) -> int:
    """Extract genome length from the LOCUS line of a GenBank file."""
    with gbk_path.open() as fh:
        for line in fh:
            if line.startswith("LOCUS"):
                match = re.search(r"(\d+)\s+bp", line)
                if match:
                    return int(match.group(1))
    raise ValueError(f"Could not parse genome length from {gbk_path}")


def find_phold_files(phold_dir: Path):
    """
    Locate the per_cds_predictions.tsv and .gbk inside a phold output directory.
    Returns (tsv_path, gbk_path, phage_name).
    """
    tsvs = list(phold_dir.glob("*_per_cds_predictions.tsv"))
    gbks  = list(phold_dir.glob("*.gbk"))

    if not tsvs:
        raise FileNotFoundError(f"No *_per_cds_predictions.tsv found in {phold_dir}")
    if not gbks:
        raise FileNotFoundError(f"No *.gbk found in {phold_dir}")

    tsv = tsvs[0]
    gbk = gbks[0]
    # Derive a clean phage name from the TSV filename
    name = tsv.stem.replace("_per_cds_predictions", "")
    return tsv, gbk, name


def analyse_phage(phold_dir: Path) -> dict:
    """
    Parse one phold output directory and return a statistics dictionary.
    Keys
    ----
    name             : str   phage display name
    genome_len       : int   genome length in bp
    total_cds        : int
    known_cds        : int   CDS with a non-unknown function
    unknown_cds      : int
    pct_known_cds    : float percentage of CDS with known function
    pct_unknown_cds  : float
    known_bp         : int   bp covered by known-function CDS
    unknown_bp       : int   bp covered by unknown-function CDS
    noncoding_bp     : int   genome bp not covered by any CDS
    pct_known_bp     : float fraction of genome with known-function annotation
    pct_unknown_bp   : float
    pct_noncoding_bp : float
    category_cds     : dict  {category: CDS count}
    category_bp      : dict  {category: bp}
    method_counts    : dict  {method: CDS count}
    """
    phold_dir = Path(phold_dir)
    tsv, gbk, name = find_phold_files(phold_dir)
    genome_len = parse_genome_length(gbk)

    df = pd.read_csv(tsv, sep="\t")

    # Gene length (always positive regardless of strand)
    df["length_bp"] = (df["end"] - df["start"]).abs() + 1

    total_cds   = len(df)
    unknown_mask = df["function"] == "unknown function"
    known_cds   = int((~unknown_mask).sum())
    unknown_cds = int(unknown_mask.sum())

    known_bp    = int(df.loc[~unknown_mask, "length_bp"].sum())
    unknown_bp  = int(df.loc[unknown_mask,  "length_bp"].sum())
    total_cds_bp = int(df["length_bp"].sum())
    noncoding_bp = genome_len - total_cds_bp

    # Per-category counts and bp
    category_cds = (
        df.groupby("function")["length_bp"]
          .agg(count="count", bp="sum")
    )
    category_cds_dict = category_cds["count"].to_dict()
    category_bp_dict  = category_cds["bp"].to_dict()

    # Annotation method (normalise "none" string)
    method_counts = (
        df["annotation_method"]
          .fillna("none")
          .value_counts()
          .to_dict()
    )

    return {
        "name":             name,
        "genome_len":       genome_len,
        "total_cds":        total_cds,
        "known_cds":        known_cds,
        "unknown_cds":      unknown_cds,
        "pct_known_cds":    round(known_cds / total_cds * 100, 1),
        "pct_unknown_cds":  round(unknown_cds / total_cds * 100, 1),
        "known_bp":         known_bp,
        "unknown_bp":       unknown_bp,
        "noncoding_bp":     noncoding_bp,
        "pct_known_bp":     round(known_bp / genome_len * 100, 1),
        "pct_unknown_bp":   round(unknown_bp / genome_len * 100, 1),
        "pct_noncoding_bp": round(noncoding_bp / genome_len * 100, 1),
        "category_cds":     category_cds_dict,
        "category_bp":      category_bp_dict,
        "method_counts":    method_counts,
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
        mpatches.Patch(color=COVERAGE_COLORS["Unknown function"], label="Unknown function"),
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


def plot_function_categories(stats_list: list, outpath: Path):
    """
    Stacked bar chart: CDS count per functional category, one bar per phage.
    One file per chunk; y-axis fixed to global max for comparability across files.
    """
    all_cats = []
    for s in stats_list:
        for cat in s["category_cds"]:
            if cat != "unknown function" and cat not in all_cats:
                all_cats.append(cat)
    cat_totals = {
        cat: sum(s["category_cds"].get(cat, 0) for s in stats_list)
        for cat in all_cats
    }
    all_cats.sort(key=lambda c: cat_totals[c], reverse=True)

    # Global y-max: total CDS per phage (all categories including unknown)
    global_ymax = max(
        sum(s["category_cds"].get(cat, 0) for cat in s["category_cds"])
        for s in stats_list
    ) * 1.12

    all_legend_cats = all_cats + ["unknown function"]
    legend_handles = [
        mpatches.Patch(color=CATEGORY_COLORS.get(cat, "#cccccc"),
                       label=textwrap.fill(cat, width=14))
        for cat in all_legend_cats
    ]

    chunks = list(_chunks(stats_list, MAX_PER_PANEL))
    bar_w  = 0.55

    for i, chunk in enumerate(chunks):
        fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))

        x       = range(len(chunk))
        bottoms = [0] * len(chunk)

        for cat in all_cats:
            values = [s["category_cds"].get(cat, 0) for s in chunk]
            color  = CATEGORY_COLORS.get(cat, "#cccccc")
            ax.bar(x, values, bar_w, bottom=bottoms, color=color)
            bottoms = [b + v for b, v in zip(bottoms, values)]

        unknown_vals = [s["category_cds"].get("unknown function", 0) for s in chunk]
        ax.bar(x, unknown_vals, bar_w, bottom=bottoms,
               color=CATEGORY_COLORS["unknown function"])

        ax.set_xticks(list(x))
        ax.set_xticklabels([_clean_name(s["name"]) for s in chunk],
                           fontsize=FONTSIZE_TICK, rotation=30, ha="right")
        ax.set_ylabel("Number of CDS", fontsize=FONTSIZE_AXIS)
        ax.set_ylim(0, global_ymax)   # fixed — same on every file
        ax.set_title("CDS by functional category", fontsize=FONTSIZE_TITLE, pad=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        ax.legend(handles=legend_handles,
                  loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  ncol=1, fontsize=FONTSIZE_LEGEND,
                  framealpha=0.9, edgecolor="#cccccc",
                  borderaxespad=0)

        plt.tight_layout()
        fig.subplots_adjust(right=0.72)   # leave room for the narrow legend
        out = _panel_path(outpath, i, len(chunks))
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


def plot_annotation_method(stats_list: list, outpath: Path):
    """
    Stacked bar chart: annotation method (foldseek / pharokka / none) per phage.
    One file per chunk; y-axis fixed to global max for comparability across files.
    """
    methods = ["foldseek", "pharokka", "none"]
    labels  = {"foldseek": "foldseek", "pharokka": "pharokka", "none": "no hit"}
    legend_handles = [
        mpatches.Patch(color=METHOD_COLORS[m], label=labels[m]) for m in methods
    ]

    # Global y-max: total CDS per phage
    global_ymax = max(s["total_cds"] for s in stats_list) * 1.12

    chunks = list(_chunks(stats_list, MAX_PER_PANEL))
    bar_w  = 0.5

    for i, chunk in enumerate(chunks):
        fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))

        x       = range(len(chunk))
        bottoms = [0] * len(chunk)

        for method in methods:
            values = [s["method_counts"].get(method, 0) for s in chunk]
            color  = METHOD_COLORS.get(method, "#cccccc")
            ax.bar(x, values, bar_w, bottom=bottoms, color=color)
            for j, (v, b) in enumerate(zip(values, bottoms)):
                if v > 0:
                    ax.text(j, b + v / 2, str(v), va="center", ha="center",
                            fontsize=FONTSIZE_LABEL, color="white", fontweight="bold")
            bottoms = [b + v for b, v in zip(bottoms, values)]

        ax.set_xticks(list(x))
        ax.set_xticklabels([_clean_name(s["name"]) for s in chunk],
                           fontsize=FONTSIZE_TICK, rotation=30, ha="right")
        ax.set_ylabel("Number of CDS", fontsize=FONTSIZE_AXIS)
        ax.set_ylim(0, global_ymax)   # fixed — same on every file
        ax.set_title("Annotation method breakdown", fontsize=FONTSIZE_TITLE, pad=10)
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
            "phage":                s["name"],
            "genome_length_bp":     s["genome_len"],
            "total_cds":            s["total_cds"],
            "known_function_cds":   s["known_cds"],
            "unknown_function_cds": s["unknown_cds"],
            "pct_known_cds":        s["pct_known_cds"],
            "pct_unknown_cds":      s["pct_unknown_cds"],
            "known_function_bp":    s["known_bp"],
            "unknown_function_bp":  s["unknown_bp"],
            "noncoding_bp":         s["noncoding_bp"],
            "pct_genome_known":     s["pct_known_bp"],
            "pct_genome_unknown":   s["pct_unknown_bp"],
            "pct_genome_noncoding": s["pct_noncoding_bp"],
            "method_foldseek":      s["method_counts"].get("foldseek", 0),
            "method_pharokka":      s["method_counts"].get("pharokka", 0),
            "method_none":          s["method_counts"].get("none", 0),
        })
    df = pd.DataFrame(rows)
    df.to_csv(outpath, index=False)
    print(f"  Saved: {outpath}")
    return df


# ---------------------------------------------------------------------------
# Dataset classification
# ---------------------------------------------------------------------------

def classify_phage_dataset(phold_dir: Path) -> str:
    """
    Return 'dataset2' if the directory name contains 'Host_', else 'dataset1'.
    Dataset 2 phages (data2_phages/) have names like Anivius_Host_6_phold_output.
    """
    return "dataset2" if "Host_" in phold_dir.name else "dataset1"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_outputs(stats_list: list, label: str, outdir: Path, fmt: str, no_plots: bool):
    """Write CSV and plots for one dataset group."""
    if not stats_list:
        return
    print(f"\n  [{label}] {len(stats_list)} phage(s)")
    write_summary_csv(stats_list, outdir / f"phold_summary_{label}.csv")
    if not no_plots:
        plot_genome_coverage(    stats_list, outdir / f"phold_coverage_{label}.{fmt}")
        plot_function_categories(stats_list, outdir / f"phold_functions_{label}.{fmt}")
        plot_annotation_method(  stats_list, outdir / f"phold_annotation_method_{label}.{fmt}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare phold annotation results across multiple phage genomes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "phold_dirs", nargs="+", metavar="PHOLD_DIR",
        help="One or more phold output directories (glob patterns accepted).",
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
    for pattern in args.phold_dirs:
        expanded = glob.glob(pattern)
        if expanded:
            dirs.extend(expanded)
        else:
            dirs.append(pattern)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Analysing {len(dirs)} phold output director{'y' if len(dirs)==1 else 'ies'}...")

    groups: dict[str, list] = {"dataset1": [], "dataset2": []}

    for d in sorted(dirs):
        d = Path(d)
        try:
            s = analyse_phage(d)
            dataset = classify_phage_dataset(d)
            s["dataset"] = dataset
            groups[dataset].append(s)
            print(f"  [{dataset}] {s['name']}: genome {s['genome_len']:,} bp, "
                  f"{s['total_cds']} CDS, {s['pct_known_cds']}% known function")
        except Exception as exc:
            print(f"  WARNING: skipping {d}: {exc}", file=sys.stderr)

    if not any(groups.values()):
        sys.exit("No valid phold directories found.")

    print("\nWriting outputs...")
    for label, stats_list in groups.items():
        run_outputs(stats_list, label, outdir, args.fmt, args.no_plots)

    print("\nDone.")


if __name__ == "__main__":
    main()