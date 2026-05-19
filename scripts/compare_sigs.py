#!/usr/bin/env python3
"""
compare_sigs.py  –  Sourmash similarity pipeline (bacteria & phage)

Steps
-----
1. sourmash compare      → similarity matrix + labels CSV
2. sourmash plot         → standard heatmap PNG
3. prefix_bact_labels.sh → genus-prefixed labels CSV  (bacteria only)
4. sourmash scripts plot2 → annotated dendrogram PNG

Usage
-----
    python compare_sigs.py [--config config.yaml] [--dry-run]

Requirements
------------
    pip install pyyaml
    sourmash  (with sourmash-scripts for plot2)
"""

import argparse
import itertools
import logging
import subprocess
import sys, os, re
from pathlib import Path
from paths import scripts_path
import yaml
import shutil
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_sig_dir(sketch_dir: Path, organism_prefix: str, n: int, k: int) -> Path:
    """
    Locate the signature directory for a given organism prefix (Bact/Phage)
    and parameter tag, without assuming the middle encoding string.

    Looks for exactly one directory matching:
        <sketch_dir>/<organism_prefix>*_n{n}_k{k}

    Examples of names this will match:
        BactMinhash_n500_k12   (SM_sketches)
        BactEncoded_n500_k12   (encoded_sketches)
        PhageAnything_n500_k12

    Raises RuntimeError if zero or more than one match is found.
    """
    tag = f"_n{n}_k{k}"
    matches = [
        d for d in sketch_dir.iterdir()
        if d.is_dir()
        and d.name.startswith(organism_prefix)
        and d.name.endswith(tag)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        raise RuntimeError(
            f"No directory matching '{organism_prefix}*{tag}' found in {sketch_dir}"
        )
    raise RuntimeError(
        f"Ambiguous match for '{organism_prefix}*{tag}' in {sketch_dir}: "
        + ", ".join(d.name for d in sorted(matches))
    )


def run(cmd: list[str], *, cwd: Path | None = None, dry_run: bool = False) -> None:
    """Log and optionally execute a shell command."""
    display = " ".join(str(c) for c in cmd)
    if cwd:
        log.info("  [cwd=%s]  %s", cwd, display)
    else:
        log.info("  %s", display)
    if not dry_run:
        subprocess.run(cmd, cwd=cwd, check=True)


def already_done(path: Path, dry_run: bool, force: bool) -> bool:
    """Skip a step whose output already exists (unless dry-running)."""
    if not dry_run and path.exists() and not force:
        log.info("  SKIP – already exists: %s", path)
        return True
    return False

def _assign_partitions(df: pd.DataFrame, num_partitions: int) -> pd.DataFrame:
    """
    Assign clusters to partitions such that partitions are balanced in size
    and clusters are not split across partitions.
    
    Clusters are sorted and assigned to partitions greedily, always adding
    the next cluster to the partition with the smallest current size.
    """
    if num_partitions < 1:
        raise ValueError("num_partitions must be >= 1")
    
    if num_partitions == 1:
        # All labels go to partition 0
        df = df.copy()
        df["Partition"] = 0
        return df
    
    # Group by cluster and count size
    cluster_sizes = df.groupby("Cluster").size().reset_index(name="size")
    cluster_sizes = cluster_sizes.sort_values("Cluster").reset_index(drop=True)
    
    # Initialize partitions
    partitions = [[] for _ in range(num_partitions)]
    partition_sizes = [0] * num_partitions
    
    # Assign clusters to partitions, balancing size
    for _, row in cluster_sizes.iterrows():
        cluster_id = int(row["Cluster"])
        size = int(row["size"])
        # Find partition with smallest current size
        min_idx = partition_sizes.index(min(partition_sizes))
        partitions[min_idx].append(cluster_id)
        partition_sizes[min_idx] += size
    
    # Create mapping from cluster to partition
    cluster_to_partition = {}
    for partition_id, clusters in enumerate(partitions):
        for cluster_id in clusters:
            cluster_to_partition[cluster_id] = partition_id
    
    # Assign partition to each row
    df = df.copy()
    df["Partition"] = df["Cluster"].map(cluster_to_partition).astype(int)
    
    return df



# ── Pipeline steps ────────────────────────────────────────────────────────────

def step1_compare(sig_dir: Path, out_mat: Path, out_labels: Path,
                  dry_run: bool, force: bool) -> None:
    """Run sourmash compare to produce a similarity matrix."""
    log.info("── Step 1: sourmash compare  →  %s", out_mat.name)
    if already_done(out_mat, dry_run, force):
        return
    out_mat.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["sourmash", "compare", str(sig_dir), "-o", str(out_mat), "--labels-to", str(out_labels)]
    if force:
        cmd.append("-f")
    run(cmd,cwd=sig_dir.parent, dry_run=dry_run)


def step2_plot_standard(mat: Path, dry_run: bool, force: bool) -> None:
    """Run sourmash plot (writes <mat>.png next to the matrix file)."""
    png = mat.with_suffix(mat.suffix + ".png")
    log.info("── Step 2: sourmash plot     →  %s", png.name)
    if already_done(png, dry_run, force):
        return
    # sourmash plot puts output in cwd, so cd into sim_matrices/
    cmd = ["sourmash", "plot", mat.name]
    if force:
        cmd.append("-f")
    run(cmd, cwd=mat.parent, dry_run=dry_run)


def step3_prefix_labels(labels_to: Path, genus_csv: Path,
                        prefix_script: Path, out_prefixed: Path,
                        dry_run: bool, force: bool) -> None:
    """Prefix bacteria sample labels with their genus names."""
    log.info("── Step 3: prefix labels     →  %s", out_prefixed.name)
    if already_done(out_prefixed, dry_run, force):
        return
    if dry_run:
        log.info("  bash %s %s %s > %s",
                 prefix_script, labels_to, genus_csv, out_prefixed)
        return
    with open(out_prefixed, "w") as fh:
        subprocess.run(
            ["bash", str(prefix_script), str(labels_to), str(genus_csv)],
            stdout=fh,
            check=True,
        )


def step4_dendrogram(mat: Path, labels: Path, out_png: Path,
                     cut_point: float, figsize_x: int, figsize_y: int,
                     dry_run: bool, force: bool) -> None:
    """Run sourmash scripts plot2 to produce an annotated dendrogram."""
    log.info("── Step 4: dendrogram        →  %s", out_png.name)
    if already_done(out_png, dry_run, force):
        return
    cmd = ["sourmash", "scripts", "plot2",
         str(mat), str(labels),
         "-o", str(out_png),
         f"--cut-point={cut_point}",
         "--cluster-out",
         "--figsize-x", str(figsize_x),
         "--figsize-y", str(figsize_y)]
    if force:
        cmd.append("-f")
    run(cmd,cwd=mat.parent,dry_run=dry_run)

def step5_collect_clusters(sim_mat_dir: Path, n: int, k: int, dry_run: bool,
                           num_partitions: int = 1, force: bool = False) -> None:
    """
    Aggregates .mat.[num].csv files and removes individual files after processing.
    Also creates partition files that assign labels to partitions.
    """
    log.info("── Step 5: collect & cleanup clusters in %s", sim_mat_dir.name)
    
    out_bact = sim_mat_dir / f"combined_bact_clusters_n{n}_k{k}.csv"
    out_phage = sim_mat_dir / f"combined_phage_clusters_n{n}_k{k}.csv"
    out_bact_partitions = sim_mat_dir / f"bact_partitions_n{n}_k{k}.csv"
    out_phage_partitions = sim_mat_dir / f"phage_partitions_n{n}_k{k}.csv"

    if not dry_run and out_bact.exists() and out_phage.exists() and not force:
        log.info("  SKIP – summary files already exist.")
        return

    combined_bact = pd.DataFrame()
    combined_phage = pd.DataFrame()
    processed_files = []
    
    bact_clust_count = 0
    phage_clust_count = 0
    n_string = f"n{n}"
    k_string = f"k{k}"
    pattern = re.compile(r'.*\.mat\.\d+\.csv$')

    # Identify and aggregate files[cite: 2]
    for file_name in os.listdir(sim_mat_dir):
        if not pattern.match(file_name):
            continue
        if n_string not in file_name or k_string not in file_name:
            continue

        file_path = sim_mat_dir / file_name
        try:
            sim_mat = pd.read_csv(file_path, index_col=0)
        except Exception as e:
            log.error("  Failed to load %s: %s", file_name, e)
            continue

        if "Phage" in file_name:
            sim_mat = sim_mat[["label"]].copy()
            sim_mat["host"] = sim_mat["label"].str.split("_").str[0]
            sim_mat["label"] = sim_mat["label"].str.split("_").str[-1]
            sim_mat["Cluster"] = phage_clust_count
            sim_mat.set_index("label", inplace=True)
            combined_phage = pd.concat([combined_phage, sim_mat], axis=0)
            phage_clust_count += 1
            processed_files.append(file_path)

        elif "Bact" in file_name:
            sim_mat = sim_mat[["label"]].copy()
            sim_mat["label"] = sim_mat["label"].str.replace("_bp=", "", regex=False)
            sim_mat["label"] = sim_mat["label"].str.lstrip("_")
            sim_mat["Cluster"] = bact_clust_count
            sim_mat.set_index("label", inplace=True)
            combined_bact = pd.concat([combined_bact, sim_mat], axis=0)
            bact_clust_count += 1
            processed_files.append(file_path)

    # Add partition assignments
    combined_bact = _assign_partitions(combined_bact, num_partitions)
    combined_phage = _assign_partitions(combined_phage, num_partitions)

    if dry_run:
        log.info("  [dry-run] Would save aggregated CSVs, partition files, and remove %d individual files", len(processed_files))
        return

    # Save aggregated results
    if not combined_bact.empty:
        combined_bact.to_csv(out_bact)
        log.info("  Saved: %s", out_bact.name)
        # Save partition file
        bact_partitions = combined_bact[["Partition"]].copy()
        bact_partitions.to_csv(out_bact_partitions)
        log.info("  Saved: %s", out_bact_partitions.name)
    if not combined_phage.empty:
        combined_phage.to_csv(out_phage)
        log.info("  Saved: %s", out_phage.name)
        # Save partition file
        phage_partitions = combined_phage[["Partition"]].copy()
        phage_partitions.to_csv(out_phage_partitions)
        log.info("  Saved: %s", out_phage_partitions.name)

    # Cleanup: Delete individual files only if aggregation was successful
    for f in processed_files:
        try:
            f.unlink()
        except Exception as e:
            log.warning("  Failed to delete %s: %s", f.name, e)
    
    if processed_files:
        log.info("  Cleanup complete: removed %d individual cluster files.", len(processed_files))


# ── Per-parameter-combination entry point ────────────────────────────────────

def process_combination(sketch_dir: Path, n: int, k: int, cfg: dict,
                        dry_run: bool, force: bool) -> None:
    sim_dir = sketch_dir / "sim_matrices"
    tag = f"n{n}_k{k}"

    # ── Bacteria ──────────────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("BACTERIA  |  sketch_dir=%s  n=%s  k=%s", sketch_dir.name, n, k)
    log.info("═" * 60)

    bact_dir = find_sig_dir(sketch_dir, "Bact", n, k)
    log.info("  Found sig dir: %s", bact_dir.name)

    if "_data2" in sketch_dir.name:
        bact_mat = sim_dir / f"BactSim_{tag}_data2.mat"
        bact_lbl = sim_dir / f"BactSim_{tag}_data2.mat.labels_to.csv"
        bact_pre = sim_dir / f"BactSim_{tag}_data2.mat.labels_prefixed.csv"
        bact_den = sim_dir / f"BactDendro_{tag}_data2.png"
    else:
        bact_mat = sim_dir / f"BactSim_{tag}.mat"
        bact_lbl = sim_dir / f"BactSim_{tag}.mat.labels_to.csv"
        bact_pre = sim_dir / f"BactSim_{tag}.mat.labels_prefixed.csv"
        bact_den = sim_dir / f"BactDendro_{tag}.png"

    

    step1_compare(bact_dir, bact_mat, bact_lbl, dry_run, force)
    step2_plot_standard(bact_mat, dry_run, force)
    step3_prefix_labels(
        bact_lbl,
        Path(cfg["bact_labels_csv"]),
        Path(cfg["prefix_script"]),
        bact_pre,
        dry_run,
        force,
    )
    step4_dendrogram(
        bact_mat, bact_pre, bact_den,
        cfg["bact_cut_point"], cfg["figsize_x"], cfg["figsize_y"],
        dry_run,
        force,
    )

    # ── Phage ─────────────────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("PHAGE     |  sketch_dir=%s  n=%s  k=%s", sketch_dir.name, n, k)
    log.info("═" * 60)

    phage_dir = find_sig_dir(sketch_dir, "Phage", n, k)
    log.info("  Found sig dir: %s", phage_dir.name)

    if "_data2" in sketch_dir.name:
        phage_mat = sim_dir / f"PhageSim_{tag}_data2.mat"
        phage_lbl = sim_dir / f"PhageSim_{tag}_data2.mat.labels_to.csv"
        phage_den = sim_dir / f"PhageDendro_{tag}_data2.png"
    else:
        phage_mat = sim_dir / f"PhageSim_{tag}.mat"
        phage_lbl = sim_dir / f"PhageSim_{tag}.mat.labels_to.csv"
        phage_den = sim_dir / f"PhageDendro_{tag}.png"

    step1_compare(phage_dir, phage_mat, phage_lbl, dry_run, force)
    step2_plot_standard(phage_mat, dry_run, force)
    # Phage uses the raw labels_to CSV directly (no genus prefixing)
    step4_dendrogram(
        phage_mat, phage_lbl, phage_den,
        cfg["phage_cut_point"], cfg["figsize_x"], cfg["figsize_y"],
        dry_run,
        force,
    )

    # ── Aggregation & Cleanup ─────────────────────────────────────────────────
    step5_collect_clusters(sim_dir, n, k, dry_run, cfg["num_partitions"], force)

# ── Main ──────────────────────────────────────────────────────────────────────

def check_dependencies() -> None:
    """Verify that required external tools are available on PATH."""
    missing = [tool for tool in ("sourmash",) if not shutil.which(tool)]
    if missing:
        log.error("Required tool(s) not found in PATH: %s", ", ".join(missing))
        sys.exit(1)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=f"{scripts_path}config_compare_sigs.yaml",
                        help=f"Path to config YAML (default: {scripts_path}config_compare_sigs.yaml)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")
    parser.add_argument("--force", action="store_true",
                        help="Ignore existing output files and re-write them")
    args = parser.parse_args()
    
    # Requirement: dry-run and force are mutually exclusive
    if args.dry_run and args.force:
        log.error("Arguments --dry-run and --force cannot be used together.")
        sys.exit(1)
    
    check_dependencies()

    config_path = Path(args.config)
    if not config_path.exists():
        log.error("Config file not found: %s", config_path)
        sys.exit(1)

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    # Apply defaults for optional keys
    cfg.setdefault("bact_cut_point",  1.08)
    cfg.setdefault("phage_cut_point", 1.12)
    cfg.setdefault("figsize_x", 20)
    cfg.setdefault("figsize_y", 18)
    cfg.setdefault("num_partitions", 6)

    if args.dry_run:
        log.info("DRY-RUN MODE – no commands will be executed")

    combos = list(itertools.product(
        cfg["sketch_dirs"],
        cfg["n_values"],
        cfg["k_values"],
    ))
    log.info("Processing %d combination(s) across %d sketch dir(s)",
             len(combos), len(cfg["sketch_dirs"]))
    log.info("Force mode: %s", "ON" if args.force else "OFF")

    for sketch_dir_str, n, k in combos:
        process_combination(Path(sketch_dir_str), int(n), int(k), cfg, args.dry_run, args.force)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()