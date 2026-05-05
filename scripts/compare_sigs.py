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
import sys
from pathlib import Path
from paths import scripts_path
import yaml

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


def already_done(path: Path, dry_run: bool) -> bool:
    """Skip a step whose output already exists (unless dry-running)."""
    if not dry_run and path.exists():
        log.info("  SKIP – already exists: %s", path)
        return True
    return False


# ── Pipeline steps ────────────────────────────────────────────────────────────

def step1_compare(sig_dir: Path, out_mat: Path, out_labels: Path,
                  dry_run: bool) -> None:
    """Run sourmash compare to produce a similarity matrix."""
    log.info("── Step 1: sourmash compare  →  %s", out_mat.name)
    if already_done(out_mat, dry_run):
        return
    out_mat.parent.mkdir(parents=True, exist_ok=True)
    run(
        ["sourmash", "compare", str(sig_dir),
         "-o", str(out_mat),
         "--labels-to", str(out_labels)],
        dry_run=dry_run,
    )


def step2_plot_standard(mat: Path, dry_run: bool) -> None:
    """Run sourmash plot (writes <mat>.png next to the matrix file)."""
    png = mat.with_suffix(mat.suffix + ".png")
    log.info("── Step 2: sourmash plot     →  %s", png.name)
    if already_done(png, dry_run):
        return
    # sourmash plot puts output in cwd, so cd into sim_matrices/
    run(["sourmash", "plot", mat.name], cwd=mat.parent, dry_run=dry_run)


def step3_prefix_labels(labels_to: Path, genus_csv: Path,
                        prefix_script: Path, out_prefixed: Path,
                        dry_run: bool) -> None:
    """Prefix bacteria sample labels with their genus names."""
    log.info("── Step 3: prefix labels     →  %s", out_prefixed.name)
    if already_done(out_prefixed, dry_run):
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
                     dry_run: bool) -> None:
    """Run sourmash scripts plot2 to produce an annotated dendrogram."""
    log.info("── Step 4: dendrogram        →  %s", out_png.name)
    if already_done(out_png, dry_run):
        return
    run(
        ["sourmash", "scripts", "plot2",
         str(mat), str(labels),
         "-o", str(out_png),
         f"--cut-point={cut_point}",
         "--cluster-out",
         "--figsize-x", str(figsize_x),
         "--figsize-y", str(figsize_y)],
        dry_run=dry_run,
    )


# ── Per-parameter-combination entry point ────────────────────────────────────

def process_combination(sketch_dir: Path, n: int, k: int, cfg: dict,
                        dry_run: bool) -> None:
    sim_dir = sketch_dir / "sim_matrices"
    tag = f"n{n}_k{k}"

    # ── Bacteria ──────────────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("BACTERIA  |  sketch_dir=%s  n=%s  k=%s", sketch_dir.name, n, k)
    log.info("═" * 60)

    bact_dir = find_sig_dir(sketch_dir, "Bact", n, k)
    log.info("  Found sig dir: %s", bact_dir.name)

    bact_mat = sim_dir / f"BactSim_{tag}.mat"
    bact_lbl = sim_dir / f"BactSim_{tag}.mat.labels_to.csv"
    bact_pre = sim_dir / f"BactSim_{tag}.mat.labels_prefixed.csv"
    bact_den = sim_dir / f"BactDendro_{tag}.png"

    step1_compare(bact_dir, bact_mat, bact_lbl, dry_run)
    step2_plot_standard(bact_mat, dry_run)
    step3_prefix_labels(
        bact_lbl,
        Path(cfg["bact_labels_csv"]),
        Path(cfg["prefix_script"]),
        bact_pre,
        dry_run,
    )
    step4_dendrogram(
        bact_mat, bact_pre, bact_den,
        cfg["bact_cut_point"], cfg["figsize_x"], cfg["figsize_y"],
        dry_run,
    )

    # ── Phage ─────────────────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("PHAGE     |  sketch_dir=%s  n=%s  k=%s", sketch_dir.name, n, k)
    log.info("═" * 60)

    phage_dir = find_sig_dir(sketch_dir, "Phage", n, k)
    log.info("  Found sig dir: %s", phage_dir.name)

    phage_mat = sim_dir / f"PhageSim_{tag}.mat"
    phage_lbl = sim_dir / f"PhageSim_{tag}.mat.labels_to.csv"
    phage_den = sim_dir / f"PhageDendro_{tag}.png"

    step1_compare(phage_dir, phage_mat, phage_lbl, dry_run)
    step2_plot_standard(phage_mat, dry_run)
    # Phage uses the raw labels_to CSV directly (no genus prefixing)
    step4_dendrogram(
        phage_mat, phage_lbl, phage_den,
        cfg["phage_cut_point"], cfg["figsize_x"], cfg["figsize_y"],
        dry_run,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=f"{scripts_path}config_compare_sigs.yaml",
                        help=f"Path to config YAML (default: {scripts_path}config_compare_sigs.yaml)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")
    args = parser.parse_args()

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

    if args.dry_run:
        log.info("DRY-RUN MODE – no commands will be executed")

    combos = list(itertools.product(
        cfg["sketch_dirs"],
        cfg["n_values"],
        cfg["k_values"],
    ))
    log.info("Processing %d combination(s) across %d sketch dir(s)",
             len(combos), len(cfg["sketch_dirs"]))

    for sketch_dir_str, n, k in combos:
        process_combination(Path(sketch_dir_str), int(n), int(k), cfg, args.dry_run)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()