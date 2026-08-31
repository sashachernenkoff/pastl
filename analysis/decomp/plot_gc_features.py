r"""
Gene length, GC content, and 3' UTR analysis of PASTL improvement.

Evaluates whether PASTL's predictive advantage over Enformer is associated with
gene-level sequence features: gene length, 3' UTR length, total GC content, and
3' UTR GC content.

For each gene in the shared test set, ΔSE = SE_Enformer − SE_PASTL measures
improvement (positive = PASTL is better). A multivariable OLS regression
assesses the joint contribution of log10(gene length), log10(3' UTR length + 1),
total GC fraction, and 3' UTR GC fraction.

Inputs:
  predictions.pkl (PASTL)    : {predictions_dir}/{run_id}_{ckpt_stem}/predictions.pkl
  predictions.pkl (Enformer) : {predictions_dir}/enformer/predictions.pkl
  genes_gc.bed               : gene coordinates + bedtools nuc columns from extract_gc.sh
  utrs_gc.bed                : 3' UTR intervals + bedtools nuc columns from extract_gc.sh

Usage:
  python plot_gc_features.py \
      --wandb_run_id m7h1162n --checkpoint best.pt \
      --genes_gc   out/analysis/decomp/genes_gc.bed \
      --utrs_gc    out/analysis/decomp/utrs_gc.bed \
      --output_dir out/analysis/m7h1162n_best/gc_features/
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from palette import FILL, INK, NEUTRALS

_PREDICTIONS_DIR = "out/predictions"

_BED_NAMES = ["chrom", "start", "end", "gene", "score", "strand"]


def ckpt_slug(run_id: str, checkpoint: str) -> str:
    return f"{run_id}_{Path(checkpoint).stem}"


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_pickle(path)
    df = df.rename(columns={"protein_id": "gene"})
    return df[["gene", "true_label", "prediction"]].dropna().set_index("gene")


def load_nuc_bed(path: Path) -> pd.DataFrame:
    """Read a bedtools nuc output file, using its own header to name columns.

    The header tokens labelled N_usercol are mapped to BED column names in
    order; all other tokens (pct_at, pct_gc, num_A, …, seq_len) keep the
    name that bedtools wrote, stripped of the leading number prefix.
    """
    with open(path) as f:
        header = f.readline().strip().lstrip("#").strip()
    col_names = []
    bed_idx = 0
    for raw in header.split("\t"):
        stripped = re.sub(r"^\d+_", "", raw.strip())
        if stripped == "usercol":
            col_names.append(_BED_NAMES[bed_idx])
            bed_idx += 1
        else:
            col_names.append(stripped)
    df = pd.read_csv(path, sep="\t", skiprows=1, header=None, names=col_names)
    for col in ("start", "end", "num_A", "num_C", "num_G", "num_T", "num_N", "num_oth", "seq_len"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_gene_features(genes_bed: pd.DataFrame) -> pd.DataFrame:
    """One row per gene: gene_len (bp) and gene_gc (fraction)."""
    df = genes_bed.copy()
    df["gene_len"] = df["end"] - df["start"]
    df = df[df["seq_len"] > 0].copy()
    df["gene_gc"] = (df["num_C"] + df["num_G"]) / df["seq_len"]
    return df.groupby("gene", as_index=False).first()[["gene", "gene_len", "gene_gc"]]


def build_utr3_features(utrs_bed: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-gene 3' UTR length and weighted GC fraction."""
    df = utrs_bed[utrs_bed["seq_len"] > 0].copy()
    df["interval_len"] = df["end"] - df["start"]
    df["n_gc"] = df["num_C"] + df["num_G"]
    agg = df.groupby("gene").agg(
        utr3_len=("interval_len", "sum"),
        n_gc=("n_gc", "sum"),
        seq_len=("seq_len", "sum"),
    ).reset_index()
    agg["utr3_gc"] = agg["n_gc"] / agg["seq_len"]
    return agg[["gene", "utr3_len", "utr3_gc"]]


def _draw_scatter(ax, x, y, color, title, xlabel, ylabel, stats_lines):
    """Scatter with OLS fit line and a stats annotation box."""
    ax.scatter(x, y, color=color, alpha=0.45, s=12, linewidths=0)

    slope, intercept = np.polyfit(x, y, 1)
    x_range = np.array([x.min(), x.max()])
    ax.plot(x_range, slope * x_range + intercept,
            color=NEUTRALS["axis_text"], linewidth=1.2)

    ax.text(0.05, 0.95, "\n".join(stats_lines),
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor=NEUTRALS["page_bg"], edgecolor="none", alpha=0.9))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wandb_run_id", required=True,
                        help="WandB run ID for PASTL.")
    parser.add_argument("--checkpoint", default="best.pt",
                        help="Checkpoint filename; its stem forms the slug "
                             "(default: best.pt).")
    parser.add_argument("--genes_gc", required=True,
                        help="Path to genes_gc.bed from extract_gc.sh.")
    parser.add_argument("--utrs_gc", required=True,
                        help="Path to utrs_gc.bed from extract_gc.sh.")
    parser.add_argument("--predictions_dir", default=_PREDICTIONS_DIR,
                        help="Root predictions directory (default: %(default)s).")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory; figures/, tables/, and logs/ "
                             "will be created inside it.")
    args = parser.parse_args()

    style_path = Path(__file__).parent.parent.parent / "pastl.mplstyle"
    if style_path.exists():
        plt.style.use(str(style_path))

    out_dir = Path(args.output_dir)
    figs_dir   = out_dir / "figures"
    tables_dir = out_dir / "tables"
    logs_dir   = out_dir / "logs"
    for d in (figs_dir, tables_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Load predictions ──────────────────────────────────────────────────────
    slug = ckpt_slug(args.wandb_run_id, args.checkpoint)
    tlp_path = Path(args.predictions_dir) / slug / "predictions.pkl"
    enf_path = Path(args.predictions_dir) / "enformer" / "predictions.pkl"

    for p, name in [(tlp_path, "PASTL"), (enf_path, "Enformer")]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {name} predictions at {p}")

    tlp = load_predictions(tlp_path)
    enf = load_predictions(enf_path)
    print(f"PASTL predictions:  {len(tlp):,} test-set genes")
    print(f"Enformer predictions: {len(enf):,} test-set genes")

    # ── Compute ΔSE per gene ─────────────────────────────────────────────────
    # ΔSE = SE_Enformer − SE_PASTL; positive values indicate PASTL improvement
    common_genes = sorted(set(tlp.index) & set(enf.index))
    print(f"Gene intersection (predictions): {len(common_genes):,}")

    se_tlp = (tlp.loc[common_genes, "prediction"] - tlp.loc[common_genes, "true_label"]) ** 2
    se_enf = (enf.loc[common_genes, "prediction"] - enf.loc[common_genes, "true_label"]) ** 2
    delta_se = (se_enf - se_tlp).rename("delta_se")

    # ── Load and parse GC bed files ───────────────────────────────────────────
    print(f"Loading genes_gc.bed: {args.genes_gc}")
    gene_feat = build_gene_features(load_nuc_bed(Path(args.genes_gc)))
    print(f"  {len(gene_feat):,} genes with seq_len > 0")
    gene_feat = gene_feat.set_index("gene")

    print(f"Loading utrs_gc.bed:  {args.utrs_gc}")
    utr3_feat = build_utr3_features(load_nuc_bed(Path(args.utrs_gc)))
    utr3_feat = utr3_feat.set_index("gene")

    # ── Merge all features ────────────────────────────────────────────────────
    merged = (
        delta_se.to_frame()
        .join(gene_feat[["gene_len", "gene_gc"]], how="inner")
        .join(utr3_feat[["utr3_len",  "utr3_gc"]],  how="left")
    )
    merged["utr3_len"] = merged["utr3_len"].fillna(0)
    merged["utr3_gc"]  = merged["utr3_gc"].fillna(merged["gene_gc"])

    merged["log_gene_len"] = np.log10(merged["gene_len"].clip(lower=1))
    merged["log_utr3_len"] = np.log10(merged["utr3_len"] + 1)

    merged = merged.dropna(subset=["delta_se", "log_gene_len", "log_utr3_len",
                                   "gene_gc", "utr3_gc"])
    n = len(merged)
    print(f"Genes after feature join: {n:,}")

    # ── Multivariable OLS regression ──────────────────────────────────────────
    features = ["log_gene_len", "log_utr3_len", "gene_gc", "utr3_gc"]
    X = sm.add_constant(merged[features])
    ols_result = sm.OLS(merged["delta_se"], X).fit()
    print("\nOLS summary:")
    print(ols_result.summary())

    # ── Bivariate Pearson r for scatter annotations ───────────────────────────
    r_gene_len, p_gene_len = pearsonr(merged["log_gene_len"], merged["delta_se"])
    r_utr3_gc,  p_utr3_gc  = pearsonr(merged["utr3_gc"],     merged["delta_se"])

    # ── Tables ────────────────────────────────────────────────────────────────
    coef_df = pd.DataFrame({
        "feature":   ols_result.params.index,
        "coef":      ols_result.params.values,
        "std_err":   ols_result.bse.values,
        "t_stat":    ols_result.tvalues.values,
        "p_value":   ols_result.pvalues.values,
    })
    coef_df.to_csv(tables_dir / "ols_coefficients.csv", index=False)

    merged.reset_index()[["gene", "delta_se", "gene_len", "gene_gc",
                           "utr3_len", "utr3_gc",
                           "log_gene_len", "log_utr3_len"]].to_csv(
        tables_dir / "per_gene_features.csv", index=False
    )

    # ── Logs ─────────────────────────────────────────────────────────────────
    with open(logs_dir / "gc_features_summary.txt", "w") as f:
        f.write("=== GC Feature Analysis: PASTL vs Enformer ΔSE ===\n")
        f.write(f"Model slug:  {slug}\n")
        f.write(f"n genes:     {n}\n\n")
        f.write("ΔSE = SE_Enformer − SE_PASTL  (positive = PASTL is better)\n\n")
        f.write("Bivariate Pearson r:\n")
        f.write(f"  log10(gene_len) vs ΔSE:  r = {r_gene_len:.4f}  p = {p_gene_len:.2e}\n")
        f.write(f"  3' UTR GC       vs ΔSE:  r = {r_utr3_gc:.4f}  p = {p_utr3_gc:.2e}\n\n")
        f.write("Multivariable OLS  (ΔSE ~ log_gene_len + log_utr3_len + gene_gc + utr3_gc):\n")
        f.write(f"  R² = {ols_result.rsquared:.4f}   adj-R² = {ols_result.rsquared_adj:.4f}\n\n")
        f.write(ols_result.summary().as_text())

    # =========================================================================
    # Figure: 2-panel scatter (figsize=(10, 4))
    # Panel A – ΔSE vs log10 Gene Length
    # Panel B – ΔSE vs 3' UTR GC Content
    # =========================================================================
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4))

    _draw_scatter(
        ax_a,
        merged["log_gene_len"].values,
        merged["delta_se"].values,
        INK["seafoam"],
        "Panel A — Gene Length vs SE Improvement",
        "log₁₀(Gene Length, bp)",
        "ΔSE  (Enformer − PASTL)",
        [f"r = {r_gene_len:.3f}", f"p = {p_gene_len:.2e}", f"n = {n:,}"],
    )

    _draw_scatter(
        ax_b,
        merged["utr3_gc"].values,
        merged["delta_se"].values,
        INK["periwinkle"],
        "Panel B — 3’ UTR GC Content vs SE Improvement",
        "3’ UTR GC Fraction",
        "ΔSE  (Enformer − PASTL)",
        [f"r = {r_utr3_gc:.3f}", f"p = {p_utr3_gc:.2e}", f"n = {n:,}"],
    )

    fig.tight_layout()
    fig.savefig(figs_dir / "gc_features_scatter.png", dpi=300, bbox_inches="tight")
    fig.savefig(figs_dir / "gc_features_scatter.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved gc_features_scatter")

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
