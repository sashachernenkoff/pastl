r"""
Gene-level correlation stratification analysis.

Two horizontal dot-plots, each sampling ~N genes evenly from a ranked list and
displaying per-gene squared residual / model MSE on the x-axis with gene names
on the y-axis (low ranking at top, high at bottom).

Analysis 1 — Tissue-specificity dot-plot
  Ranks test-set genes by their tau specificity index (from GTEx median TPM per
  tissue; tau = 0 broadly expressed, tau = 1 highly tissue-specific), samples N
  genes evenly across the spectrum, and plots two series per gene row:
    - PASTL per-gene squared residual / model MSE
    - Enformer per-gene squared residual / model MSE

Analysis 2 — UTR-complexity dot-plot
  Ranks test-set genes by post-transcriptional regulatory complexity (total
  5'UTR + 3'UTR length from the GENCODE GTF, matched to the canonical transcript
  via TxID in sequences_protein.bed), ordered by ascending complexity, and shows
  two series:
    - PASTL prediction cross-gene r (same for all rows)
    - Enformer prediction cross-gene r (same for all rows)

Usage:
  python stratified_correlation.py \
      --wandb_run_id m7h1162n --checkpoint best.pt \
      --gtex_median_tpm /path/to/GTEx_gene_median_tpm.gct.gz \
      --gtf /path/to/gencode.v36.annotation.gtf.gz \
      --output_dir out/analysis/m7h1162n_best/stratified_correlation/
"""
import argparse
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from statsmodels.nonparametric.smoothers_lowess import lowess

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from palette import INK, NEUTRALS

_PREDICTIONS_DIR = "out/predictions"
_DEFAULT_SEQUENCES_BED = str(
    Path(__file__).parent.parent.parent / "data" / "sequences_protein.bed"
)

# Colour assignments (project INK shades; semantic match to series role)
_COLOR_PASTL   = INK["seafoam"]     # teal   – PASTL prediction
_COLOR_ENFORMER = INK["terracotta"]  # coral  – Enformer prediction


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def ckpt_slug(run_id: str, checkpoint: str) -> str:
    return f"{run_id}_{Path(checkpoint).stem}"


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_pickle(path)
    df = df.rename(columns={"protein_id": "gene"})
    return df[["gene", "true_label", "prediction"]].dropna().set_index("gene")


def crossgene_r(preds: pd.DataFrame) -> tuple[float, int]:
    r, _ = pearsonr(preds["true_label"], preds["prediction"])
    return float(r), len(preds)


def sample_evenly(series: pd.Series, n: int) -> pd.Series:
    """Return n items from series spaced evenly across its sorted range."""
    s = series.dropna().sort_values()
    if len(s) <= n:
        return s
    idx = np.round(np.linspace(0, len(s) - 1, n)).astype(int)
    return s.iloc[idx]


# ---------------------------------------------------------------------------
# Analysis 1 helpers
# ---------------------------------------------------------------------------

def compute_tau(gtex_path: str) -> pd.Series:
    """
    Compute the tau tissue-specificity index per gene from GTEx median TPM.

    tau = sum(1 - x_hat_i) / (n - 1), where x_hat_i = x_i / max(x).
    Returns a Series indexed by gene symbol; genes with max TPM = 0 get NaN.
    Range: 0 = ubiquitously expressed, 1 = tissue-specific.
    """
    print(f"Loading GTEx median TPM: {gtex_path}")
    gtex = pd.read_csv(gtex_path, sep="\t", skiprows=2)
    # Deduplicate by gene symbol, keep first occurrence
    gtex = gtex.drop_duplicates(subset="Description")
    gtex = gtex.set_index("Description")
    # Tissue columns are everything after Name and Description
    tissue_cols = [c for c in gtex.columns if c != "Name"]
    tpm = gtex[tissue_cols].astype(float)

    n = tpm.shape[1]
    max_tpm = tpm.max(axis=1)
    # Genes with all-zero expression get NaN
    valid = max_tpm > 0
    tau = pd.Series(np.nan, index=tpm.index, name="tau")
    x_hat = tpm.loc[valid].div(max_tpm[valid], axis=0)
    tau[valid] = (1 - x_hat).sum(axis=1) / (n - 1)
    return tau


# ---------------------------------------------------------------------------
# Analysis 2 helpers
# ---------------------------------------------------------------------------

def parse_utr_lengths(symbols: pd.Series, txids: pd.Series,
                      strands: pd.Series, gtf_path: str) -> pd.DataFrame:
    """
    Compute per-gene total 5'UTR and 3'UTR lengths from the GTF.
    Uses canonical transcript ID (TxID, version suffix stripped) for matching.
    Returns a DataFrame indexed by gene symbol with columns:
      utr5_len, utr3_len, complexity (= utr5_len + utr3_len).
    """
    target_tx = dict(zip(symbols, txids))
    strand_map = dict(zip(symbols, strands))

    raw: dict = defaultdict(lambda: {"UTR": [], "CDS": []})

    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    print(f"Parsing GTF: {gtf_path}")
    with opener(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 9:
                continue
            feature = fields[2]
            if feature not in ("UTR", "CDS"):
                continue
            attrs = {
                kv[0]: kv[1].strip('"')
                for item in fields[8].strip().split(";")
                if len(kv := item.strip().split(" ", 1)) == 2
            }
            symbol = attrs.get("gene_name")
            tx = attrs.get("transcript_id", "").split(".")[0]
            if symbol not in target_tx or tx != target_tx[symbol]:
                continue
            raw[symbol][feature].append((int(fields[3]), int(fields[4])))

    records = []
    for symbol in symbols:
        strand = strand_map[symbol]
        cds = raw[symbol]["CDS"]
        utrs = raw[symbol]["UTR"]
        utr5, utr3 = [], []
        if cds and utrs:
            cds_min = min(s for s, _ in cds)
            cds_max = max(e for _, e in cds)
            for u_s, u_e in utrs:
                if strand == "+":
                    (utr5 if u_e <= cds_min else utr3).append((u_s, u_e))
                else:
                    (utr5 if u_s >= cds_max else utr3).append((u_s, u_e))
        len5 = sum(e - s for s, e in utr5)
        len3 = sum(e - s for s, e in utr3)
        records.append({"gene": symbol, "utr5_len": len5,
                        "utr3_len": len3, "complexity": len5 + len3})
    return pd.DataFrame(records).set_index("gene")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def draw_dotplot(
    ax,
    gene_order: list[str],
    series: list[tuple[str, str, dict]],
    title: str,
    xlabel: str = "Pearson r",
    all_genes_data: list[tuple[str, str, pd.Series]] | None = None,
) -> None:
    """
    Horizontal dot-plot.

    gene_order : genes listed top→bottom (index 0 = top of y-axis).
    series     : list of (label, colour, {gene: x_value}).
    """
    y = np.arange(len(gene_order))
    n = len(series)
    offsets = np.linspace(-0.18, 0.18, n) if n > 1 else [0.0]

    if all_genes_data is not None:
        for label, color, full_series in all_genes_data:
            window = max(3, len(full_series) // 5)
            rolling_med = full_series.rolling(window, center=True, min_periods=1).median()
            y_mapped = np.linspace(0, len(gene_order) - 1, len(full_series))
            ax.plot(rolling_med.values, y_mapped, color=color, alpha=0.4, linewidth=2.5, zorder=1)

    for (label, color, x_map), dy in zip(series, offsets):
        x_vals = np.array([x_map.get(g, np.nan) for g in gene_order])
        ax.scatter(x_vals, y + dy, color=color, s=55, alpha=0.88,
                   linewidths=0, label=label, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(gene_order, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="lower right")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wandb_run_id", required=True,
                        help="WandB run ID for PASTL.")
    parser.add_argument("--checkpoint", default="best.pt",
                        help="Checkpoint filename (default: best.pt).")
    parser.add_argument("--gtex_median_tpm", required=True,
                        help="GTEx gene-level median TPM GCT file (.gct.gz).")
    parser.add_argument("--sequences_bed", default=_DEFAULT_SEQUENCES_BED,
                        help="sequences_protein.bed (default: data/ in repo root).")
    parser.add_argument("--gtf", required=True,
                        help="GENCODE GTF (.gtf.gz) for 5'/3' UTR annotation.")
    parser.add_argument("--predictions_dir", default=_PREDICTIONS_DIR,
                        help="Root predictions directory (default: %(default)s).")
    parser.add_argument("--n_genes", type=int, default=40,
                        help="Genes to sample per plot (default: 40).")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory; figures/, tables/, logs/ created inside.")
    args = parser.parse_args()

    style_path = Path(__file__).parent.parent.parent / "pastl.mplstyle"
    if style_path.exists():
        plt.style.use(str(style_path))

    out_dir = Path(args.output_dir)
    figs_dir, tables_dir, logs_dir = (out_dir / d for d in ("figures", "tables", "logs"))
    for d in (figs_dir, tables_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Predictions ───────────────────────────────────────────────────────────
    slug = ckpt_slug(args.wandb_run_id, args.checkpoint)
    pastl_path  = Path(args.predictions_dir) / slug / "predictions.pkl"
    enformer_path = Path(args.predictions_dir) / "enformer" / "predictions.pkl"

    for p, name in [(pastl_path, "PASTL"), (enformer_path, "Enformer")]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {name} predictions at {p}")

    pastl  = load_predictions(pastl_path)
    enformer = load_predictions(enformer_path)

    r_pastl,  n_pastl  = crossgene_r(pastl)
    r_enformer, n_enformer = crossgene_r(enformer)
    mse_pastl   = ((pastl["prediction"]  - pastl["true_label"])  ** 2).mean()
    mse_enformer = ((enformer["prediction"] - enformer["true_label"]) ** 2).mean()
    test_genes = set(pastl.index) & set(enformer.index)

    print(f"PASTL  cross-gene r = {r_pastl:.4f}  ({n_pastl:,} test genes)")
    print(f"Enformer cross-gene r = {r_enformer:.4f}  ({n_enformer:,} test genes)")

    # ── Analysis 1: Tissue-specificity dot-plot ───────────────────────────────
    print("\n--- Analysis 1: Tissue-specificity dot-plot ---")
    tau = compute_tau(args.gtex_median_tpm)

    # Restrict to genes present in both prediction DataFrames
    tau_test = tau[tau.index.isin(test_genes)].dropna()
    print(f"  {len(tau):,} genes with tau  →  {len(tau_test):,} overlap with test set")
    print(f"  Median tau = {tau_test.median():.4f}  "
          f"range [{tau_test.min():.4f}, {tau_test.max():.4f}]")

    # Rank descending: broadly expressed (low tau) at top, tissue-specific at bottom
    sampled1 = sample_evenly(tau_test, args.n_genes)
    gene_order1 = sampled1.sort_values(ascending=False).index.tolist()

    tlp_err1 = {g: (pastl.loc[g, "prediction"] - pastl.loc[g, "true_label"]) ** 2 / mse_pastl
                for g in gene_order1 if g in pastl.index}
    enf_err1 = {g: (enformer.loc[g, "prediction"] - enformer.loc[g, "true_label"]) ** 2 / mse_enformer
                for g in gene_order1 if g in enformer.index}

    tau_all_sorted = tau_test.sort_values(ascending=False)
    all_tlp_err1 = pd.Series(
        {g: (pastl.loc[g, "prediction"] - pastl.loc[g, "true_label"]) ** 2 / mse_pastl
         for g in tau_all_sorted.index if g in pastl.index}
    ).reindex(tau_all_sorted.index)
    all_enf_err1 = pd.Series(
        {g: (enformer.loc[g, "prediction"] - enformer.loc[g, "true_label"]) ** 2 / mse_enformer
         for g in tau_all_sorted.index if g in enformer.index}
    ).reindex(tau_all_sorted.index)

    fig_h = max(5.0, 0.32 * args.n_genes + 2.0)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    draw_dotplot(
        ax, gene_order1,
        series=[
            ("PASTL (per-gene residual² / MSE)",  _COLOR_PASTL,   tlp_err1),
            ("Enformer (per-gene residual² / MSE)", _COLOR_ENFORMER, enf_err1),
        ],
        title="Tissue Specificity (τ) vs Model Prediction Error",
        xlabel="Squared residual / model MSE (log scale)",
        all_genes_data=[
            ("PASTL (per-gene residual² / MSE)",  _COLOR_PASTL,   all_tlp_err1),
            ("Enformer (per-gene residual² / MSE)", _COLOR_ENFORMER, all_enf_err1),
        ],
    )
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(figs_dir / "tissue_specificity_dotplot.png", dpi=300, bbox_inches="tight")
    fig.savefig(figs_dir / "tissue_specificity_dotplot.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved tissue_specificity_dotplot")

    pd.DataFrame({
        "gene":                  gene_order1,
        "tau":                   [sampled1[g] for g in gene_order1],
        "pastl_rel_sq_error":   [tlp_err1.get(g, float("nan")) for g in gene_order1],
        "enformer_rel_sq_error": [enf_err1.get(g, float("nan")) for g in gene_order1],
    }).to_csv(tables_dir / "tissue_specificity_genes.csv", index=False)

    # ── Analysis 2: UTR-complexity dot-plot ───────────────────────────────────
    print("\n--- Analysis 2: UTR-complexity dot-plot ---")
    bed_raw = pd.read_csv(args.sequences_bed, sep="\t")
    bed_df = (
        bed_raw[["symbol", "strand", "txid"]]
        .drop_duplicates("symbol")
        .assign(txid_clean=lambda d: d["txid"].astype(str).str.split(".").str[0])
    )

    utr_df = parse_utr_lengths(
        bed_df["symbol"], bed_df["txid_clean"], bed_df["strand"], args.gtf
    )
    n_annotated = utr_df["complexity"].gt(0).sum()
    print(f"  {n_annotated:,} genes with non-zero UTR annotation "
          f"({len(utr_df):,} total in BED)")

    # Restrict to test-set genes with UTR annotation
    utr_test = utr_df[utr_df.index.isin(test_genes)]
    print(f"  {len(utr_test):,} overlap with test set")

    sampled2 = sample_evenly(utr_test["complexity"], args.n_genes)
    gene_order2 = sampled2.sort_values(ascending=False).index.tolist()

    tlp_err2 = {g: (pastl.loc[g, "prediction"] - pastl.loc[g, "true_label"]) ** 2 / mse_pastl
                for g in gene_order2 if g in pastl.index}
    enf_err2 = {g: (enformer.loc[g, "prediction"] - enformer.loc[g, "true_label"]) ** 2 / mse_enformer
                for g in gene_order2 if g in enformer.index}

    utr_all_sorted = utr_test["complexity"].sort_values(ascending=False)
    all_tlp_err2 = pd.Series(
        {g: (pastl.loc[g, "prediction"] - pastl.loc[g, "true_label"]) ** 2 / mse_pastl
         for g in utr_all_sorted.index if g in pastl.index}
    ).reindex(utr_all_sorted.index)
    all_enf_err2 = pd.Series(
        {g: (enformer.loc[g, "prediction"] - enformer.loc[g, "true_label"]) ** 2 / mse_enformer
         for g in utr_all_sorted.index if g in enformer.index}
    ).reindex(utr_all_sorted.index)

    fig, ax = plt.subplots(figsize=(8, fig_h))
    draw_dotplot(
        ax, gene_order2,
        series=[
            ("PASTL (per-gene residual² / MSE)",  _COLOR_PASTL,   tlp_err2),
            ("Enformer (per-gene residual² / MSE)", _COLOR_ENFORMER, enf_err2),
        ],
        title="Post-Transcriptional Regulatory Complexity vs Model Prediction Error",
        xlabel="Squared residual / model MSE (log scale)",
        all_genes_data=[
            ("PASTL (per-gene residual² / MSE)",  _COLOR_PASTL,   all_tlp_err2),
            ("Enformer (per-gene residual² / MSE)", _COLOR_ENFORMER, all_enf_err2),
        ],
    )
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(figs_dir / "utr_complexity_dotplot.png", dpi=300, bbox_inches="tight")
    fig.savefig(figs_dir / "utr_complexity_dotplot.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved utr_complexity_dotplot")

    tbl2 = utr_test.loc[gene_order2, ["utr5_len", "utr3_len", "complexity"]].copy()
    tbl2["pastl_rel_sq_error"]  = [tlp_err2.get(g, float("nan")) for g in gene_order2]
    tbl2["enformer_rel_sq_error"] = [enf_err2.get(g, float("nan")) for g in gene_order2]
    tbl2.index.name = "gene"
    tbl2.to_csv(tables_dir / "utr_complexity_genes.csv")

    # ── Analysis 3: Binned Pearson r vs tissue specificity ────────────────────
    print("\n--- Analysis 3: Binned Pearson r vs tau ---")
    tau_genes = tau[tau.index.isin(test_genes)].dropna().sort_values()
    tau_bins = pd.qcut(tau_genes, q=10, labels=False)

    bin_records = []
    for b in range(10):
        mask = tau_bins == b
        genes_b = tau_genes.index[mask].tolist()
        tlp_b = pastl.loc[[g for g in genes_b if g in pastl.index]]
        enf_b = enformer.loc[[g for g in genes_b if g in enformer.index]]
        r_tlp = pearsonr(tlp_b["true_label"], tlp_b["prediction"])[0] if len(tlp_b) > 1 else float("nan")
        r_enf = pearsonr(enf_b["true_label"], enf_b["prediction"])[0] if len(enf_b) > 1 else float("nan")
        bin_records.append({
            "bin":         b,
            "median_tau":  float(tau_genes[mask].median()),
            "min_tau":     float(tau_genes[mask].min()),
            "max_tau":     float(tau_genes[mask].max()),
            "n_genes":     int(mask.sum()),
            "pastl_r":    r_tlp,
            "enformer_r":  r_enf,
            "delta_r":     r_tlp - r_enf,
        })
    bin_df = pd.DataFrame(bin_records)
    print(f"  Binned {len(tau_genes):,} genes into 10 deciles")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = bin_df["median_tau"].values
    y_tlp = bin_df["pastl_r"].values
    y_enf = bin_df["enformer_r"].values

    ax.axhline(0, color="gray", linestyle="--", linewidth=1, zorder=1)
    above = y_tlp > y_enf
    ax.fill_between(x, y_tlp, y_enf, where=above, color="#006D77", alpha=0.1, zorder=2)
    ax.plot(x, y_tlp, color="#006D77", marker="o", label="PASTL", zorder=3)
    ax.plot(x, y_enf, color="#E29578", marker="o", label="Enformer", zorder=3)

    ax.set_xlabel("Tissue specificity (τ)")
    ax.set_ylabel("Pearson r")
    ax.set_title("Per-bin Prediction Accuracy vs Tissue Specificity",
                 fontweight="bold", pad=15)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.grid(False)
    ax.minorticks_off()
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(figs_dir / "tau_binned_pearsonr.png", dpi=300, bbox_inches="tight")
    fig.savefig(figs_dir / "tau_binned_pearsonr.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved tau_binned_pearsonr")

    bin_df.to_csv(tables_dir / "tau_binned_pearsonr.csv", index=False)

    # ── Logs ──────────────────────────────────────────────────────────────────
    with open(logs_dir / "stratified_correlation_summary.txt", "w") as f:
        f.write("=== Stratified Correlation Analysis ===\n")
        f.write(f"Model slug:        {slug}\n")
        f.write(f"Genes sampled:     {args.n_genes}\n\n")
        f.write("Model reference (cross-gene, test set):\n")
        f.write(f"  PASTL  r = {r_pastl:.4f}  ({n_pastl:,} genes)\n")
        f.write(f"  Enformer r = {r_enformer:.4f}  ({n_enformer:,} genes)\n\n")
        f.write("Tau tissue-specificity (test-set genes with GTEx data):\n")
        f.write(f"  N genes  = {len(tau_test):,}\n")
        f.write(f"  Median   = {tau_test.median():.4f}\n")
        f.write(f"  Range    = [{tau_test.min():.4f}, {tau_test.max():.4f}]\n\n")
        f.write("UTR complexity (sampled test-set genes):\n")
        f.write(f"  Range    = [{sampled2.min():.0f}, {sampled2.max():.0f}] bp\n")
        f.write(f"  Median   = {sampled2.median():.0f} bp\n")

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
