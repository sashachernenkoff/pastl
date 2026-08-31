r"""
Exploratory plots for ISM results.

Reads all per-gene .h5 files from --scores_dir and produces three figures:

  1. Score distributions  — violin per model × region type, PASTL vs Enformer
  2. Model agreement      — per-position scatter of PASTL ISM vs Enformer ISM,
                            coloured by region type
  3. Top positions        — horizontal bar chart of the 30 highest-scoring positions
                            (by mean disruption score), labelled gene:ref>alt

Output PNGs go to --output_dir.

Usage:
  python plot_ism.py \
      --scores_dir     out/analysis/ism_test/scores \
      --priority_genes out/analysis/ism_test/priority_genes.tsv \
      --output_dir     out/analysis/ism_test/figures
"""
import argparse
import sys
from pathlib import Path

import glob
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import palette

_ISM_DIR = "out/analysis/ism"

# Model colours — fixed PASTL (teal) vs Enformer (brick red) pairing.
MODEL_FILL = {
    "PASTL": palette.MODEL_FILL["pastl"],
    "Enformer": palette.MODEL_FILL["enformer"],
}

# Region colours — bold palette: tss=purple, utr5=blue, utr3=gold, gene_body=teal.
_BOLD = palette.PALETTES["bold"]
REGION_FILL = {
    "tss":       _BOLD["cat3"],   # deep blue
    "utr5":      _BOLD["cat4"],   # purple
    "utr3":      _BOLD["cat5"],   # gold/amber
    "gene_body": _BOLD["cat1"],   # teal
}

# Muted palette — same category assignments.
_MUTED = palette.PALETTES["muted"]
REGION_FILL_MUTED = {
    "tss":       _MUTED["cat3"],  # navy
    "utr5":      _MUTED["cat4"],  # mauve
    "utr3":      _MUTED["cat5"],  # gold
    "gene_body": _MUTED["cat1"],  # muted teal
}
MODEL_FILL_MUTED = {
    "PASTL": _MUTED["cat1"],
    "Enformer": _MUTED["cat2"],
}

REGION_LABEL = {"tss": "TSS ±2 kb", "utr5": "5' UTR", "utr3": "3' UTR", "gene_body": "Gene body"}

# Shared grey for reference lines (y=x diagonal, x=0 vertical).
_REF_LINE_COLOR = "#aaaaaa"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scores(scores_dir, priority_genes_tsv=None):
    """Read all .h5 files; return a tidy DataFrame."""
    group_map = {}
    if priority_genes_tsv and Path(priority_genes_tsv).exists():
        pg = pd.read_csv(priority_genes_tsv, sep="\t")
        group_map = dict(zip(pg["symbol"], pg["group"]))

    rows = []
    for h5_path in sorted(glob.glob(str(Path(scores_dir) / "*.h5"))):
        with h5py.File(h5_path, "r") as f:
            symbol      = str(f.attrs.get("symbol", Path(h5_path).stem))
            chrom       = str(f.attrs.get("chrom", ""))
            strand      = str(f.attrs.get("strand", "+"))
            ref_pastl  = float(f.attrs.get("ref_pastl",  0.0))
            ref_enformer = float(f.attrs.get("ref_enformer", 0.0))
            pos          = f["position"][:]
            ref_base     = f["ref_base"][:]
            alt_base     = f["alt_base"][:]
            region_type  = f["region_type"][:]
            pastl_ism   = f["pastl_ism"][:]
            enformer_ism = f["enformer_ism"][:]

        def _str(arr):
            return [v.decode() if isinstance(v, bytes) else str(v) for v in arr]

        n = len(pos)
        rows.append(pd.DataFrame({
            "symbol":       pd.Series([symbol] * n, dtype="category"),
            "chrom":        pd.Series([chrom] * n, dtype="category"),
            "strand":       pd.Series([strand] * n, dtype="category"),
            "group":        pd.Series([group_map.get(symbol, "unknown")] * n, dtype="category"),
            "ref_pastl":   [ref_pastl] * n,
            "ref_enformer": [ref_enformer] * n,
            "position":     pos.astype(int),
            "ref_base":     pd.Series(_str(ref_base), dtype="category"),
            "alt_base":     pd.Series(_str(alt_base), dtype="category"),
            "region_type":  pd.Series(_str(region_type), dtype="category"),
            "pastl_ism":   pastl_ism.astype(float),
            "enformer_ism": enformer_ism.astype(float),
        }))

    if not rows:
        raise FileNotFoundError(f"No .h5 files found in {scores_dir}")

    df = pd.concat(rows, ignore_index=True)
    df["mean_ism"] = 0.5 * (df["pastl_ism"] + df["enformer_ism"])
    df["pastl_ism_z"]   = (df["pastl_ism"]   - df["pastl_ism"].mean())   / df["pastl_ism"].std()
    df["enformer_ism_z"] = (df["enformer_ism"] - df["enformer_ism"].mean()) / df["enformer_ism"].std()
    df["mean_ism_z"] = 0.5 * (df["pastl_ism_z"] + df["enformer_ism_z"])
    print(f"Loaded {len(df):,} mutations across {df['symbol'].nunique()} genes")
    return df


# ---------------------------------------------------------------------------
# Figure 1: score distributions per model × region type
# ---------------------------------------------------------------------------

def plot_distributions(df, out_dir):
    region_types = [r for r in ["tss", "utr5", "utr3", "gene_body"] if r in df["region_type"].unique()]
    models = ["PASTL", "Enformer"]
    score_col = {"PASTL": "pastl_ism_z", "Enformer": "enformer_ism_z"}

    n_regions = len(region_types)
    fig, axes = plt.subplots(1, n_regions, figsize=(3.5 * n_regions, 4), sharey=True)
    if n_regions == 1:
        axes = [axes]

    for ax, rtype in zip(axes, region_types):
        sub = df[df["region_type"] == rtype]
        data_by_model = []
        for m in models:
            vals = sub[score_col[m]].values
            q02, q98 = np.quantile(vals, [0.02, 0.98])
            data_by_model.append(vals[(vals >= q02) & (vals <= q98)])

        parts = ax.violinplot(data_by_model, positions=[0, 1], widths=0.6,
                              showmedians=True, showextrema=False)
        for k, (body, m) in enumerate(zip(parts["bodies"], models)):
            body.set_facecolor(MODEL_FILL_MUTED[m])
            body.set_edgecolor("black")
            body.set_linewidth(0.5)
            body.set_alpha(0.85)
        parts["cmedians"].set_color("#4a4a4a")
        parts["cmedians"].set_linewidth(1.5)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(models, fontsize=8)
        ax.set_title(REGION_LABEL.get(rtype, rtype))
        if ax is axes[0]:
            ax.set_ylabel("ISM disruption score (z-scored)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    fig.suptitle("ISM score distributions by region type", y=1.01)
    fig.tight_layout()
    out = Path(out_dir) / "ism_distributions.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


# ---------------------------------------------------------------------------
# Figure 2: per-position model agreement scatter
# ---------------------------------------------------------------------------

def plot_model_agreement(df, out_dir):
    region_types = [r for r in ["tss", "utr5", "utr3", "gene_body"] if r in df["region_type"].unique()]

    fig, ax = plt.subplots(figsize=(5, 5))

    for rtype in region_types:
        sub = df[df["region_type"] == rtype]
        ax.scatter(
            sub["enformer_ism_z"], sub["pastl_ism_z"],
            c=REGION_FILL_MUTED.get(rtype, "#888"),
            label=REGION_LABEL.get(rtype, rtype),
            s=6, alpha=0.5, linewidths=0.3, edgecolors="black",
        )

    # Diagonal (perfect agreement) — thin grey line in axes coordinates
    ax.plot([0, 1], [0, 1], transform=ax.transAxes,
            color=_REF_LINE_COLOR, linewidth=0.6, linestyle="--", zorder=0)

    r = np.corrcoef(df["enformer_ism_z"], df["pastl_ism_z"])[0, 1]
    ax.set_xlabel("Enformer ISM score (z)")
    ax.set_ylabel("PASTL ISM score (z)")
    ax.set_title(f"Model agreement  (r = {r:.3f})")
    ax.legend(frameon=False, fontsize=8, markerscale=2)

    fig.tight_layout()
    out = Path(out_dir) / "ism_model_agreement.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


# ---------------------------------------------------------------------------
# Figure 3: top positions by mean disruption score
# ---------------------------------------------------------------------------

def plot_top_positions(df, out_dir, n=30):
    top = (
        df.sort_values("pastl_ism_z", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    top["label"] = (
        top["symbol"].astype(str) + "  "
        + top["ref_base"].astype(str) + ">" + top["alt_base"].astype(str)
        + "  (" + top["region_type"].astype(str) + ")"
    )
    colors_tlp = [MODEL_FILL["PASTL"]] * len(top)
    colors_enf = [MODEL_FILL["Enformer"]] * len(top)

    fig, ax = plt.subplots(figsize=(7, max(4, n * 0.28)))
    y = np.arange(len(top))
    bar_h = 0.35

    ax.barh(y - bar_h / 2, top["pastl_ism_z"],   bar_h, color=MODEL_FILL["PASTL"],  label="PASTL",  alpha=0.85)
    ax.barh(y + bar_h / 2, top["enformer_ism_z"], bar_h, color=MODEL_FILL["Enformer"], label="Enformer", alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(top["label"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("ISM disruption score (z-scored)")
    ax.set_title(f"Top {n} positions by PASTL disruption")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    out = Path(out_dir) / "ism_top_positions.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


# ---------------------------------------------------------------------------
# Figure 4: ISM ratio distribution by region type
# ---------------------------------------------------------------------------

def plot_ism_ratio(df, out_dir):
    from scipy.stats import gaussian_kde

    df = df.copy()
    df["z_diff"] = df["pastl_ism_z"] - df["enformer_ism_z"]

    region_types = [r for r in ["tss", "utr5", "utr3", "gene_body"] if r in df["region_type"].unique()]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    x_grid = np.linspace(-1, 1, 500)

    for rtype in region_types:
        vals = df.loc[df["region_type"] == rtype, "z_diff"].dropna().values
        if len(vals) < 2:
            continue
        
        q02, q98 = np.quantile(vals, [0.02, 0.98])
        vals = vals[(vals >= q02) & (vals <= q98)]
        if len(vals) < 2:
            continue
            
        if len(vals) > 10000:
            vals = np.random.choice(vals, size=10000, replace=False)

        kde = gaussian_kde(vals)
        density = kde(x_grid)
        median = np.median(vals)
        sign = "+" if median >= 0 else ""
        label = f"{REGION_LABEL.get(rtype, rtype)} (median = {sign}{median:.2f})"
        ax.plot(x_grid, density,
                color=REGION_FILL.get(rtype, "#888"),
                linewidth=0.7, alpha=0.85, label=label)
        ax.fill_between(x_grid, density,
                        color=REGION_FILL.get(rtype, "#ccc"),
                        alpha=0.2)

    ax.axvline(0, color=_REF_LINE_COLOR, linewidth=0.6, linestyle="--")
    ax.set_xlim(-1, 1)

    ax.set_title("Relative ISM sensitivity: PASTL vs Enformer")
    ax.set_xlabel("PASTL ISM score (z) - Enformer ISM score (z)")
    ax.set_ylabel("Density")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.grid(False)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = Path(out_dir) / f"ism_ratio_distribution.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"  Saved -> {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scores_dir", required=True,
                        help="Directory containing per-gene .h5 files from run_ism.py")
    parser.add_argument("--priority_genes", default=None,
                        help="Path to priority_genes.tsv (optional; adds group labels)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_n", type=int, default=30,
                        help="Number of positions to show in the top-positions plot (default: 30)")
    args = parser.parse_args()

    palette.use_style()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    df = load_scores(args.scores_dir, args.priority_genes)

    print("Plotting distributions ...")
    plot_distributions(df, args.output_dir)

    print("Plotting model agreement ...")
    plot_model_agreement(df, args.output_dir)

    print(f"Plotting top {args.top_n} positions ...")
    plot_top_positions(df, args.output_dir, n=args.top_n)

    print("Plotting ISM ratio distribution ...")
    plot_ism_ratio(df, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()