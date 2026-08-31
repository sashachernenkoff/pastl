r"""
Stratified prediction-error analysis over lung eGenes (test set only).

Takes the significant GTEx lung eGenes matched to the test set and scores model
prediction error within three axes:

  1. cis-eQTL effect size -> fixed absolute-threshold bins of |log2_aFC|
                             (Weak/Moderate/Strong at 0.3 / 0.6 by default;
                             falls back to |slope| if log2_aFC is absent).
  2. Subcellular          -> GO Cellular Component compartment spectrum, ordered
     localization            nuclear -> extracellular.
  3. Across-patient       -> fixed-threshold bins of the IQR (or std) of log2
     variability             abundance across the cohort.

All three axes stratify the same universe (significant eGenes in the test set)
and share the same figure template: an absolute-error boxplot over every stratum
plus a vertical stack of obs-vs-pred scatter panels.

Predictions:
  {predictions_dir}/{wandb_run_id}_{ckpt_stem}/predictions.pkl
  columns protein_id / true_label / prediction (test split; see analysis/predict.py).

Inputs precomputed by the sibling builder scripts:
  build_localization_map.py  -> localization_symbols.tsv  (gene -> compartment)
  build_variance_table.py    -> protein_variance.tsv      (gene, iqr, std)

Usage:
  python plot_egene_stratification.py \
      --wandb_run_id m7h1162n --checkpoint best.pt \
      --egenes           /path/to/Lung.v8.egenes.txt.gz \
      --localization_map out/analysis/egene_eval/localization_symbols.tsv \
      --variance_table   out/analysis/egene_eval/protein_variance.tsv \
      --output_dir out/analysis/m7h1162n_best/egene_eval/
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "model"))

from evaluate import compute_all_metrics
from palette import FILL, INK, GROUP_FILLS, GROUP_INKS

_PREDICTIONS_DIR = "out/predictions"
_DEFAULT_LOCALIZATION = "out/analysis/egene_eval/localization_symbols.tsv"
_DEFAULT_VARIANCE     = "out/analysis/egene_eval/protein_variance.tsv"

# Absolute cut points (~1.23-fold and ~1.52-fold interquantile spread). Both the
# eGene |log2_aFC| and the across-patient IQR distributions are heavily skewed
# toward small values (medians ~0.2-0.3), so a 2-fold cut leaves the top bin
# nearly empty; 0.3/0.6 keeps every bin populated. Bins are half-open [lo, hi)
# and NOT equal-count, so sizes reflect the real distribution.
_DEFAULT_EFFECT_THRESHOLDS = [0.3, 0.6]
_DEFAULT_VARIANCE_THRESHOLDS = [0.3, 0.6]

# Subcellular compartments ordered genome-proximal (transcription-coupled) ->
# genome-distal (secreted). This is the boxplot display order.
_LOCALIZATION_ORDER = ["Nuclear", "Cytoplasm", "Cytoskeleton", "Mitochondrion",
                       "Secretory", "Membrane", "Extracellular"]
# Assignment priority for multi-compartment proteins: the most peripheral /
# distinctive compartment wins (a secreted protein transiently in the cytosol is
# "Extracellular"). Cytoplasm is the catch-all of last resort.
_LOCALIZATION_PRIORITY = ["Extracellular", "Membrane", "Secretory", "Mitochondrion",
                          "Cytoskeleton", "Nuclear", "Cytoplasm"]


def bin_labels(thresholds: list[float], names: list[str]) -> list[str]:
    """Labels for bins from ascending cut points: `names` if it fits, else ranges."""
    if len(names) == len(thresholds) + 1:
        return list(names)
    edges = [None] + list(thresholds) + [None]
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo is None:
            labels.append(f"<{hi:g}")
        elif hi is None:
            labels.append(f"≥{lo:g}")
        else:
            labels.append(f"{lo:g}–{hi:g}")
    return labels


def assign_threshold_bins(values: pd.Series, thresholds: list[float],
                          labels: list[str]) -> pd.Series:
    """Bin a numeric series by fixed absolute thresholds (half-open [lo, hi))."""
    edges = [-np.inf] + list(thresholds) + [np.inf]
    return pd.cut(values, bins=edges, labels=labels, right=False)


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def ckpt_slug(run_id: str, checkpoint: str) -> str:
    """Derive the predictions subdirectory name from run ID and checkpoint filename."""
    return f"{run_id}_{Path(checkpoint).stem}"


def load_predictions(path: Path) -> pd.DataFrame:
    """Load the test-split predictions produced by analysis/predict.py."""
    df = pd.read_pickle(path)
    df = df.rename(columns={"protein_id": "gene"})
    df["abs_error"] = (df["true_label"] - df["prediction"]).abs()
    return df[["gene", "true_label", "prediction", "abs_error"]]


def load_egenes(path: Path, qval_threshold: float) -> tuple[pd.DataFrame, str]:
    """
    Load the GTEx v8 eGenes table and keep significant eGenes only.

    Returns (dataframe, effect_col) where effect_col is the column used for the
    effect-size stratification (prefers log2_aFC, falls back to slope).
    """
    eg = pd.read_csv(path, sep="\t")
    if "gene_name" not in eg.columns:
        raise ValueError(f"eGenes file {path} has no 'gene_name' column; columns={list(eg.columns)}")
    if "qval" not in eg.columns:
        raise ValueError(f"eGenes file {path} has no 'qval' column; columns={list(eg.columns)}")

    effect_col = "log2_aFC" if "log2_aFC" in eg.columns else "slope"
    if effect_col not in eg.columns:
        raise ValueError(f"eGenes file {path} has neither 'log2_aFC' nor 'slope'; columns={list(eg.columns)}")

    n_total = len(eg)
    eg = eg[eg["qval"] <= qval_threshold].copy()
    eg["effect"] = eg[effect_col].abs()
    eg = eg.sort_values("qval").drop_duplicates(subset="gene_name", keep="first")
    print(f"eGenes: {len(eg):,} significant (qval<={qval_threshold}) of {n_total:,} tested "
          f"[effect column: {effect_col}]")
    return eg[["gene_name", "effect"]].rename(columns={"gene_name": "gene"}), effect_col


def load_category_map(path: Path) -> dict[str, set[str]]:
    """
    Load a two-column gene -> category TSV (e.g. gene_symbol <tab> compartment).

    A header row is auto-detected. Gene keys are upper-cased for case-insensitive
    matching. Returns {SYMBOL -> {categories}}.
    """
    gene2cat: dict[str, set[str]] = {}
    header_keys = ("gene", "gene_symbol", "symbol", "category", "compartment",
                   "localization", "term")
    with open(path) as fh:
        for i, raw in enumerate(fh):
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = [f.strip() for f in line.split("\t")] if "\t" in line else line.split()
            if len(fields) < 2:
                continue
            if i == 0 and any(h.lower() in header_keys for h in fields[:2]):
                continue
            gene, cat = fields[0], fields[1]
            gene2cat.setdefault(gene.upper(), set()).add(cat)
    if not gene2cat:
        raise ValueError(f"No gene->category mappings parsed from {path}")
    return gene2cat


def load_variance(path: Path, metric: str) -> dict[str, float]:
    """Load per-protein across-patient variability (gene -> value) for `metric`."""
    v = pd.read_csv(path, sep="\t")
    if "gene" not in v.columns or metric not in v.columns:
        raise ValueError(f"variance table {path} needs 'gene' and '{metric}' columns; "
                         f"columns={list(v.columns)}")
    v = v.dropna(subset=[metric]).drop_duplicates("gene")
    return dict(zip(v["gene"].astype(str).str.upper(), v[metric].astype(float)))


def assign_localization(genes: pd.Series, gene2loc: dict[str, set[str]],
                        priority: list[str]) -> pd.Series:
    """Assign each gene its highest-priority annotated compartment, else 'Other'."""
    rank = {c: i for i, c in enumerate(priority)}

    def pick(g) -> str:
        comps = [c for c in gene2loc.get(str(g).upper(), set()) if c in rank]
        return min(comps, key=lambda c: rank[c]) if comps else "Other"

    return genes.map(pick)


# ---------------------------------------------------------------------------
# Metrics + plotting
# ---------------------------------------------------------------------------

def stratum_metrics(sub: pd.DataFrame) -> dict[str, float]:
    """Prediction metrics within one stratum. Pearson/R2 need >=2 varying points."""
    out = {
        "n": int(len(sub)),
        "median_abs_error": float(sub["abs_error"].median()) if len(sub) else float("nan"),
        "mean_abs_error": float(sub["abs_error"].mean()) if len(sub) else float("nan"),
        "pearson": float("nan"),
        "r2": float("nan"),
    }
    if len(sub) >= 2 and sub["true_label"].nunique() > 1 and sub["prediction"].nunique() > 1:
        m = compute_all_metrics(sub["true_label"].values, sub["prediction"].values)
        out["pearson"], out["r2"] = m["pearson"], m["r2"]
    return out


def pick_representative(metrics: dict[str, dict[str, float]], groups: list[str],
                        k: int = 3, min_n: int = 10) -> list[str]:
    """
    Choose up to `k` groups spanning the correlation range (high / mid / low r)
    among groups with enough points, ordered strongest-to-weakest for display.
    """
    scored = [(g, metrics[g]["pearson"]) for g in groups
              if metrics[g]["n"] >= min_n and not np.isnan(metrics[g]["pearson"])]
    scored.sort(key=lambda gr: -gr[1])  # high -> low r
    if len(scored) <= k:
        return [g for g, _ in scored]
    picks = {0, (len(scored) - 1) // 2, len(scored) - 1} if k >= 3 else {0, len(scored) - 1}
    return [scored[i][0] for i in sorted(picks)]


def make_stratified_figure(df: pd.DataFrame, group_col: str, group_order: list[str],
                           scatter_order: list[str],
                           fill_map: dict[str, str], ink_map: dict[str, str],
                           box_title: str, out_stem: Path) -> pd.DataFrame:
    """
    Left: abs-error boxplot over all strata in `group_order`. Right: a vertical
    stack of short, wide obs-vs-pred scatter panels (one per group in
    `scatter_order`) sharing common x/y limits. Returns the per-stratum metrics.
    """
    # Drop "Other" and restrict df to only the plotted groups.
    group_order = [g for g in group_order if g != "Other"]
    df = df[df[group_col].isin(group_order)].copy()

    counts = {g: int((df[group_col] == g).sum()) for g in group_order}
    metrics = {g: stratum_metrics(df[df[group_col] == g]) for g in group_order}

    # Sort boxplot by median error ascending.
    group_order = sorted(group_order, key=lambda g: metrics[g]["median_abs_error"])

    k = len(scatter_order)
    fig = plt.figure(figsize=(11, 4.2))
    gs = fig.add_gridspec(k, 2, width_ratios=[1.0, 1.35], wspace=0.32, hspace=0.6)
    ax_box = fig.add_subplot(gs[:, 0])

    # --- Left: absolute error by stratum ---
    data = [df.loc[df[group_col] == g, "abs_error"].values for g in group_order]
    bp = ax_box.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
    for patch, g in zip(bp["boxes"], group_order):
        patch.set_facecolor(fill_map[g])
        patch.set_edgecolor(ink_map[g])
    for element in ("whiskers", "caps"):
        for ln in bp[element]:
            ln.set_color("#4a4a4a")
    for med in bp["medians"]:
        med.set_color("black")
    ax_box.set_xticklabels([f"{g}  (n={counts[g]})" for g in group_order], fontsize=8,
                           rotation=30, ha="right", rotation_mode="anchor")
    ax_box.set_ylabel("Absolute Error  |Obs − Pred|")
    ax_box.set_title(box_title, pad=15)

    # --- Right: stacked obs-vs-pred scatters, shared axes ---
    sc = df[df[group_col].isin(scatter_order)]
    lo = min(sc["true_label"].min(), sc["prediction"].min())
    hi = max(sc["true_label"].max(), sc["prediction"].max())
    pad = (hi - lo) * 0.05

    base = None
    for i, g in enumerate(scatter_order):
        ax = fig.add_subplot(gs[i, 1], sharex=base, sharey=base)
        base = base or ax
        s = df[df[group_col] == g]
        ax.plot([lo, hi], [lo, hi], color="gray", lw=1.0, ls="--", zorder=1)
        ax.scatter(s["true_label"], s["prediction"], color=ink_map[g],
                   s=12, alpha=0.6, linewidths=0, zorder=2)
        r = metrics[g]["pearson"]
        r_str = f"r={r:.2f}" if not np.isnan(r) else "r=NA"
        ax.set_title(f"{g}  (n={counts[g]}, {r_str})", fontsize=9, pad=3)
        if i < k - 1:
            ax.tick_params(labelbottom=False)
    base.set_xlim(lo - pad, hi + pad)
    base.set_ylim(lo - pad, hi + pad)
    fig.axes[-1].set_xlabel("Observed (z-score)")
    fig.axes[1 + k // 2].set_ylabel("Predicted (z-score)")

    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame([{"Stratum": g, **metrics[g]} for g in group_order])


def build_palette(order: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Group->fill and group->ink maps; 'Other' uses the neutral palette entry."""
    fill, ink = {}, {}
    for i, g in enumerate([g for g in order if g != "Other"]):
        fill[g] = GROUP_FILLS[i % len(GROUP_FILLS)]
        ink[g] = GROUP_INKS[i % len(GROUP_INKS)]
    if "Other" in order:
        fill["Other"], ink["Other"] = FILL["other"], INK["other"]
    return fill, ink


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--wandb_run_id",
                      help="WandB run ID for a PASTL model (predictions/<run>_<ckpt>/).")
    mode.add_argument("--enformer", action="store_true",
                      help="Use the Enformer elastic-net baseline (predictions/enformer/).")
    parser.add_argument("--checkpoint", default="best.pt",
                        help="Checkpoint filename; its stem forms the slug (default: best.pt). "
                             "Ignored when --enformer is set.")
    parser.add_argument("--egenes", required=True,
                        help="GTEx v8 lung eGenes table (.txt.gz).")
    parser.add_argument("--localization_map", default=_DEFAULT_LOCALIZATION,
                        help="gene->compartment TSV from build_localization_map.py.")
    parser.add_argument("--variance_table", default=_DEFAULT_VARIANCE,
                        help="gene,iqr,std TSV from build_variance_table.py.")
    parser.add_argument("--variance_metric", choices=["iqr", "std"], default="iqr",
                        help="Which across-patient variability column to bin (default: iqr).")
    parser.add_argument("--qval_threshold", type=float, default=0.05,
                        help="eGene significance cutoff defining the gene universe (default: 0.05).")
    parser.add_argument("--effect_thresholds", type=float, nargs="+",
                        default=_DEFAULT_EFFECT_THRESHOLDS,
                        help="Ascending |effect| cut points for cis-eQTL strength bins "
                             "(default: 0.3 0.6 -> Weak/Moderate/Strong on |log2_aFC|).")
    parser.add_argument("--variance_thresholds", type=float, nargs="+",
                        default=_DEFAULT_VARIANCE_THRESHOLDS,
                        help="Ascending cut points for across-patient variability bins "
                             "(default: 0.3 0.6 -> Low/Mid/High on IQR of log2 abundance).")
    parser.add_argument("--predictions_dir", default=_PREDICTIONS_DIR,
                        help="Root predictions directory (default: %(default)s).")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory; caller constructs the slug-based path "
                             "(e.g. .../analysis/<run>_<ckpt>/egene_eval/).")
    args = parser.parse_args()

    for name, thr in (("--effect_thresholds", args.effect_thresholds),
                      ("--variance_thresholds", args.variance_thresholds)):
        if any(b <= a for a, b in zip(thr, thr[1:])):
            parser.error(f"{name} must be ascending, got {thr}")

    style_path = Path(__file__).parent.parent.parent / "pastl.mplstyle"
    if style_path.exists():
        plt.style.use(str(style_path))

    out_dir = Path(args.output_dir)
    figs_dir, tables_dir, logs_dir = out_dir / "figures", out_dir / "tables", out_dir / "logs"
    for d in (figs_dir, tables_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- Load test-set predictions ---
    if args.enformer:
        slug = "enformer"
        pred_path = Path(args.predictions_dir) / "enformer" / "predictions.pkl"
    else:
        slug = ckpt_slug(args.wandb_run_id, args.checkpoint)
        pred_path = Path(args.predictions_dir) / slug / "predictions.pkl"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions at {pred_path}")
    preds = load_predictions(pred_path)
    print(f"Predictions: {len(preds):,} test-set genes from {pred_path}")

    # --- Build the eGene universe (significant eGenes present in the test set) ---
    eg, effect_col = load_egenes(Path(args.egenes), args.qval_threshold)
    uni = preds.merge(eg, on="gene", how="inner")
    if uni.empty:
        raise ValueError("No overlap between test-set predictions and significant eGenes "
                         "(check that protein_id matches gene_name symbols).")
    uni["gene_u"] = uni["gene"].astype(str).str.upper()
    print(f"Universe: {len(uni):,} test-set eGenes matched to predictions")

    # --- Axis 1: cis-eQTL effect-size bins ---
    eff_thr = args.effect_thresholds
    if effect_col != "log2_aFC":
        print(f"NOTE: effect column is '{effect_col}', not log2_aFC; default thresholds "
              f"{eff_thr} are calibrated for |log2_aFC| -- adjust --effect_thresholds.")
    eff_labels = bin_labels(eff_thr, ["Weak", "Moderate", "Strong"])
    uni["effect_bin"] = assign_threshold_bins(uni["effect"], eff_thr, eff_labels).astype("object")
    uni.loc[uni["effect"].isna(), "effect_bin"] = "NA"
    eff_order = [lab for lab in eff_labels if (uni["effect_bin"] == lab).any()]

    # --- Axis 2: subcellular localization (GO Cellular Component spectrum) ---
    gene2loc = load_category_map(Path(args.localization_map))
    uni["localization"] = assign_localization(uni["gene_u"], gene2loc, _LOCALIZATION_PRIORITY)
    loc_order = [c for c in _LOCALIZATION_ORDER if (uni["localization"] == c).any()]
    loc_other = float((uni["localization"] == "Other").mean())
    print(f"Localization coverage: {1 - loc_other:.0%} assigned, {loc_other:.0%} 'Other' "
          f"across {len([c for c in loc_order if c != 'Other'])} compartments")
    if loc_other > 0.5:
        print("  WARNING: >50% 'Other' -- check symbol matching / rebuild localization map.")

    # --- Axis 3: across-patient variability bins ---
    var_map = load_variance(Path(args.variance_table), args.variance_metric)
    uni["variance"] = uni["gene_u"].map(var_map)
    var_thr = args.variance_thresholds
    var_labels = bin_labels(var_thr, ["Low", "Mid", "High"])
    uni["variance_bin"] = assign_threshold_bins(uni["variance"], var_thr, var_labels).astype("object")
    uni.loc[uni["variance"].isna(), "variance_bin"] = "NA"
    var_order = [lab for lab in var_labels if (uni["variance_bin"] == lab).any()]
    var_missing = float(uni["variance"].isna().mean())
    if var_missing:
        print(f"Variance: {var_missing:.0%} of universe missing a {args.variance_metric} value")

    # --- Figures ---
    eff_thr_str = " / ".join(f"{t:g}" for t in eff_thr)
    var_thr_str = " / ".join(f"{t:g}" for t in var_thr)
    eff_fill, eff_ink = build_palette(eff_labels)
    loc_fill, loc_ink = build_palette(loc_order)
    var_fill, var_ink = build_palette(var_labels)

    # Effect size: stack all bins.
    eff_metrics = make_stratified_figure(
        uni[uni["effect_bin"].isin(eff_order)], "effect_bin", eff_order, eff_order,
        eff_fill, eff_ink,
        f"Error by cis-eQTL Strength\n(|{effect_col}| cutoffs: {eff_thr_str})",
        figs_dir / "effectsize_stratification",
    )
    # Localization: boxplot over the full compartment spectrum, scatters for a
    # representative high/mid/low-correlation subset (Nuclear vs Extracellular story).
    loc_metric_map = {g: stratum_metrics(uni[uni["localization"] == g]) for g in loc_order}
    loc_scatter = pick_representative(loc_metric_map, [g for g in loc_order if g != "Other"], k=3)
    loc_metrics = make_stratified_figure(
        uni, "localization", loc_order, loc_scatter, loc_fill, loc_ink,
        "Error by Subcellular Localization\n(GO Cellular Component)",
        figs_dir / "localization_stratification",
    )
    print(f"Localization scatter panels (representative): {loc_scatter}")
    # Across-patient variability: stack all bins.
    var_metrics = make_stratified_figure(
        uni[uni["variance_bin"].isin(var_order)], "variance_bin", var_order, var_order,
        var_fill, var_ink,
        f"Error by Across-Patient Variability\n({args.variance_metric.upper()} of log2 "
        f"abundance, cutoffs: {var_thr_str})",
        figs_dir / "variance_stratification",
    )

    # --- Tables + log ---
    eff_metrics.to_csv(tables_dir / "effectsize_metrics.csv", index=False)
    loc_metrics.to_csv(tables_dir / "localization_metrics.csv", index=False)
    var_metrics.to_csv(tables_dir / "variance_metrics.csv", index=False)
    uni.drop(columns=["gene_u"]).sort_values("abs_error").to_csv(
        tables_dir / "gene_annotations.csv", index=False)

    overall = compute_all_metrics(uni["true_label"].values, uni["prediction"].values)
    with open(logs_dir / "egene_summary.txt", "w") as f:
        f.write("=== eGene Stratified Evaluation ===\n")
        f.write(f"Run/checkpoint slug: {slug}\n")
        f.write(f"eGenes file:         {args.egenes}\n")
        f.write(f"qval threshold:      {args.qval_threshold}\n")
        f.write(f"Effect-size column:  {effect_col} (|abs|), cutoffs {eff_thr}\n")
        f.write(f"Variance metric:     {args.variance_metric}, cutoffs {var_thr}\n")
        f.write(f"Universe size:       {len(uni)} test-set eGenes\n")
        f.write(f"Overall  R2={overall['r2']:.4f}  Pearson={overall['pearson']:.4f}\n\n")
        f.write(f"-- cis-eQTL effect-size bins (|{effect_col}| cutoffs {eff_thr_str}) --\n")
        f.write(eff_metrics.to_string(index=False) + "\n\n")
        f.write("-- subcellular localization (GO CC) --\n")
        f.write(loc_metrics.to_string(index=False) + "\n\n")
        f.write(f"-- across-patient variability ({args.variance_metric} cutoffs {var_thr_str}) --\n")
        f.write(var_metrics.to_string(index=False) + "\n")

    print(f"\nSaved to {out_dir}")
    print("\nEffect-size bins:\n" + eff_metrics.to_string(index=False))
    print("\nLocalization:\n" + loc_metrics.to_string(index=False))
    print("\nAcross-patient variability:\n" + var_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
