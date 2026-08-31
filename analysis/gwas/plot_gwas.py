r"""
GWAS variant-effect figures: PASTL vs base Enformer on LUAD GWAS variants.

Loads the per-variant GWAS predictions written by analysis/predict_variants.py
(which scored the VEP-annotated manifest, so every row already carries P_VALUE,
TSS, TARGET_GENE and consequence_class) and produces four figures:

  1. gwas_enrichment            fold-enrichment of GWAS-significant hits among the
                                top-N PASTL variants + cumulative GWAS-hit recall.
  2. two_model_venn             PASTL vs Enformer matched top-K variant sets,
                                sized by GWAS-significant capture per region.
  3. {model}_class_x_distance   predicted effect by consequence class, stratified
                                by distance to the target-gene TSS (median±IQR
                                point plot + median heatmap), for each model.
  4. {model}_residual_by_class  per-class |residual| effect after regressing out
                                log distance-to-TSS, for each model.

All variant-level figures use the physically-deduped table (one row per unique
CHROM/POS/REF/ALT, keeping the highest-magnitude window), so genes near several
windows aren't double-counted.

Usage:
  python plot_gwas.py --wandb_run_id s5fo3aak --checkpoint best.pt \
      --output_dir out/analysis/s5fo3aak_best/gwas_eval/
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import brentq
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import palette
from gwas_config import (GWAS_SIG_THRESHOLD, CLASS_ORDER, CLASS_PALETTE, DISTANCE_BINS,
                         DISTANCE_LABELS, DISTANCE_INKS, PASTL_FILL, PASTL_INK,
                         ENFORMER_FILL, ENFORMER_INK, heatmap_cmap, style_axes)

_PREDICTIONS_DIR = "out/predictions"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def ckpt_slug(run_id: str, checkpoint: str) -> str:
    return f"{run_id}_{Path(checkpoint).stem}"


def load_scored(path: Path, score_col: str) -> pd.DataFrame:
    """Load a scored GWAS prediction TSV; add abs score, GWAS-sig flag, TSS distance."""
    df = pd.read_csv(path, sep="\t")
    df["abs_score"] = df[score_col].abs()
    df["IS_GWAS_SIG"] = df["P_VALUE"] < GWAS_SIG_THRESHOLD
    df["distance_to_TSS"] = (df["POS"].astype(int) - df["TSS"].astype(int)).abs()
    if "consequence_class" not in df.columns:
        df["consequence_class"] = "unannotated"
    return df


def merge_consequence(df: pd.DataFrame, annotated_path: Path) -> pd.DataFrame:
    """
    Attach consequence_class from the VEP-annotated manifest, joined on the
    physical variant + assigned gene. Lets the (expensive) prediction step score
    the plain manifest and keeps VEP an independent, model-free artifact — so
    adding/refreshing consequences never means re-running the GPU scoring.
    """
    if "consequence_class" in df.columns and (df["consequence_class"] != "unannotated").any():
        return df   # predictions already carry consequences
    if not annotated_path.exists():
        print(f"  no annotated manifest at {annotated_path}; consequence figures will be skipped")
        return df
    ann = pd.read_csv(annotated_path, sep="\t",
                      usecols=lambda c: c in {"CHROM", "POS", "REF", "ALT", "TARGET_GENE",
                                              "consequence_class", "primary_consequence"})
    keys = ["CHROM", "POS", "REF", "ALT", "TARGET_GENE"]
    df = df.drop(columns=[c for c in ("consequence_class", "primary_consequence")
                          if c in df.columns], errors="ignore")
    merged = df.merge(ann.drop_duplicates(keys), on=keys, how="left")
    merged["consequence_class"] = merged["consequence_class"].fillna("unannotated")
    return merged


def dedup_variants(df: pd.DataFrame) -> pd.DataFrame:
    """One row per physical variant, keeping the highest |score| window."""
    return (df.sort_values("abs_score", ascending=False)
              .drop_duplicates(subset=["CHROM", "POS", "REF", "ALT"])
              .reset_index(drop=True))


def variant_key(df: pd.DataFrame) -> pd.Series:
    return (df["CHROM"].astype(str) + "_" + df["POS"].astype(str) + "_"
            + df["REF"].astype(str) + "_" + df["ALT"].astype(str))


# ---------------------------------------------------------------------------
# Figure 1 — GWAS risk-variant enrichment
# ---------------------------------------------------------------------------

def figure_enrichment(tl: pd.DataFrame, figs_dir: Path, tables_dir: Path):
    total = len(tl)
    n_sig = int(tl["IS_GWAS_SIG"].sum())
    rate = n_sig / total if total else 0.0
    print(f"Enrichment: {n_sig:,}/{total:,} GWAS-sig ({100 * rate:.3f}% baseline)")

    ranked = tl.sort_values("abs_score", ascending=False).reset_index(drop=True)
    thresholds = [t for t in (250_000, 500_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000)
                  if t <= total]
    if not thresholds:  # small run — use fractional cut points
        thresholds = [int(total * f) for f in (0.05, 0.1, 0.25, 0.5) if int(total * f) > 0]

    rows = []
    for k in thresholds:
        hits = int(ranked.iloc[:k]["IS_GWAS_SIG"].sum())
        expected = k * rate
        rows.append({"Threshold": f"{k // 1000}K" if k >= 1000 else str(k),
                     "Top_K": k, "Sig_Hits": hits, "Expected": round(expected, 1),
                     "Fold_Enrichment": hits / expected if expected > 0 else 0.0})
    enrich = pd.DataFrame(rows)
    enrich.to_csv(tables_dir / "gwas_enrichment.tsv", sep="\t", index=False,
                  float_format="%.4f")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].bar(enrich["Threshold"], enrich["Fold_Enrichment"],
                color=PASTL_FILL, width=palette.BAR_WIDTH, zorder=3)
    axes[0].set_xlim(*palette.bar_limits(len(enrich)))
    axes[0].axhline(1.0, color="gray", ls="--", lw=0.5, alpha=0.6, zorder=2)
    axes[0].set_title("GWAS Risk Variant Enrichment")
    axes[0].set_ylabel("Fold Enrichment (Observed / Expected)")
    axes[0].set_xlabel("PASTL Prioritization Threshold (Top N)")
    style_axes(axes[0])

    pct = (np.arange(1, total + 1) / total) * 100
    cum = ranked["IS_GWAS_SIG"].cumsum().values
    axes[1].plot(pct, (np.arange(1, total + 1) / total) * n_sig, color="gray",
                 ls="--", lw=0.8, alpha=0.6, label="Random Expectation", zorder=2)
    axes[1].plot(pct, cum, color=PASTL_INK, lw=0.9, label="PASTL Capture", zorder=3)
    axes[1].set_title("Cumulative GWAS Hit Recall")
    axes[1].set_xlabel("Percentage of Dataset Ranked (%)")
    axes[1].set_ylabel("Cumulative GWAS Hits Captured")
    axes[1].legend(frameon=False)
    style_axes(axes[1])

    fig.savefig(figs_dir / "gwas_enrichment.png", dpi=300, bbox_inches="tight")
    fig.savefig(figs_dir / "gwas_enrichment.pdf", bbox_inches="tight")
    plt.close(fig)
    return enrich


# ---------------------------------------------------------------------------
# Figure 2 — PASTL vs Enformer matched top-K Venn
# ---------------------------------------------------------------------------

def _circle_overlap_area(d, r1, r2):
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return np.pi * min(r1, r2) ** 2
    d = np.clip(d, abs(r1 - r2) + 1e-9, r1 + r2 - 1e-9)
    t1 = r1 ** 2 * np.arccos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
    t2 = r2 ** 2 * np.arccos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
    t3 = 0.5 * np.sqrt(max((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2), 0))
    return t1 + t2 - t3


def _center_distance(target_area, r1, r2):
    if target_area <= 0:
        return r1 + r2
    if target_area >= np.pi * min(r1, r2) ** 2:
        return abs(r1 - r2)
    return brentq(lambda d: _circle_overlap_area(d, r1, r2) - target_area,
                  abs(r1 - r2) + 1e-6, r1 + r2 - 1e-6)


def _draw_venn(ax, tl_only_sig, enf_only_sig, overlap_sig, title):
    tl_total = tl_only_sig + overlap_sig
    enf_total = enf_only_sig + overlap_sig
    max_sig = max(tl_total, enf_total, 1)
    r_tl = max(tl_total / max_sig, 0.05)
    r_enf = max(enf_total / max_sig, 0.05)
    r_min = min(tl_total, enf_total)
    overlap_frac = (overlap_sig / r_min) if r_min > 0 else 0
    d = _center_distance(overlap_frac * np.pi * min(r_tl, r_enf) ** 2, r_tl, r_enf)

    cx_tl, cx_enf = -d / 2, d / 2
    ax.add_patch(plt.Circle((cx_tl, 0), r_tl, color=PASTL_FILL, alpha=0.45, ec="black", lw=0.5, zorder=2))
    ax.add_patch(plt.Circle((cx_enf, 0), r_enf, color=ENFORMER_FILL, alpha=0.45, ec="black", lw=0.5, zorder=2))
    ax.text(cx_tl - r_tl * 0.65, 0, f"{tl_only_sig:,}", ha="center", va="center", fontsize=9)
    ax.text(cx_enf + r_enf * 0.65, 0, f"{enf_only_sig:,}", ha="center", va="center", fontsize=9)
    if overlap_frac > 0.02:
        ax.text(0, 0, f"{overlap_sig:,}", ha="center", va="center", fontsize=9)

    r_max = max(r_tl, r_enf)
    extent = max(r_max, d / 2 + r_tl, d / 2 + r_enf)
    ax.set_xlim(-extent * 1.3, extent * 1.3)
    ax.set_ylim(-r_max * 1.3, r_max * 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0, r_max * 1.08, title, ha="center", va="bottom", fontsize=10)


def figure_venn(tl: pd.DataFrame, enf: pd.DataFrame, figs_dir: Path, tables_dir: Path):
    tl_k = tl.copy(); tl_k["_key"] = variant_key(tl_k)
    enf_k = enf.copy(); enf_k["_key"] = variant_key(enf_k)
    shared = set(tl_k["_key"]) & set(enf_k["_key"])
    tl_k = tl_k[tl_k["_key"].isin(shared)].reset_index(drop=True)
    enf_k = enf_k[enf_k["_key"].isin(shared)].reset_index(drop=True)
    print(f"Venn shared universe: {len(shared):,} variants")

    sig_keys = set(tl_k.loc[tl_k["IS_GWAS_SIG"], "_key"])
    tl_rank = tl_k.sort_values("abs_score", ascending=False)["_key"].tolist()
    enf_rank = enf_k.sort_values("abs_score", ascending=False)["_key"].tolist()

    thresholds = [k for k in (250_000, 500_000, 1_000_000) if k <= len(shared)]
    if not thresholds:
        thresholds = [int(len(shared) * f) for f in (0.1, 0.25, 0.5) if int(len(shared) * f) > 0]

    rows = []
    for k in thresholds:
        tl_top, enf_top = set(tl_rank[:k]), set(enf_rank[:k])
        only_tl, only_enf, overlap = tl_top - enf_top, enf_top - tl_top, tl_top & enf_top
        rows.append({"Threshold": f"{k // 1000}K" if k >= 1000 else str(k), "K": k,
                     "TL_only_sig": len(only_tl & sig_keys),
                     "Enformer_only_sig": len(only_enf & sig_keys),
                     "Overlap_sig": len(overlap & sig_keys)})
    venn = pd.DataFrame(rows)
    venn.to_csv(tables_dir / "two_model_venn.tsv", sep="\t", index=False)

    n = len(venn)
    if n == 0:
        print("  Venn: empty shared universe or no valid top-K threshold — skipping figure")
        return venn
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.8), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (_, r) in zip(axes, venn.iterrows()):
        _draw_venn(ax, r["TL_only_sig"], r["Enformer_only_sig"], r["Overlap_sig"],
                   f"Top {r['Threshold']}")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=PASTL_FILL, alpha=0.7, label="PASTL"),
                        plt.Rectangle((0, 0), 1, 1, color=ENFORMER_FILL, alpha=0.7, label="Enformer")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("PASTL vs. Enformer top-K variant sets (GWAS-significant hits per region)")
    fig.savefig(figs_dir / "two_model_venn.png", dpi=300, bbox_inches="tight")
    fig.savefig(figs_dir / "two_model_venn.pdf", bbox_inches="tight")
    plt.close(fig)
    return venn


# ---------------------------------------------------------------------------
# Figures 3 & 4 — consequence class × distance-to-TSS (per model)
# ---------------------------------------------------------------------------

def _annotated_with_bins(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    ann = df[df["consequence_class"] != "unannotated"].copy()
    ann["distance_bin"] = pd.cut(ann["distance_to_TSS"], bins=DISTANCE_BINS,
                                 labels=DISTANCE_LABELS, include_lowest=True)
    present = [c for c in CLASS_ORDER
               if c not in ("unannotated",) and (ann["consequence_class"] == c).sum() >= 10]
    return ann, present


def figure_class_distance(ann: pd.DataFrame, present: list[str], metric_label: str,
                          out_stem: Path):
    agg = (ann.groupby(["distance_bin", "consequence_class"], observed=True)["abs_score"]
             .agg(median="median", q25=lambda x: x.quantile(0.25),
                  q75=lambda x: x.quantile(0.75), n="count").reset_index())
    agg = agg[agg["n"] >= 10]

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    width = 0.18
    for i, dbin in enumerate(DISTANCE_LABELS):
        sub = agg[agg["distance_bin"] == dbin].set_index("consequence_class")
        xs, meds, lo, hi = [], [], [], []
        for j, cls in enumerate(present):
            if cls not in sub.index:
                continue
            xs.append(j + (i - (len(DISTANCE_LABELS) - 1) / 2) * width)
            meds.append(sub.loc[cls, "median"])
            lo.append(sub.loc[cls, "median"] - sub.loc[cls, "q25"])
            hi.append(sub.loc[cls, "q75"] - sub.loc[cls, "median"])
        ax.errorbar(xs, meds, yerr=[lo, hi], fmt="o", color=DISTANCE_INKS[i % len(DISTANCE_INKS)],
                    label=dbin, capsize=2, markersize=5, lw=1, zorder=5)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=35, ha="right")
    ax.set_ylabel(f"{metric_label} (median ± IQR)")
    ax.set_ylim(bottom=0)
    ax.legend(title="Distance from TSS", loc="upper right", frameon=True)
    ax.set_title("Predicted effect by consequence class, stratified by distance from TSS")
    style_axes(ax)
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

def figure_class_distance_heatmap(ann: pd.DataFrame, present: list[str], metric_label: str,
                                  out_stem: Path, class_order=None):
    table = (ann.groupby(["consequence_class", "distance_bin"], observed=True)["abs_score"]
             .median().unstack().reindex(index=present, columns=DISTANCE_LABELS))

    if class_order is None:
        # Sort classes by median magnitude at the smallest distance bin descending
        first_bin = DISTANCE_LABELS[0]
        sorted_classes = table[first_bin].sort_values(ascending=False, na_position='last').index.tolist()
    else:
        sorted_classes = [c for c in class_order if c in table.index]
        sorted_classes += [c for c in table.index if c not in sorted_classes]

    table = table.reindex(index=sorted_classes)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    data = table.values.astype(float)

    cmap = heatmap_cmap()
    if "pastl" in out_stem.name.lower():
        model_name = "PASTL"
    elif "enformer" in out_stem.name.lower():
        model_name = "Enformer"
    else:
        model_name = "Model"

    im = ax.imshow(data, cmap=cmap, aspect="equal")

    ax.set_xticks(range(len(DISTANCE_LABELS)))
    ax.set_xticklabels(DISTANCE_LABELS, rotation=90, ha="center", fontsize=8)
    ax.set_yticks(range(len(sorted_classes)))
    ax.set_yticklabels(sorted_classes, fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Median {metric_label}", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    ax.set_title(f"{model_name} median predicted effect by consequence\nclass and distance from TSS")

    style_axes(ax)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False)

    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return sorted_classes


def _compute_residuals(ann: pd.DataFrame) -> pd.DataFrame:
    mask = ann["distance_to_TSS"].notna() & ann["abs_score"].notna() & (ann["distance_to_TSS"] > 0)
    resid = ann[mask].copy()
    resid["log_distance"] = np.log10(resid["distance_to_TSS"])
    ols = LinearRegression().fit(resid[["log_distance"]].values, resid["abs_score"].values)
    resid["abs_residual"] = np.abs(resid["abs_score"].values - ols.predict(resid[["log_distance"]].values))
    return resid


def figure_residual(ann: pd.DataFrame, present: list[str], metric_label: str,
                    model_ink: str, out_stem: Path, tables_dir: Path, prefix: str):
    resid = _compute_residuals(ann)

    summary = (resid.groupby("consequence_class")["abs_residual"]
                    .agg(median="median", q25=lambda x: x.quantile(0.25),
                         q75=lambda x: x.quantile(0.75)).reindex(present))
    summary.to_csv(tables_dir / f"{prefix}_residual_by_class.tsv", sep="\t", float_format="%.6f")

    xs, meds, lo, hi = [], [], [], []
    for j, cls in enumerate(present):
        if cls not in summary.index or pd.isna(summary.loc[cls, "median"]):
            continue
        xs.append(j)
        meds.append(summary.loc[cls, "median"])
        lo.append(summary.loc[cls, "median"] - summary.loc[cls, "q25"])
        hi.append(summary.loc[cls, "q75"] - summary.loc[cls, "median"])

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.errorbar(xs, meds, yerr=[lo, hi], fmt="o", color=model_ink, capsize=3,
                markersize=5, lw=1, zorder=5)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=35, ha="right")
    ax.set_ylabel(f"|Residual {metric_label}| (after regressing out log distance)")
    ax.set_title("Per-class |residual| effect after distance correction")
    style_axes(ax)
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def figure_combined_residual(tl_ann: pd.DataFrame, tl_present: list[str],
                             enf_ann: pd.DataFrame, enf_present: list[str],
                             figs_dir: Path, tables_dir: Path):
    present = [c for c in CLASS_ORDER if c in tl_present and c in enf_present]
    if not present:
        print("  combined residual: no shared consequence classes — skipping")
        return

    tl_resid = _compute_residuals(tl_ann)
    enf_resid = _compute_residuals(enf_ann)

    def summarise(resid, classes):
        return (resid.groupby("consequence_class")["abs_residual"]
                .agg(median="median", q25=lambda x: x.quantile(0.25),
                     q75=lambda x: x.quantile(0.75)).reindex(classes))

    tl_sum = summarise(tl_resid, present)
    enf_sum = summarise(enf_resid, present)

    # Normalize by the max q75 across present classes so the top whisker
    # of each model reaches 1.0, avoiding outlier variants deflating the scale.
    for df in (tl_sum, enf_sum):
        scale = df["q75"].max()
        if scale > 0:
            df[["median", "q25", "q75"]] /= scale

    # Sort consequence classes by PASTL's median delta (lowest to highest)
    present = tl_sum["median"].sort_values(ascending=True).index.tolist()

    for df, prefix in [(tl_sum, "pastl"), (enf_sum, "enformer")]:
        df.to_csv(tables_dir / f"{prefix}_relative_residual_by_class.tsv", sep="\t",
                  float_format="%.6f")

    # Compressed offset for tighter line grouping
    offset = 0.10
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for j, cls in enumerate(present):
        for y_off, summary, ink, label in [
            (offset, tl_sum, PASTL_INK, "PASTL"),
            (-offset, enf_sum, ENFORMER_INK, "Enformer"),
        ]:
            if cls not in summary.index or pd.isna(summary.loc[cls, "median"]):
                continue
            med = summary.loc[cls, "median"]
            lo = med - summary.loc[cls, "q25"]
            hi = summary.loc[cls, "q75"] - med
            # Plot horizontally (x=med, y=j+y_off)
            ax.errorbar(med, j + y_off, xerr=[[lo], [hi]], fmt="o", color=ink,
                        capsize=0, markersize=5, lw=1, zorder=5,
                        label=label if j == 0 else None)

    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present)
    ax.set_xlabel("|Residual| / max(|Residual|)  (relative, per model)")
    ax.set_title("Per-class |residual| effect after distance correction")
    ax.legend(loc="lower right", frameon=False)
    style_axes(ax)
    out_stem = figs_dir / "combined_residual_by_class"
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def per_model_distance_figures(df_dedup, metric_label, model_ink, prefix, figs_dir, tables_dir, class_order=None):
    ann, present = _annotated_with_bins(df_dedup)
    if not present:
        print(f"  [{prefix}] no consequence classes with >=10 annotated variants — skipping")
        return ann, present, None
    figure_class_distance(ann, present, metric_label, figs_dir / f"{prefix}_class_x_distance")
    order = figure_class_distance_heatmap(ann, present, metric_label,
                                          figs_dir / f"{prefix}_class_x_distance_heatmap", class_order)
    figure_residual(ann, present, metric_label, model_ink,
                    figs_dir / f"{prefix}_residual_by_class", tables_dir, prefix)
    return ann, present, order


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wandb_run_id", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--predictions_dir", default=_PREDICTIONS_DIR)
    parser.add_argument("--annotated_manifest", default=None,
                        help="VEP-annotated GWAS manifest for consequence classes "
                             "(default: {predictions_dir}/variants/gwas_manifest_annotated.tsv).")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    palette.use_style()

    out_dir = Path(args.output_dir)
    figs_dir, tables_dir, logs_dir = out_dir / "figures", out_dir / "tables", out_dir / "logs"
    for d in (figs_dir, tables_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    pred_dir = Path(args.predictions_dir)
    slug = ckpt_slug(args.wandb_run_id, args.checkpoint)
    tl_path = pred_dir / slug / "variants" / "gwas.tsv"
    enf_path = pred_dir / "enformer" / "variants" / "gwas.tsv"
    for p in (tl_path, enf_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing GWAS predictions at {p}")

    annotated_path = (Path(args.annotated_manifest) if args.annotated_manifest
                      else pred_dir / "variants" / "gwas_manifest_annotated.tsv")
    tl = merge_consequence(load_scored(tl_path, "DELTA_Z"), annotated_path)
    enf = merge_consequence(load_scored(enf_path, "LOG2FC"), annotated_path)
    tl_dedup = dedup_variants(tl)
    enf_dedup = dedup_variants(enf)
    print(f"PASTL: {len(tl):,} rows -> {len(tl_dedup):,} variants; "
          f"Enformer: {len(enf):,} rows -> {len(enf_dedup):,} variants")

    enrich = figure_enrichment(tl_dedup, figs_dir, tables_dir)
    venn = figure_venn(tl_dedup, enf_dedup, figs_dir, tables_dir)
    tl_ann, tl_present, tl_order = per_model_distance_figures(tl_dedup, r"$|\Delta Z|$", PASTL_INK, "pastl", figs_dir, tables_dir)
    enf_ann, enf_present, _ = per_model_distance_figures(enf_dedup, r"$|\log_2 \mathrm{FC}|$", ENFORMER_INK, "enformer",
                                                       figs_dir, tables_dir, class_order=tl_order)
    if tl_present and enf_present:
        figure_combined_residual(tl_ann, tl_present, enf_ann, enf_present, figs_dir, tables_dir)

    with open(logs_dir / "gwas_summary.txt", "w") as f:
        f.write("=== GWAS Variant-Effect Analysis ===\n")
        f.write(f"Run/checkpoint slug: {slug}\n")
        f.write(f"GWAS-sig threshold:  p < {GWAS_SIG_THRESHOLD:.0e}\n")
        f.write(f"PASTL variants:    {len(tl_dedup):,}\n")
        f.write(f"Enformer variants:   {len(enf_dedup):,}\n\n")
        f.write("-- top-N enrichment --\n" + enrich.to_string(index=False) + "\n\n")
        f.write("-- matched top-K venn (GWAS-sig per region) --\n" + venn.to_string(index=False) + "\n")

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
