r"""
GWAS distance-residualized enrichment figures: PASTL vs Enformer.

Reproduces the fold-enrichment bar chart and cumulative recall curve from
plot_gwas.py, but ranks variants by distance-residualized scores instead of
raw |DELTA_Z| / |LOG2FC|.  The residualization fits OLS:

    abs_score ~ log10(distance_to_TSS + 1)

and ranks by the signed residual (abs_score - predicted).

Usage:
  python plot_gwas_residualized.py --wandb_run_id s5fo3aak --checkpoint best.pt \
      --output_dir out/analysis/s5fo3aak_best/gwas_eval/
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import palette
from gwas_config import (GWAS_SIG_THRESHOLD, PASTL_FILL, PASTL_INK,
                         ENFORMER_FILL, ENFORMER_INK, style_axes)
from plot_gwas import ckpt_slug, load_scored, dedup_variants, variant_key, _draw_venn

_PREDICTIONS_DIR = "out/predictions"


def _residualize(df: pd.DataFrame) -> tuple[pd.DataFrame, LinearRegression, float]:
    """Add residualized_score; return augmented df, fitted OLS, and R²."""
    out = df.copy()
    out["log_dist"] = np.log10(out["distance_to_TSS"].clip(lower=0) + 1)
    valid = out["log_dist"].notna() & out["abs_score"].notna()
    X = out.loc[valid, ["log_dist"]].values
    y = out.loc[valid, "abs_score"].values
    ols = LinearRegression().fit(X, y)
    r2 = float(ols.score(X, y))
    out["ols_pred"] = np.nan
    out.loc[valid, "ols_pred"] = ols.predict(X)
    out["residualized_score"] = (out["abs_score"] - out["ols_pred"]).fillna(out["abs_score"])
    return out, ols, r2


def _enrichment_rows(ranked: pd.DataFrame, rate: float, label: str) -> list[dict]:
    total = len(ranked)
    all_ths = [250_000, 500_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000]
    ths = [t for t in all_ths if t <= total]
    if not ths:
        ths = [int(total * f) for f in (0.05, 0.1, 0.25, 0.5) if int(total * f) > 0]
    rows = []
    for k in ths:
        hits = int(ranked.iloc[:k]["IS_GWAS_SIG"].sum())
        expected = k * rate
        rows.append({
            "Model": label,
            "Threshold": f"{k // 1000}K" if k >= 1000 else str(k),
            "Top_K": k,
            "Sig_Hits": hits,
            "Expected": round(expected, 1),
            "Fold_Enrichment": hits / expected if expected > 0 else 0.0,
        })
    return rows


def figure_residualized_enrichment(tl: pd.DataFrame, enf: pd.DataFrame,
                                   figs_dir: Path, tables_dir: Path) -> pd.DataFrame:
    """Both dataframes must already have a residualized_score column (from _residualize)."""
    tl_ranked  = tl.sort_values("residualized_score",  ascending=False).reset_index(drop=True)
    enf_ranked = enf.sort_values("residualized_score", ascending=False).reset_index(drop=True)

    tl_total  = len(tl_ranked)
    enf_total = len(enf_ranked)
    tl_sig_n  = int(tl_ranked["IS_GWAS_SIG"].sum())
    enf_sig_n = int(enf_ranked["IS_GWAS_SIG"].sum())
    tl_rate   = tl_sig_n  / tl_total  if tl_total  else 0.0
    enf_rate  = enf_sig_n / enf_total if enf_total else 0.0

    print(f"\nBaseline GWAS-sig rate:")
    print(f"  PASTL   {tl_sig_n:,}/{tl_total:,} ({100*tl_rate:.3f}%)")
    print(f"  Enformer  {enf_sig_n:,}/{enf_total:,} ({100*enf_rate:.3f}%)")

    tl_rows  = _enrichment_rows(tl_ranked,  tl_rate,  "PASTL")
    enf_rows = _enrichment_rows(enf_ranked, enf_rate, "Enformer")
    enrich   = pd.DataFrame(tl_rows + enf_rows)
    enrich.to_csv(tables_dir / "gwas_enrichment_residualized.tsv", sep="\t", index=False,
                  float_format="%.4f")

    print("\nFold enrichment (distance-residualized):")
    print(enrich.to_string(index=False))

    # ── ordered threshold labels for bar chart ──────────────────────────────
    def label_key(s):
        return int(s.rstrip("K")) * (1000 if s.endswith("K") else 1)

    all_labels = sorted(set(r["Threshold"] for r in tl_rows + enf_rows), key=label_key)
    tl_fe  = {r["Threshold"]: r["Fold_Enrichment"] for r in tl_rows}
    enf_fe = {r["Threshold"]: r["Fold_Enrichment"] for r in enf_rows}

    x = np.arange(len(all_labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    # ── subplot 1: fold-enrichment bar chart ────────────────────────────────
    axes[0].bar(x - width / 2, [tl_fe.get(t, 0.0) for t in all_labels],
                width, label="PASTL",  color=PASTL_FILL,  zorder=3)
    axes[0].bar(x + width / 2, [enf_fe.get(t, 0.0) for t in all_labels],
                width, label="Enformer", color=ENFORMER_FILL, zorder=3)
    axes[0].axhline(1.0, color="gray", ls="--", lw=1.5, alpha=0.6, zorder=2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(all_labels)
    axes[0].set_title("GWAS Risk Variant Enrichment (distance-corrected)")
    axes[0].set_ylabel("Fold Enrichment (Observed / Expected)")
    axes[0].set_xlabel("Prioritization Threshold (Top N)")
    axes[0].legend(frameon=False)
    style_axes(axes[0])

    # ── subplot 2: cumulative recall curve ──────────────────────────────────
    tl_pct  = (np.arange(1, tl_total  + 1) / tl_total)  * 100
    enf_pct = (np.arange(1, enf_total + 1) / enf_total) * 100
    tl_cum  = tl_ranked["IS_GWAS_SIG"].cumsum().values
    enf_cum = enf_ranked["IS_GWAS_SIG"].cumsum().values

    axes[1].plot(tl_pct, (np.arange(1, tl_total + 1) / tl_total) * tl_sig_n,
                 color="gray", ls="--", lw=1.5, alpha=0.6, label="Random Expectation", zorder=2)
    axes[1].plot(tl_pct,  tl_cum,  color=PASTL_INK,  lw=1.8, label="PASTL",  zorder=3)
    axes[1].plot(enf_pct, enf_cum, color=ENFORMER_INK, lw=1.8, label="Enformer", zorder=4)
    axes[1].set_title("Cumulative GWAS Hit Recall (distance-corrected)")
    axes[1].set_xlabel("Percentage of Dataset Ranked (%)")
    axes[1].set_ylabel("Cumulative GWAS Hits Captured")
    axes[1].legend(frameon=False)
    style_axes(axes[1])

    fig.savefig(figs_dir / "gwas_enrichment_residualized.png", dpi=300, bbox_inches="tight")
    fig.savefig(figs_dir / "gwas_enrichment_residualized.pdf", bbox_inches="tight")
    plt.close(fig)
    return enrich


def figure_residualized_venn(tl: pd.DataFrame, enf: pd.DataFrame,
                             figs_dir: Path, tables_dir: Path) -> pd.DataFrame:
    """Matched top-K Venn diagram using residualized rankings."""
    tl_k = tl.copy();  tl_k["_key"]  = variant_key(tl_k)
    enf_k = enf.copy(); enf_k["_key"] = variant_key(enf_k)
    shared = set(tl_k["_key"]) & set(enf_k["_key"])
    tl_k  = tl_k[tl_k["_key"].isin(shared)].reset_index(drop=True)
    enf_k = enf_k[enf_k["_key"].isin(shared)].reset_index(drop=True)
    print(f"\nVenn shared universe (residualized): {len(shared):,} variants")

    sig_keys = set(tl_k.loc[tl_k["IS_GWAS_SIG"], "_key"])
    tl_rank  = tl_k.sort_values("residualized_score",  ascending=False)["_key"].tolist()
    enf_rank = enf_k.sort_values("residualized_score", ascending=False)["_key"].tolist()

    thresholds = [k for k in (250_000, 500_000, 1_000_000) if k <= len(shared)]
    if not thresholds:
        thresholds = [int(len(shared) * f) for f in (0.1, 0.25, 0.5) if int(len(shared) * f) > 0]

    rows = []
    for k in thresholds:
        tl_top, enf_top = set(tl_rank[:k]), set(enf_rank[:k])
        only_tl, only_enf, overlap = tl_top - enf_top, enf_top - tl_top, tl_top & enf_top
        rows.append({"Threshold": f"{k // 1000}K" if k >= 1000 else str(k), "K": k,
                     "TL_only_sig":      len(only_tl  & sig_keys),
                     "Enformer_only_sig": len(only_enf & sig_keys),
                     "Overlap_sig":       len(overlap  & sig_keys)})
    venn = pd.DataFrame(rows)
    venn.to_csv(tables_dir / "two_model_venn_residualized.tsv", sep="\t", index=False)

    n = len(venn)
    if n == 0:
        print("  Venn (residualized): empty shared universe — skipping figure")
        return venn

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.8), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (_, r) in zip(axes, venn.iterrows()):
        _draw_venn(ax, r["TL_only_sig"], r["Enformer_only_sig"], r["Overlap_sig"],
                   f"Top {r['Threshold']}")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=PASTL_FILL,  alpha=0.7, label="PASTL"),
                        plt.Rectangle((0, 0), 1, 1, color=ENFORMER_FILL, alpha=0.7, label="Enformer")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("PASTL vs. Enformer top-K variant sets — distance-corrected "
                 "(GWAS-significant hits per region)")
    fig.savefig(figs_dir / "two_model_venn_residualized.png", dpi=300, bbox_inches="tight")
    fig.savefig(figs_dir / "two_model_venn_residualized.pdf", bbox_inches="tight")
    plt.close(fig)
    return venn


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wandb_run_id", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--predictions_dir", default=_PREDICTIONS_DIR)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    palette.use_style()

    out_dir    = Path(args.output_dir)
    figs_dir   = out_dir / "figures"
    tables_dir = out_dir / "tables"
    for d in (figs_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    pred_dir = Path(args.predictions_dir)
    slug     = ckpt_slug(args.wandb_run_id, args.checkpoint)
    tl_path  = pred_dir / slug / "variants" / "gwas.tsv"
    enf_path = pred_dir / "enformer" / "variants" / "gwas.tsv"
    for p in (tl_path, enf_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing GWAS predictions at {p}")

    tl  = dedup_variants(load_scored(tl_path,  "DELTA_Z"))
    enf = dedup_variants(load_scored(enf_path, "LOG2FC"))
    print(f"PASTL: {len(tl):,} variants;  Enformer: {len(enf):,} variants")

    tl,  tl_ols,  tl_r2  = _residualize(tl)
    enf, enf_ols, enf_r2 = _residualize(enf)
    print("\nOLS regression stats (abs_score ~ log10(dist_to_TSS + 1)):")
    print(f"  PASTL   slope={tl_ols.coef_[0]:.4f}  intercept={tl_ols.intercept_:.4f}  R²={tl_r2:.4f}")
    print(f"  Enformer  slope={enf_ols.coef_[0]:.4f}  intercept={enf_ols.intercept_:.4f}  R²={enf_r2:.4f}")

    figure_residualized_enrichment(tl, enf, figs_dir, tables_dir)
    figure_residualized_venn(tl, enf, figs_dir, tables_dir)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
