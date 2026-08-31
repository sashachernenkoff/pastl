r"""
Build residualized-score priority SNV sets for PWAS analysis.

Loads GWAS variant scores from PASTL and Enformer, deduplicates by variant
(keeping the max absolute-score row per variant), residualizes absolute score
against log10(distance_to_TSS + 1) via OLS, and saves ranked priority sets
at three thresholds plus an unfiltered baseline.

Inputs:
  PASTL:  {pastl_gwas}   — columns include DELTA_Z, TSS, TARGET_SYMBOL, P_VALUE
  Enformer: {enformer_gwas} — same format but LOG2FC instead of DELTA_Z

Outputs (per model, under --output_dir):
  pastl_all_variants.tsv         full deduplicated set
  pastl_top250K.tsv              top 250,000 by residualized score
  pastl_top500K.tsv              top 500,000
  pastl_top1M.tsv                top 1,000,000
  enformer_all_variants.tsv
  enformer_top{N}.tsv             same thresholds
  build_priority_sets.log         duplicate of stdout

Usage:
  python build_priority_sets.py \
      --pastl_gwas out/predictions/s5fo3aak_best/variants/gwas.tsv \
      --enformer_gwas out/predictions/enformer/variants/gwas.tsv \
      --output_dir out/analysis/pwas/priority_snvs/
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_DEFAULT_ENFORMER = (
    "out/predictions/enformer/variants/gwas.tsv"
)
_DEFAULT_OUT = "out/analysis/pwas/priority_snvs/"

GWAS_P_THRESH = 5e-8
THRESHOLDS = [250_000, 500_000, 1_000_000]
THRESHOLD_LABELS = ["250K", "500K", "1M"]
VARIANT_KEY = ["CHROM", "POS", "REF", "ALT"]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _log(msg: str, log_fh):
    print(msg)
    log_fh.write(msg + "\n")
    log_fh.flush()


def load_and_dedup(path: str, score_col: str, abs_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    required = VARIANT_KEY + ["TSS", "TARGET_SYMBOL", "P_VALUE", score_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    df[abs_col] = df[score_col].abs()

    # Deduplicate by variant: keep the row with the highest abs score per variant.
    # Tie-breaking is stable (first occurrence wins after sort).
    df = (
        df.sort_values(abs_col, ascending=False)
        .drop_duplicates(subset=VARIANT_KEY, keep="first")
        .reset_index(drop=True)
    )
    return df


# ---------------------------------------------------------------------------
# Residualization
# ---------------------------------------------------------------------------

def residualize(df: pd.DataFrame, abs_col: str, log_fh) -> pd.DataFrame:
    df = df.copy()
    df["distance_to_tss"] = (df["POS"].astype(int) - (df["TSS"].astype(int) + 1)).abs()  # +1: TSS in gwas.tsv is 0-based (BED origin); POS is 1-based (VCF)
    x = np.log10(df["distance_to_tss"].values + 1)
    y = df[abs_col].values

    slope, intercept, r, _p, _se = stats.linregress(x, y)
    r2 = r ** 2
    _log(
        f"  OLS {abs_col} ~ log10(dist+1): "
        f"slope={slope:.4f}  intercept={intercept:.4f}  R²={r2:.4f}",
        log_fh,
    )

    predicted = intercept + slope * x
    df["residualized_score"] = y - predicted
    return df


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def dist_summary(df: pd.DataFrame) -> str:
    d = df["distance_to_tss"]
    q25, q75 = d.quantile(0.25), d.quantile(0.75)
    return f"median={d.median():.0f}  IQR=[{q25:.0f}, {q75:.0f}]"



# ---------------------------------------------------------------------------
# Per-model processing
# ---------------------------------------------------------------------------

def process_model(
    path: str,
    score_col: str,
    abs_col: str,
    prefix: str,
    out_dir: Path,
    log_fh,
):
    _log(f"\n{'='*60}", log_fh)
    _log(f"Model: {prefix}  ({score_col})", log_fh)
    _log(f"Input: {path}", log_fh)

    df = load_and_dedup(path, score_col, abs_col)
    _log(f"  Total unique variants after dedup: {len(df):,}", log_fh)

    total_gwas_hits = (df["P_VALUE"] < GWAS_P_THRESH).sum()
    _log(f"  GWAS hits (p < {GWAS_P_THRESH:.0e}): {total_gwas_hits:,}", log_fh)

    df = residualize(df, abs_col, log_fh)

    df_ranked = df.sort_values("residualized_score", ascending=False).reset_index(drop=True)

    out_cols = VARIANT_KEY + ["residualized_score", abs_col, "distance_to_tss", "TARGET_SYMBOL", "P_VALUE"]
    rename_map = {abs_col: "raw_score", "TARGET_SYMBOL": "gene_symbol"}

    # Save full baseline.
    all_path = out_dir / f"{prefix}_all_variants.tsv"
    df_ranked[out_cols].rename(columns=rename_map).to_csv(all_path, sep="\t", index=False)
    _log(f"\n  [all] {len(df_ranked):,} variants  dist: {dist_summary(df_ranked)}", log_fh)
    total_n = len(df_ranked)
    n_all_hits = (df_ranked["P_VALUE"] < GWAS_P_THRESH).sum()
    baseline_rate = n_all_hits / total_n if total_n else 0
    _log(f"  [all] GWAS hits: {n_all_hits:,} / {total_n:,}  (rate={baseline_rate:.4e})", log_fh)
    _log(f"  -> {all_path}", log_fh)

    # Save threshold sets.
    for n, label in zip(THRESHOLDS, THRESHOLD_LABELS):
        subset = df_ranked.head(n)
        if len(subset) == 0:
            _log(f"  [top{label}] no variants (total={total_n:,})", log_fh)
            continue
        subset_path = out_dir / f"{prefix}_top{label}.tsv"
        subset[out_cols].rename(columns=rename_map).to_csv(subset_path, sep="\t", index=False)

        n_hits = (subset["P_VALUE"] < GWAS_P_THRESH).sum()
        expected = baseline_rate * len(subset)
        fold = n_hits / expected if expected > 0 else float("nan")

        _log(
            f"  [top{label}] {len(subset):,} variants  "
            f"dist: {dist_summary(subset)}  "
            f"GWAS hits: {n_hits:,} (expected {expected:.1f}, fold={fold:.2f}x)",
            log_fh,
        )
        _log(f"  -> {subset_path}", log_fh)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pastl_gwas", required=True,
                        help="Path to PASTL gwas.tsv")
    parser.add_argument("--enformer_gwas", default=_DEFAULT_ENFORMER,
                        help="Path to Enformer gwas.tsv")
    parser.add_argument("--output_dir", default=_DEFAULT_OUT,
                        help="Output directory (default: %(default)s)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "build_priority_sets.log"
    with open(log_path, "w") as log_fh:
        _log(f"build_priority_sets.py", log_fh)
        _log(f"output_dir: {out_dir}", log_fh)

        process_model(
            path=args.pastl_gwas,
            score_col="DELTA_Z",
            abs_col="ABS_DELTA_Z",
            prefix="pastl",
            out_dir=out_dir,
            log_fh=log_fh,
        )

        process_model(
            path=args.enformer_gwas,
            score_col="LOG2FC",
            abs_col="ABS_LOG2FC",
            prefix="enformer",
            out_dir=out_dir,
            log_fh=log_fh,
        )

        _log(f"\nDone. Log -> {log_path}", log_fh)


if __name__ == "__main__":
    main()
