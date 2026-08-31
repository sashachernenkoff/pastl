"""
Shared helpers for building variant→gene-window manifests from raw data.

The gene-window universe comes entirely from data/sequences_protein.bed: each
protein defines a 196,608 bp Enformer input window centred on its TSS, so
    TSS = start + 98,304   and   window = [TSS - 98,304, TSS + 98,304].
A variant is assigned to a gene when it is on the same chromosome and within
that window (|POS - TSS| <= 98,304). No external gene-metadata table is needed.

Used by analysis/gwas/build_variants.py and analysis/qtl/build_variants.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW_SIZE = 196_608
HALF_WINDOW = WINDOW_SIZE // 2   # 98,304


# ---------------------------------------------------------------------------
# Gene-window universe
# ---------------------------------------------------------------------------

def load_windows(sequences_bed: str) -> pd.DataFrame:
    """
    Load the protein windows and derive TSS from the window centre.

    Returns columns: chrom, tss, strand, symbol, ensembl_gene_id.
    """
    bed = pd.read_csv(sequences_bed, sep="\t")
    win = pd.DataFrame({
        "chrom":           bed["chrom"].astype(str),
        "tss":             (bed["start"].astype(int) + HALF_WINDOW),
        "strand":          bed["strand"].astype(str),
        "symbol":          bed["symbol"].astype(str),
        "ensembl_gene_id": bed["ensembl_gene_id"].astype(str).str.split(".").str[0],
    })
    return win


# ---------------------------------------------------------------------------
# Liftover
# ---------------------------------------------------------------------------

def liftover(df: pd.DataFrame, chain_path: str,
             chrom_col: str = "CHROM", pos_col: str = "POS") -> pd.DataFrame:
    """
    Lift 1-based coordinates to a new assembly with a local chain file.

    Rows that fail to map are dropped. Returns a copy with `pos_col` replaced by
    the lifted 1-based position (int).
    """
    from pyliftover import LiftOver

    lo = LiftOver(chain_path)
    df = df.copy()
    new_pos = []
    for chrom, pos in zip(df[chrom_col].astype(str), df[pos_col].astype(int)):
        res = lo.convert_coordinate(chrom, pos - 1)   # pyliftover is 0-based
        new_pos.append(res[0][1] + 1 if res else np.nan)   # back to 1-based
    df[pos_col] = new_pos
    n_before = len(df)
    df = df.dropna(subset=[pos_col]).copy()
    df[pos_col] = df[pos_col].astype(int)
    print(f"  liftover: {n_before - len(df):,} of {n_before:,} dropped (unmapped)")
    return df


# ---------------------------------------------------------------------------
# Variant → window mapping
# ---------------------------------------------------------------------------

def _add_window_columns(mapped: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Attach TSS / STRAND / TARGET_GENE / TARGET_SYMBOL from a windows row join."""
    mapped = mapped.rename(columns={
        "tss": "TSS", "strand": "STRAND",
        "ensembl_gene_id": "TARGET_GENE", "symbol": "TARGET_SYMBOL",
    })
    return mapped


def map_to_target_gene(var_df: pd.DataFrame, windows: pd.DataFrame,
                       var_key: str, window_key: str) -> pd.DataFrame:
    """
    Map variants that already name a target gene (eQTLs) to its window.

    var_key    : column in var_df identifying the gene (Ensembl ID or symbol).
    window_key : matching column in `windows` (ensembl_gene_id or symbol).

    Keeps only same-chromosome, in-window pairs. Adds TSS/STRAND/TARGET_GENE/
    TARGET_SYMBOL columns.
    """
    merged = var_df.merge(windows, left_on=var_key, right_on=window_key, how="inner",
                          suffixes=("", "_win"))
    win_chrom = merged["chrom"].astype(str)
    var_chrom = merged["CHROM"].astype(str)
    same_chrom = win_chrom.str.replace("chr", "", regex=False) == \
        var_chrom.str.replace("chr", "", regex=False)
    in_window = (merged["POS"].astype(int) - merged["tss"]).abs() <= HALF_WINDOW
    merged = merged[same_chrom & in_window].copy()
    return _add_window_columns(merged, windows)


def map_to_all_windows(var_df: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """
    Map variants with no known target gene (GWAS) to every window that contains
    them — one output row per (variant, gene) pair.

    Uses a per-chromosome sorted-position interval scan (searchsorted), so it
    scales to genome-wide summary statistics without an O(V×G) distance matrix.
    """
    def norm(s):
        return s.astype(str).str.replace("chr", "", regex=False)

    var_df = var_df.reset_index(drop=True).copy()
    windows = windows.copy()
    var_df["_chr"] = norm(var_df["CHROM"])
    windows["_chr"] = norm(windows["chrom"])

    out_rows = []
    for chrom, genes in windows.groupby("_chr"):
        vars_c = var_df[var_df["_chr"] == chrom]
        if vars_c.empty:
            continue
        order = np.argsort(vars_c["POS"].astype(int).values)
        pos_sorted = vars_c["POS"].astype(int).values[order]
        vars_idx = vars_c.index.values[order]
        for _, g in genes.iterrows():
            lo = int(g["tss"]) - HALF_WINDOW
            hi = int(g["tss"]) + HALF_WINDOW
            a = np.searchsorted(pos_sorted, lo, side="left")
            b = np.searchsorted(pos_sorted, hi, side="right")
            if b <= a:
                continue
            hit = var_df.loc[vars_idx[a:b]].copy()
            hit["tss"] = int(g["tss"])
            hit["strand"] = g["strand"]
            hit["ensembl_gene_id"] = g["ensembl_gene_id"]
            hit["symbol"] = g["symbol"]
            out_rows.append(hit)

    if not out_rows:
        return pd.DataFrame(columns=list(var_df.columns) +
                            ["TSS", "STRAND", "TARGET_GENE", "TARGET_SYMBOL"])
    mapped = pd.concat(out_rows, ignore_index=True).drop(columns=["_chr"])
    return _add_window_columns(mapped, windows)


def finalize_manifest(mapped: pd.DataFrame, extra_cols: list[str],
                      source: str) -> pd.DataFrame:
    """Select the canonical manifest columns, tag SOURCE, sort and dedup."""
    base = ["CHROM", "POS", "REF", "ALT", "TARGET_GENE", "TARGET_SYMBOL", "TSS", "STRAND"]
    cols = base + [c for c in extra_cols if c in mapped.columns]
    out = mapped[cols].copy()
    out["SOURCE"] = source
    out = out.drop_duplicates().sort_values(["CHROM", "POS"]).reset_index(drop=True)
    return out
