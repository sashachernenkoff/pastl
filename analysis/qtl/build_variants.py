r"""
Build eQTL variant→gene-window manifests for the QTL analysis.

Two fine-mapped cis-eQTL sources, each keyed to a known target gene:

  GTEx v10 lung SuSiE  (Lung.v10.eQTLs.SuSiE_summary.parquet)
      GRCh38 already (variant_id like chr1_12345_A_G_b38). Carries a signed
      empirical effect (afc = allelic fold change) and a fine-mapping PIP, so it
      drives both the AUROC and the directional-concordance panels.

  PancanQTL LUAD SuSiE (LUAD.cis.susie.txt.gz)
      GRCh37 → lifted to hg38. Carries a PIP but no signed effect, so it drives
      the AUROC panel only.

Each variant is mapped to its target gene's Enformer window via
data/sequences_protein.bed (GTEx on Ensembl gene id, Pancan on gene symbol),
keeping only variants that fall inside the window.

Outputs (model-independent; shared by both PASTL and Enformer scoring):
  {output_dir}/eqtl_gtex_manifest.tsv
  {output_dir}/eqtl_pancan_manifest.tsv

Usage:
  python build_variants.py \
      --eqtl_gtex   /.../eqtl/Lung.v10.eQTLs.SuSiE_summary.parquet \
      --eqtl_pancan /.../eqtl/LUAD.cis.susie.txt.gz \
      --sequences_bed data/sequences_protein.bed \
      --chain /.../hg19ToHg38.over.chain.gz \
      --output_dir out/predictions/variants
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from variant_common import load_windows, liftover, map_to_target_gene, finalize_manifest


def build_gtex(path: str, windows: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    print(f"\n[GTEx lung eQTL] reading {path}")
    raw = pd.read_parquet(path)
    if limit:
        raw = raw.head(limit)
    parts = raw["variant_id"].str.split("_", expand=True)
    var = pd.DataFrame({
        "CHROM":            parts[0].astype(str),
        "POS":              parts[1].astype(int),
        "REF":              parts[2].astype(str),
        "ALT":              parts[3].astype(str),
        "gene_ensembl":     raw["phenotype_id"].str.split(".").str[0],
        "EMPIRICAL_EFFECT": pd.to_numeric(raw["afc"], errors="coerce").fillna(0.0),
        "PIP":              pd.to_numeric(raw["pip"], errors="coerce"),
    })
    print(f"  {len(var):,} raw eQTL rows")
    mapped = map_to_target_gene(var, windows, var_key="gene_ensembl",
                                window_key="ensembl_gene_id")
    manifest = finalize_manifest(mapped, ["PIP", "EMPIRICAL_EFFECT"], "GTEX_LUNG_eQTL")
    print(f"  {len(manifest):,} variants mapped to windows "
          f"({manifest['TARGET_GENE'].nunique():,} genes)")
    return manifest


def build_sqtl(path: str, windows: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    print(f"\n[GTEx lung sQTL] reading {path}")
    raw = pd.read_parquet(path)
    if limit:
        raw = raw.head(limit)
    parts = raw["variant_id"].str.split("_", expand=True)   # chrN_pos_ref_alt_b38 (GRCh38)
    # An sQTL phenotype is a Leafcutter intron/cluster; the Ensembl gene is carried
    # either in a gene_id column or embedded in phenotype_id
    # (e.g. chr1:497299:497354:clu_412_+:ENSG00000239945.1).
    gene_src = raw["gene_id"] if "gene_id" in raw.columns else raw["phenotype_id"]
    gene = gene_src.astype(str).str.extract(r"(ENSG\d+)", expand=False)
    effect = (pd.to_numeric(raw["afc"], errors="coerce").fillna(0.0)
              if "afc" in raw.columns else 0.0)
    var = pd.DataFrame({
        "CHROM":            parts[0].astype(str),
        "POS":              parts[1].astype(int),
        "REF":              parts[2].astype(str),
        "ALT":              parts[3].astype(str),
        "gene_ensembl":     gene,
        "EMPIRICAL_EFFECT": effect,
        "PIP":              pd.to_numeric(raw["pip"], errors="coerce"),
    })
    n_nogene = int(var["gene_ensembl"].isna().sum())
    if n_nogene:
        print(f"  dropping {n_nogene:,} rows with no Ensembl gene in gene_id/phenotype_id")
        var = var.dropna(subset=["gene_ensembl"])
    print(f"  {len(var):,} raw sQTL rows")
    mapped = map_to_target_gene(var, windows, var_key="gene_ensembl",
                                window_key="ensembl_gene_id")
    # One variant can be a QTL for several introns of the same gene — collapse to
    # one row per (variant, gene), keeping its most-causal (max-PIP) credible set.
    mapped = (mapped.sort_values("PIP", ascending=False)
              .drop_duplicates(subset=["CHROM", "POS", "REF", "ALT", "TARGET_GENE"]))
    manifest = finalize_manifest(mapped, ["PIP", "EMPIRICAL_EFFECT"], "GTEX_LUNG_sQTL")
    print(f"  {len(manifest):,} variants mapped to windows "
          f"({manifest['TARGET_GENE'].nunique():,} genes)")
    return manifest


def build_pancan(path: str, windows: pd.DataFrame, chain: str,
                 limit: int | None) -> pd.DataFrame:
    print(f"\n[PancanQTL LUAD eQTL] reading {path}")
    raw = pd.read_csv(path, sep=r"\s+", compression="gzip")
    if limit:
        raw = raw.head(limit)
    parts = raw["Variant"].str.split("_", expand=True)
    var = pd.DataFrame({
        "CHROM":     "chr" + parts[0].astype(str),
        "POS":       parts[1].astype(int),
        "REF":       parts[2].astype(str),
        "ALT":       parts[3].astype(str),
        "GENE_NAME": raw["Gene_name"].astype(str),
        "PIP":       pd.to_numeric(raw["PIP"], errors="coerce"),
    })
    print(f"  {len(var):,} raw eQTL rows")
    var = liftover(var, chain)
    mapped = map_to_target_gene(var, windows, var_key="GENE_NAME", window_key="symbol")
    manifest = finalize_manifest(mapped, ["PIP"], "PANCAN_LUAD_eQTL")
    print(f"  {len(manifest):,} variants mapped to windows "
          f"({manifest['TARGET_GENE'].nunique():,} genes)")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eqtl_gtex", default=None, help="GTEx v10 Lung eQTL SuSiE parquet.")
    parser.add_argument("--eqtl_sqtl", default=None, help="GTEx v10 Lung sQTL SuSiE parquet.")
    parser.add_argument("--eqtl_pancan", default=None, help="PancanQTL LUAD SuSiE .txt.gz.")
    parser.add_argument("--sequences_bed", required=True)
    parser.add_argument("--chain", default=None,
                        help="Local hg19→hg38 chain file (required only for the "
                             "PancanQTL source; GTEx eQTL/sQTL are already GRCh38).")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--datasets", nargs="+",
                        default=["eqtl_gtex", "eqtl_pancan"],
                        choices=["eqtl_gtex", "eqtl_sqtl", "eqtl_pancan"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Read only the first N raw rows per source (debugging).")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = load_windows(args.sequences_bed)
    print(f"Loaded {len(windows):,} protein windows from {args.sequences_bed}")

    if "eqtl_gtex" in args.datasets:
        if not args.eqtl_gtex:
            parser.error("--eqtl_gtex is required to build the GTEx eQTL manifest")
        g = build_gtex(args.eqtl_gtex, windows, args.limit)
        g.to_csv(out_dir / "eqtl_gtex_manifest.tsv", sep="\t", index=False)
        print(f"  -> {out_dir / 'eqtl_gtex_manifest.tsv'}")

    if "eqtl_sqtl" in args.datasets:
        if not args.eqtl_sqtl:
            parser.error("--eqtl_sqtl is required to build the GTEx sQTL manifest")
        s = build_sqtl(args.eqtl_sqtl, windows, args.limit)
        s.to_csv(out_dir / "eqtl_sqtl_manifest.tsv", sep="\t", index=False)
        print(f"  -> {out_dir / 'eqtl_sqtl_manifest.tsv'}")

    if "eqtl_pancan" in args.datasets:
        if not args.eqtl_pancan:
            parser.error("--eqtl_pancan is required to build the PancanQTL manifest")
        if not args.chain:
            parser.error("--chain is required to build the PancanQTL (GRCh37) manifest")
        p = build_pancan(args.eqtl_pancan, windows, args.chain, args.limit)
        p.to_csv(out_dir / "eqtl_pancan_manifest.tsv", sep="\t", index=False)
        print(f"  -> {out_dir / 'eqtl_pancan_manifest.tsv'}")


if __name__ == "__main__":
    main()
