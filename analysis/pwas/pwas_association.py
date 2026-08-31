r"""
FUSION / S-PrediXcan style summary-statistics-based association for PWAS.

Combines per-gene ElasticNet imputation weights (from train_imputation_models.py)
with GWAS summary-statistic Z-scores and a genotype-derived LD reference to
compute a gene-level association Z and p-value for each imputable gene, then
applies multiple-testing correction. Genes are filtered by cross-validated
imputation R^2 before testing.

Inputs:
  {weights_dir}/{gene}_weights.tsv   per-gene non-zero ElasticNet weights
  cv_performance.tsv                 per-gene CV R^2
  gwas_sumstats                      parsed GWAS summary stats (TSV, not raw VCF)
  LD reference                       per-gene genotype matrices or VCFs

Output:
  {output_dir}/pwas_results.tsv

Usage:
  python pwas_association.py \
      --weights_dir out/analysis/pwas/models/pastl_250K \
      --cv_performance out/analysis/pwas/models/pastl_250K/cv_performance.tsv \
      --gwas_sumstats /path/to/gwas_sumstats.tsv \
      --ld_genotype_dir out/analysis/pwas/genotypes \
      --output_dir out/analysis/pwas/results/pastl_250K
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

def load_gwas_sumstats(path: str) -> pd.DataFrame:
    """
    Load GWAS summary stats and construct a variant_id -> Z_gwas mapping.
    Expects some combination of CHROM, POS, REF, ALT and effect/p-value/se/z-score.
    Returns a DataFrame indexed by variant_id with columns Z_gwas and variant_id_flipped.
    """
    print(f"Loading GWAS sumstats from {path} ...")
    if path.endswith(".vcf") or path.endswith(".vcf.gz"):
        raise ValueError("Please provide a TSV/CSV parsed GWAS summary stats file, not raw VCF.")

    df = pd.read_csv(path, sep="\t", low_memory=False)

    # Identify standard columns
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ["chrom", "chr", "chromosome"]: col_map["chrom"] = c
        elif cl in ["pos", "position"]: col_map["pos"] = c
        elif cl in ["ref", "reference_allele", "a0"]: col_map["ref"] = c
        elif cl in ["alt", "effect_allele", "a1"]: col_map["alt"] = c
        elif cl in ["beta", "effect", "empirical_effect", "b"]: col_map["beta"] = c
        elif cl in ["se", "standard_error", "stderr"]: col_map["se"] = c
        elif cl in ["z", "zscore", "z_score", "gwas_z"]: col_map["z"] = c
        elif cl in ["p", "pval", "pvalue", "p_value"]: col_map["p"] = c

    if "chrom" not in col_map or "pos" not in col_map or "ref" not in col_map or "alt" not in col_map:
        raise ValueError(f"Could not find CHROM/POS/REF/ALT in columns: {df.columns.tolist()}")

    chrom = df[col_map["chrom"]].astype(str)
    chrom = np.where(chrom.str.startswith("chr"), chrom, "chr" + chrom)
    pos = df[col_map["pos"]].astype(str)
    ref = df[col_map["ref"]].astype(str)
    alt = df[col_map["alt"]].astype(str)

    df["variant_id"] = chrom + ":" + pos + ":" + ref + ":" + alt

    # Build a second index for allele-flipped matching.
    # If weight file has chrom:pos:A:B but GWAS has chrom:pos:B:A,
    # the variant is the same but Z must be negated.
    flipped_id = chrom + ":" + pos + ":" + alt + ":" + ref
    df["variant_id_flipped"] = flipped_id

    # Compute Z
    if "z" in col_map:
        df["Z_gwas"] = pd.to_numeric(df[col_map["z"]], errors="coerce")
    elif "beta" in col_map and "se" in col_map:
        beta = pd.to_numeric(df[col_map["beta"]], errors="coerce")
        se = pd.to_numeric(df[col_map["se"]], errors="coerce")
        df["Z_gwas"] = beta / se
    elif "beta" in col_map and "p" in col_map:
        beta = pd.to_numeric(df[col_map["beta"]], errors="coerce")
        p = pd.to_numeric(df[col_map["p"]], errors="coerce")
        # Z = sign(beta) * norm.ppf(1 - p/2)
        p = np.clip(p, 1e-300, 1.0)
        z_mag = norm.isf(p / 2.0)
        df["Z_gwas"] = np.sign(beta) * z_mag
    else:
        raise ValueError("GWAS sumstats must contain either Z, beta+se, or beta+p.")

    df = df.dropna(subset=["Z_gwas"]).drop_duplicates(subset=["variant_id"])
    print(f"  Loaded {len(df):,} variants with valid Z-scores.")
    return df.set_index("variant_id")[["Z_gwas", "variant_id_flipped"]]


def _resolve_ld_vcf(ld_vcf_dir: Path, chrom: str) -> Path | None:
    """Find the per-chromosome VCF in ld_vcf_dir matching the given chrom (e.g. 'chr1')."""
    bare = chrom.replace("chr", "")
    prefixed = f"chr{bare}"
    # Try common 1KG naming patterns
    for pattern in [f"*{prefixed}.vcf.gz", f"*.{prefixed}.*.vcf.gz"]:
        matches = sorted(ld_vcf_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def compute_ld_from_vcf(
    ld_vcf_dir: Path,
    ld_sample_ids: list[str] | None,
    variant_ids: list[str],
) -> pd.DataFrame:
    """
    Extract dosages for the given variant IDs from per-chromosome
    tabix-indexed VCFs and return a dosage DataFrame.

    variant_ids are 'chr:pos:ref:alt' strings (1-based pos).
    """
    # Parse variant IDs into per-chromosome lookup
    by_chrom = {}
    for vid in variant_ids:
        parts = vid.split(":")
        chrom, pos, ref, alt = parts[0], int(parts[1]), parts[2], parts[3]
        by_chrom.setdefault(chrom, []).append((pos, ref, alt, vid))

    dosage_cols = {}
    samples = None

    for chrom, var_list in by_chrom.items():
        vcf_path = _resolve_ld_vcf(ld_vcf_dir, chrom)
        if vcf_path is None:
            continue

        vcf = pysam.VariantFile(str(vcf_path))
        if ld_sample_ids is not None:
            vcf.subset_samples(ld_sample_ids)

        if samples is None:
            samples = list(vcf.header.samples)

        # Group by position for efficient fetching
        pos_lookup = {}
        for pos, ref, alt, vid in var_list:
            pos_lookup.setdefault(pos, []).append((ref, alt, vid))

        for pos, allele_list in pos_lookup.items():
            # Try chr-prefixed and bare chromosome names
            fetch_chrom = chrom
            try:
                records = list(vcf.fetch(fetch_chrom, pos - 1, pos))
            except ValueError:
                bare = chrom.replace("chr", "") if chrom.startswith("chr") else "chr" + chrom
                try:
                    records = list(vcf.fetch(bare, pos - 1, pos))
                except ValueError:
                    continue

            for rec in records:
                if rec.pos != pos:
                    continue
                if rec.alts is None:
                    continue
                for ref, alt, vid in allele_list:
                    if rec.ref == ref and alt in rec.alts:
                        alt_idx = list(rec.alts).index(alt)
                        dosages = []
                        for s in samples:
                            gt = rec.samples[s].allele_indices
                            if gt is None or any(a is None for a in gt):
                                dosages.append(np.nan)
                            else:
                                dosages.append(float(sum(1 for a in gt if a == alt_idx + 1)))
                        dosage_cols[vid] = np.array(dosages, dtype=float)

        vcf.close()

    if not dosage_cols or samples is None:
        return pd.DataFrame()

    return pd.DataFrame(dosage_cols, index=samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights_dir", required=True, help="Directory with per-gene *_weights.tsv files")
    parser.add_argument("--cv_performance", required=True, help="Path to cv_performance.tsv")
    parser.add_argument("--gwas_sumstats", required=True, help="Path to parsed GWAS summary stats TSV")
    parser.add_argument("--ld_vcf_dir", default=None,
                        help="Directory of per-chromosome tabix-indexed VCFs for LD reference "
                             "(e.g., 1000G EUR). Expected naming: "
                             "*chr{N}.vcf.gz or *chr{N}.*.vcf.gz. "
                             "If not provided, LD is computed from within-cohort genotype matrices.")
    parser.add_argument("--ld_genotype_dir", default=None,
                        help="Directory with per-gene genotype matrices (*.tsv.gz) for within-cohort LD. "
                             "Used only when --ld_vcf_dir is not provided.")
    parser.add_argument("--ld_samples", default=None,
                        help="Text file with sample IDs to use from LD VCF (one per line). "
                             "If not provided, all samples in the VCF are used.")
    parser.add_argument("--output_dir", required=True, help="Directory to save pwas_results.tsv")
    parser.add_argument("--r2_threshold", type=float, default=0.01, help="R² threshold to filter well-predicted genes (default: 0.01)")
    parser.add_argument("--cv_performance_all", nargs="+", default=None,
                        help="cv_performance.tsv files from ALL model sets. When provided, a gene "
                             "is included if it passes --r2_threshold in ANY of these files, "
                             "enforcing a consistent gene universe across models.")
    args = parser.parse_args()

    if args.ld_vcf_dir is None and args.ld_genotype_dir is None:
        parser.error("Either --ld_vcf_dir (external reference LD) or --ld_genotype_dir (within-cohort LD) is required.")
    if args.ld_vcf_dir is not None and args.ld_genotype_dir is not None:
        print("  [NOTE] Both --ld_vcf_dir and --ld_genotype_dir provided; using external reference LD (--ld_vcf_dir).")

    use_external_ld = args.ld_vcf_dir is not None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = Path(args.weights_dir)

    ld_sample_ids = None
    if args.ld_samples:
        with open(args.ld_samples) as f:
            ld_sample_ids = [line.strip() for line in f if line.strip()]
        print(f"  Using {len(ld_sample_ids)} samples from LD reference.")

    if use_external_ld:
        print(f"  LD mode: external reference ({args.ld_vcf_dir})")
    else:
        print(f"  LD mode: within-cohort ({args.ld_genotype_dir})")

    print(f"Loading CV performance from {args.cv_performance} ...")
    cv_df = pd.read_csv(args.cv_performance, sep="\t")

    if args.cv_performance_all is not None:
        passing_sets = []
        for path in args.cv_performance_all:
            df_other = pd.read_csv(path, sep="\t")
            passing_sets.append(set(df_other.loc[df_other["cv_r2"] > args.r2_threshold, "gene_symbol"]))
        intersection_genes = passing_sets[0].intersection(*passing_sets[1:])
        well_predicted = cv_df[cv_df["gene_symbol"].isin(intersection_genes)].copy()
        print(f"  {len(well_predicted)} / {len(cv_df)} genes pass R² > {args.r2_threshold} in ALL models (intersection across {len(args.cv_performance_all)} sets)")
    else:
        well_predicted = cv_df[cv_df["cv_r2"] > args.r2_threshold].copy()
        print(f"  {len(well_predicted)} / {len(cv_df)} genes passed R² > {args.r2_threshold}")

    if len(well_predicted) == 0:
        print("No genes passed threshold. Exiting.")
        return

    gwas_df = load_gwas_sumstats(args.gwas_sumstats)

    results = []

    genes = well_predicted["gene_symbol"].tolist()
    for idx, gene in enumerate(genes):
        if (idx + 1) % 50 == 0:
            print(f"  Processing gene {idx + 1}/{len(genes)}: {gene}")

        w_path = weights_dir / f"{gene}_weights.tsv"
        if not w_path.exists():
            continue

        w_df = pd.read_csv(w_path, sep="\t")
        if w_df.empty:
            continue

        w_df = w_df.set_index("variant_id")["weight"]

        # Match with GWAS (direct and allele-flipped)
        direct = w_df.index.intersection(gwas_df.index)
        gwas_flip_map = gwas_df["variant_id_flipped"].reset_index().set_index("variant_id_flipped")["variant_id"]
        remaining = w_df.index.difference(direct)
        flipped_in_gwas = remaining.intersection(gwas_flip_map.index)

        z_direct = gwas_df.loc[direct, "Z_gwas"]
        z_flipped = -gwas_df.loc[gwas_flip_map.loc[flipped_in_gwas].values, "Z_gwas"]
        z_flipped.index = flipped_in_gwas
        gwas_z_matched = pd.concat([z_direct, z_flipped])
        common = gwas_z_matched.index
        n_direct = len(direct)
        n_flipped = len(flipped_in_gwas)

        if len(common) == 0:
            results.append({
                "gene_symbol": gene,
                "z_score": np.nan,
                "p_value": np.nan,
                "n_snps_in_model": len(w_df),
                "n_snps_matched_gwas": 0,
                "n_snps_flipped": 0,
                "cv_r2": cv_df.loc[cv_df["gene_symbol"] == gene, "cv_r2"].values[0]
            })
            continue

        if use_external_ld:
            # Compute LD from external reference VCF
            avail_list = list(common)
            ld_geno = compute_ld_from_vcf(Path(args.ld_vcf_dir), ld_sample_ids, avail_list)

            # Filter to variants actually found in the LD reference
            avail = common.intersection(ld_geno.columns)
        else:
            # Compute LD from within-cohort genotype matrix
            geno_path = Path(args.ld_genotype_dir) / f"{gene}.tsv.gz"
            if not geno_path.exists():
                geno_path = Path(args.ld_genotype_dir) / f"{gene}.tsv"
                if not geno_path.exists():
                    print(f"  [WARN] Missing genotype matrix for {gene} at {args.ld_genotype_dir}/{gene}.tsv.gz")
                    continue

            geno = pd.read_csv(geno_path, sep="\t", index_col=0, low_memory=False)
            avail = common.intersection(geno.columns)
            if len(avail) > 0:
                ld_geno = geno[list(avail)]

        if len(avail) == 0:
            results.append({
                "gene_symbol": gene,
                "z_score": np.nan,
                "p_value": np.nan,
                "n_snps_in_model": len(w_df),
                "n_snps_matched_gwas": len(common),
                "n_snps_flipped": n_flipped,
                "cv_r2": cv_df.loc[cv_df["gene_symbol"] == gene, "cv_r2"].values[0]
            })
            continue

        # Extract weight and z-score vectors for the available SNPs
        w = w_df.loc[avail].values
        z_gwas = gwas_z_matched.loc[avail].values

        X_ld = ld_geno[list(avail)].astype(float)
        # Impute missing dosages to the column mean
        X_ld = X_ld.fillna(X_ld.mean(axis=0))

        # Compute correlation matrix (LD)
        Sigma_LD = X_ld.corr().fillna(0).to_numpy(copy=True)
        np.fill_diagonal(Sigma_LD, 1.0)

        # Compute PWAS Z-score
        var_z = w.T @ Sigma_LD @ w
        if var_z <= 0:
            z_pwas = np.nan
            p_val = np.nan
        else:
            z_pwas = (w.T @ z_gwas) / np.sqrt(var_z)
            p_val = 2 * norm.sf(abs(z_pwas))

        results.append({
            "gene_symbol": gene,
            "z_score": z_pwas,
            "p_value": p_val,
            "n_snps_in_model": len(w_df),
            "n_snps_matched_gwas": len(avail),
            "n_snps_flipped": n_flipped,
            "cv_r2": cv_df.loc[cv_df["gene_symbol"] == gene, "cv_r2"].values[0]
        })

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        # Drop genes that failed computation for Bonferroni calculation
        valid_df = res_df.dropna(subset=["p_value"]).copy()

        n_tests = len(valid_df)
        if n_tests > 0:
            valid_df["p_bonferroni"] = np.minimum(valid_df["p_value"] * n_tests, 1.0)

            _, fdr_vals, _, _ = multipletests(valid_df["p_value"].values,
                                              method="fdr_bh")
            valid_df["q_fdr"] = fdr_vals

            valid_df["sig_bonferroni"] = valid_df["p_bonferroni"] < 0.05
            valid_df["sig_fdr"]        = valid_df["q_fdr"]        < 0.05
            valid_df = valid_df.sort_values("p_value")

            failed_df = res_df[res_df["p_value"].isna()].copy()
            failed_df["p_bonferroni"] = np.nan
            failed_df["q_fdr"]        = np.nan
            failed_df["sig_bonferroni"] = False
            failed_df["sig_fdr"]        = False

            out_df = pd.concat([valid_df, failed_df], ignore_index=True)

            out_path = out_dir / "pwas_results.tsv"
            out_df.to_csv(out_path, sep="\t", index=False)

            n_sig_bonf = int(valid_df["sig_bonferroni"].sum())
            n_sig_fdr  = int(valid_df["sig_fdr"].sum())
            print("\n" + "="*50)
            print(f"Total genes tested:            {n_tests:,}")
            print(f"Significant (Bonferroni p<0.05): {n_sig_bonf:,}")
            print(f"Significant (FDR q<0.05):        {n_sig_fdr:,}")
            print("="*50)
            print("Top 20 genes:")
            print(valid_df.head(20).to_string(index=False))
            print("="*50)
            print(f"Saved results to {out_path}")
        else:
            print("No valid results computed.")
    else:
        print("No valid results computed.")

if __name__ == "__main__":
    main()
