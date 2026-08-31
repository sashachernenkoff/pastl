r"""
Train per-gene ElasticNet imputation models for PWAS.

For each gene in a filtered_genotypes/{model}_{threshold}/ directory, loads
the filtered dosage matrix, aligns samples to the protein abundance matrix,
fits an ElasticNetCV model, and records CV R² and per-variant weights.

Protein data is loaded from CPTAC GCT files (Solid Tissue Normal),
matching PASTL training.

Sample alignment: VCF sample IDs (CPT aliquot IDs) → C3L case IDs via a
mapping built from the gVCF headers ({gvcf_dir}/{case_id}.g.vcf.gz → SM tag).

Inputs:
  {genotype_dir}/{gene_symbol}.tsv.gz   filtered dosage matrices
  GCT files (discovery + confirmatory)  raw protein abundance
  {gvcf_dir}/*.g.vcf.gz                 for building CPT → case_id mapping

Outputs (under --output_dir):
  {gene_symbol}_weights.tsv             non-zero ElasticNet weights
  cv_performance{_gene_index}.tsv       per-gene CV R², model stats

CLI usage (single job):
  python train_imputation_models.py \
      --genotype_dir out/analysis/pwas/filtered_genotypes/pastl_250K/ \
      --output_dir   out/analysis/pwas/models/pastl_250K/

SLURM array usage (N genes per task):
  python train_imputation_models.py \
      --genotype_dir ... --output_dir ... \
      --gene_index $((SLURM_ARRAY_TASK_ID * N)) --n_genes N
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from preprocessing.build_sequences_bed import load_gct_matrix

_DEFAULT_GENO_BASE = "out/analysis/pwas/filtered_genotypes"
_DEFAULT_OUT_BASE  = "out/analysis/pwas/models"

ALPHAS    = np.logspace(-4, 1, 20)
L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
MIN_SAMPLES  = 50
MIN_VARIANTS = 2

IMPUTE_N_SD_BELOW = 1.8   # Perseus default: shift mean down by 1.8 σ
IMPUTE_WIDTH      = 0.3   # Perseus default: imputed std = 0.3 × observed σ


# ---------------------------------------------------------------------------
# Perseus-style left-censored imputation
# ---------------------------------------------------------------------------

def impute_protein_matrix(
    prot_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Replace NaN entries with values drawn from the lower tail of each
    patient's (column's) observed protein-abundance distribution.

    For each patient column:
      imputed ~ N(μ_obs − IMPUTE_N_SD_BELOW × σ_obs,  (IMPUTE_WIDTH × σ_obs)²)

    This mirrors the Perseus left-censored imputation used in build_sequences_bed
    compute_targets(), but uses global (all-protein) per-patient statistics rather
    than train-gene-only statistics, since the PWAS has no gene-level train/test
    concept for patients.

    Parameters
    ----------
    prot_df : DataFrame  shape (proteins × patients), may contain NaN
    seed    : int        random seed for reproducibility

    Returns
    -------
    DataFrame with the same shape, NaN replaced by imputed values.
    """
    np.random.seed(seed)
    imputed = prot_df.copy()

    for col in imputed.columns:          # each column = one patient
        missing = imputed[col].isnull()
        if not missing.any():
            continue
        observed = imputed[col].dropna().values
        if len(observed) < 2:
            # Too few observations to estimate a distribution — use global floor.
            imputed.loc[missing, col] = np.nanmin(imputed.values) - 1.0
            continue
        mu  = observed.mean()
        sig = observed.std(ddof=1)
        if sig == 0 or np.isnan(sig):
            imputed.loc[missing, col] = mu - IMPUTE_N_SD_BELOW
            continue
        imputed.loc[missing, col] = np.random.normal(
            loc   = mu  - IMPUTE_N_SD_BELOW * sig,
            scale = max(IMPUTE_WIDTH * sig, 1e-6),
            size  = missing.sum(),
        )

    n_imputed = int(prot_df.isnull().sum().sum())
    print(f"  Perseus imputation: filled {n_imputed:,} missing values "
          f"across {prot_df.shape[0]:,} proteins × {prot_df.shape[1]:,} patients")
    return imputed


# ---------------------------------------------------------------------------
# Sample ID mapping  (CPT aliquot ID → C3L/C3N case ID)
# ---------------------------------------------------------------------------

def build_sample_map(gvcf_dir: str) -> dict[str, str]:
    """
    Read the #CHROM line from each {case_id}.g.vcf.gz to extract the internal
    sample name (CPT aliquot ID).  Returns {vcf_sample_id: case_id}.
    """
    import gzip as gz
    mapping: dict[str, str] = {}
    gvcf_path = Path(gvcf_dir)
    for gvcf in sorted(gvcf_path.glob("*.g.vcf.gz")):
        case_id = gvcf.name.replace(".g.vcf.gz", "")
        try:
            with gz.open(gvcf, "rt") as fh:
                for line in fh:
                    if line.startswith("#CHROM"):
                        vcf_sample_id = line.strip().split("\t")[-1]
                        mapping[vcf_sample_id] = case_id
                        break
        except Exception as e:
            print(f"  [WARN] could not read {gvcf.name}: {e}")
    return mapping


# ---------------------------------------------------------------------------
# Per-gene model fitting
# ---------------------------------------------------------------------------

def fit_gene(
    gene: str,
    geno_path: Path,
    prot_df: pd.DataFrame,
    vcf_to_case: dict[str, str],
    n_folds: int,
) -> tuple[dict | None, pd.DataFrame | None]:
    """
    Fit ElasticNetCV for one gene.

    Returns (perf_dict, weights_df) or (None, None) if the gene is skipped.
    """
    try:
        geno = pd.read_csv(geno_path, sep="\t", index_col=0, low_memory=False)
    except Exception as e:
        print(f"  [WARN] {gene}: could not read dosage matrix: {e}")
        return None, None

    if geno.shape[1] < MIN_VARIANTS:
        return None, None

    # Map VCF sample IDs → case IDs; drop samples without a mapping.
    geno.index = geno.index.map(lambda s: vcf_to_case.get(s, None))
    geno = geno[geno.index.notna()].copy()
    geno.index = geno.index.astype(str)

    # Align to protein matrix.
    # prot_df has already been Perseus-imputed so there are no NaNs;
    # we simply intersect on sample IDs.
    if gene not in prot_df.index:
        return None, None
    prot = prot_df.loc[gene]          # no dropna — imputation already handled
    common = sorted(set(geno.index) & set(prot.index))
    if len(common) < MIN_SAMPLES:
        return None, None

    X = geno.loc[common].astype(float)
    y = prot.loc[common].astype(float)

    # Impute missing dosages to per-variant column mean.
    col_means = X.mean(axis=0)
    X = X.fillna(col_means)

    X = X.dropna(axis=1)
    if X.shape[1] < MIN_VARIANTS:
        return None, None

    # Z-score the protein target (standard PWAS practice).
    y_std = float(y.std(ddof=1))
    if y_std == 0 or np.isnan(y_std):
        return None, None
    y_norm = (y - y.mean()) / y_std

    X_arr = X.values

    # --- Hyperparameter tuning (inner CV) ---
    kf_inner = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    pipe_tune = Pipeline([
        ("scaler", StandardScaler()),
        ("enet",   ElasticNetCV(
            alphas=ALPHAS,
            l1_ratio=L1_RATIOS,
            cv=kf_inner,
            max_iter=50_000,
            n_jobs=1,
        )),
    ])
    pipe_tune.fit(X_arr, y_norm)

    best_alpha    = pipe_tune.named_steps["enet"].alpha_
    best_l1_ratio = pipe_tune.named_steps["enet"].l1_ratio_

    # --- Evaluation (outer CV, different folds) ---
    kf_outer = KFold(n_splits=n_folds, shuffle=True, random_state=123)
    pipe_eval = Pipeline([
        ("scaler", StandardScaler()),
        ("enet",   ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio,
                              max_iter=50_000)),
    ])
    cv_scores = cross_val_score(pipe_eval, X_arr, y_norm, cv=kf_outer, scoring="r2")
    cv_r2 = float(np.mean(cv_scores))

    # --- Final model for weight extraction ---
    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("enet",   ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio,
                              max_iter=50_000)),
    ])
    final_pipe.fit(X_arr, y_norm)

    coefs = final_pipe.named_steps["enet"].coef_ / final_pipe.named_steps["scaler"].scale_
    nonzero_mask = coefs != 0
    n_nonzero = int(nonzero_mask.sum())

    weights_df = None
    if n_nonzero > 0:
        weights_df = pd.DataFrame({
            "variant_id": X.columns[nonzero_mask],
            "weight":     coefs[nonzero_mask],
        })

    perf = {
        "gene_symbol":       gene,
        "cv_r2":             round(cv_r2, 6),
        "n_variants":        X.shape[1],
        "n_nonzero_weights": n_nonzero,
        "best_alpha":        round(best_alpha, 8),
        "best_l1_ratio":     round(best_l1_ratio, 4),
        "n_samples":         len(common),
    }
    return perf, weights_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--genotype_dir", required=True,
                        help="Path to one filtered_genotypes/{model}_{threshold}/ dir")
    parser.add_argument("--output_dir",   required=True,
                        help="Output directory for weights and cv_performance.tsv")
    parser.add_argument("--gct_paths", nargs="+", required=True,
                        help="CPTAC proteomics GCT files (discovery + confirmatory)")
    parser.add_argument("--gvcf_dir", required=True,
                        help="Directory containing per-sample *.g.vcf.gz files")
    parser.add_argument("--n_folds",  type=int, default=5)
    parser.add_argument("--r2_threshold", type=float, default=0.01,
                        help="R² threshold for summary logging (default: 0.01)")
    parser.add_argument("--gene_index", type=int, default=None,
                        help="0-based start index for SLURM array slicing")
    parser.add_argument("--n_genes", type=int, default=None,
                        help="Genes to process per job; required with --gene_index")
    args = parser.parse_args()

    if args.gene_index is not None and args.n_genes is None:
        parser.error("--n_genes is required when --gene_index is set")

    geno_dir = Path(args.genotype_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading protein matrix (sample_type='Solid Tissue Normal') ...")
    prot_df = load_gct_matrix(args.gct_paths, sample_type="Solid Tissue Normal")

    print("Applying Perseus left-censored imputation to protein matrix ...")
    prot_df = impute_protein_matrix(prot_df)

    print(f"Building sample ID mapping from {args.gvcf_dir} ...")
    vcf_to_case = build_sample_map(args.gvcf_dir)
    print(f"  Mapped {len(vcf_to_case):,} VCF sample IDs to case IDs")

    gene_files = sorted(geno_dir.glob("*.tsv.gz"))
    if args.gene_index is not None:
        lo = args.gene_index
        hi = min(lo + args.n_genes, len(gene_files))
        gene_files = gene_files[lo:hi]
        print(f"Array slice: genes[{lo}:{hi}]  ({len(gene_files):,} genes this task)")

    print(f"\nFitting models for {len(gene_files):,} genes ...")
    print(f"  Genotype dir : {geno_dir}")
    print(f"  Output dir   : {out_dir}")
    print(f"  CV folds     : {args.n_folds}")

    perf_rows: list[dict] = []
    n_attempted = 0

    for i, geno_path in enumerate(gene_files):
        gene = geno_path.name.replace(".tsv.gz", "")
        if (i + 1) % 100 == 0:
            print(f"  {i + 1:,} / {len(gene_files):,} genes ...")

        n_attempted += 1
        perf, weights_df = fit_gene(
            gene, geno_path, prot_df, vcf_to_case, args.n_folds
        )
        if perf is None:
            continue

        perf_rows.append(perf)

        if weights_df is not None:
            weights_df.to_csv(
                out_dir / f"{gene}_weights.tsv", sep="\t", index=False
            )

    perf_df = pd.DataFrame(perf_rows) if perf_rows else pd.DataFrame(columns=[
        "gene_symbol", "cv_r2", "n_variants", "n_nonzero_weights",
        "best_alpha", "best_l1_ratio", "n_samples",
    ])

    suffix = f"_{args.gene_index}" if args.gene_index is not None else ""
    perf_path = out_dir / f"cv_performance{suffix}.tsv"
    perf_df.to_csv(perf_path, sep="\t", index=False)

    n_fit = len(perf_df)
    n_good = int((perf_df["cv_r2"] > args.r2_threshold).sum()) if n_fit else 0
    med_r2 = float(perf_df["cv_r2"].median()) if n_fit else float("nan")
    med_nz = float(perf_df["n_nonzero_weights"].median()) if n_fit else float("nan")

    print()
    print("=" * 50)
    print(f"Genes attempted   : {n_attempted:,}")
    print(f"Genes fit         : {n_fit:,}")
    print(f"CV R² > {args.r2_threshold:.2f}      : {n_good:,}")
    print(f"Median CV R²      : {med_r2:.4f}")
    print(f"Median non-zero w : {med_nz:.0f}")
    print(f"CV performance    : {perf_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()