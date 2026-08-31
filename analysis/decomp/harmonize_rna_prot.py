r"""
Align CPTAC LUAD RNA-seq and proteomics into matched genes x patients matrices.

Loads Solid Tissue Normal samples from CPTAC-3 LUAD proteomics (GCT, log2 TMT
ratio) and RNA-seq (per-patient TSVs, TPM), intersects to a common gene and
patient set, applies log2(TPM+1) to the RNA matrix, and writes rna_aligned_log2.csv
and prot_aligned.csv. --rna_base_dir optionally reroots the RNA file paths from the
index, preserving the trailing {uuid_dir}/{filename} structure.

Usage:
  python harmonize_rna_prot.py \
      --gct_paths /path/to/CPTAC_LUAD_discovery_proteome.gct \
                  /path/to/CPTAC_LUAD_confirmatory_proteome.gct \
      --rna_index /path/to/cptac3_rna_index_simple.tsv \
      --output_dir out/analysis/decomp/
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_cptac_gct_normals(paths: list[str]) -> pd.DataFrame:
    """Extract Solid Tissue Normal protein abundance from GCT files."""
    print("Loading proteomics GCT files...")
    combined = []
    for p in paths:
        df = pd.read_csv(p, sep="\t", skiprows=2, low_memory=False)
        type_mask = df[df["id"] == "sample_type"].iloc[0, 4:]
        is_normal = type_mask == "Solid Tissue Normal"
        id_row = df[df["id"] == "diagnosis_submitter_id"].iloc[0, 4:]
        clean_ids = id_row[is_normal].str.replace("-DX", "", regex=False).values
        data_start = df[df["id"] == "A1BG"].index[0]
        data_df = df.iloc[data_start:].set_index("id").iloc[:, 3:]
        data_df = data_df.loc[:, is_normal.values]
        data_df.columns = clean_ids
        data_df = data_df.apply(pd.to_numeric, errors="coerce")
        combined.append(data_df)
    result = pd.concat(combined, axis=1, join="inner")
    result = result.T.groupby(level=0).mean().T
    print(f"  Proteome: {result.shape[0]:,} genes × {result.shape[1]:,} patients")
    return result


def resolve_rna_path(file_path: str, base_dir: str | None) -> str:
    """Return the resolved file path, optionally rerooting under base_dir.

    When base_dir is given the last two path components (uuid_dir/filename)
    are preserved and joined to base_dir, replacing whatever root the index
    was built with.
    """
    if base_dir is None:
        return file_path
    p = Path(file_path)
    return str(Path(base_dir) / p.parent.name / p.name)


def load_rna_normals(index_path: str, base_dir: str | None = None) -> pd.DataFrame:
    """Build a per-patient log2(TPM+1) matrix from an index of TSV files."""
    print("Loading RNA-seq files...")
    index_df = pd.read_csv(index_path, sep="\t")
    normals = index_df[index_df["sample_type"] == "Solid Tissue Normal"]
    if normals.empty:
        raise ValueError("No 'Solid Tissue Normal' samples found in the RNA index.")
    print(f"  Found {len(normals)} normal RNA samples in index")

    series = []
    n_missing = 0
    for _, row in normals.iterrows():
        pid = row["case_submitter_id"]
        fp = resolve_rna_path(row["file_path"], base_dir)
        if not os.path.exists(fp):
            print(f"  [warn] missing {pid}: {fp}")
            n_missing += 1
            continue
        df = pd.read_csv(fp, sep="\t", skiprows=1,
                         usecols=["gene_name", "tpm_unstranded"])
        df = (df[~df["gene_name"].str.startswith("N_", na=False)]
              .dropna(subset=["gene_name"]))
        # Collapse duplicates by taking max TPM per gene symbol
        df = df.groupby("gene_name")["tpm_unstranded"].max()
        df.name = pid
        series.append(df)

    if not series:
        raise ValueError(f"All {n_missing} RNA files were missing from disk.")
    rna = pd.concat(series, axis=1)
    print(f"  RNA (raw TPM): {rna.shape[0]:,} genes × {rna.shape[1]:,} patients")
    return rna


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gct_paths", nargs="+", required=True,
                        help="One or more CPTAC proteomics GCT files.")
    parser.add_argument("--rna_index", required=True,
                        help="TSV index with columns: case_submitter_id, sample_type, "
                             "file_path.")
    parser.add_argument("--rna_base_dir", default=None,
                        help="Optional root directory for the RNA TSV files. Replaces "
                             "the path prefix in the index while keeping the "
                             "uuid_dir/filename structure.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write rna_aligned_log2.csv and prot_aligned.csv.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prot_df = load_cptac_gct_normals(args.gct_paths)
    rna_df = load_rna_normals(args.rna_index, base_dir=args.rna_base_dir)

    common_genes = sorted(set(prot_df.index) & set(rna_df.index))
    common_patients = sorted(set(prot_df.columns) & set(rna_df.columns))

    prot_matched = prot_df.loc[common_genes, common_patients]
    rna_matched = rna_df.loc[common_genes, common_patients]

    print(f"\nIntersection: {len(common_genes):,} genes × {len(common_patients):,} patients")

    rna_log = np.log2(rna_matched + 1.0)

    rna_out = out_dir / "rna_aligned_log2.csv"
    prot_out = out_dir / "prot_aligned.csv"
    rna_log.to_csv(rna_out)
    prot_matched.to_csv(prot_out)

    print(f"\nSaved:")
    print(f"  {rna_out}")
    print(f"  {prot_out}")

    # Report the cross-gene TMT-protein vs RNA correlation as a diagnostic
    rna_med = rna_log.median(axis=1)
    prot_med = prot_matched.median(axis=1)
    valid = ~rna_med.isna() & ~prot_med.isna()
    r = np.corrcoef(rna_med[valid], prot_med[valid])[0, 1]
    print(f"\nDiagnostic — cross-gene (raw TMT median vs RNA median) r = {r:.4f}")


if __name__ == "__main__":
    main()
