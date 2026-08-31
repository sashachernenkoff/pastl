r"""
Build the GWAS variant→gene-window manifest (with VEP consequence classes).

Input: an OpenGWAS-format LUAD GWAS VCF (ebi-a-GCST004744.vcf.gz), GRCh37, whose
per-variant DATA column encodes ES:SE:LP:AF:ID (ES = effect size, LP = -log10 p).

Steps:
  1. Parse the VCF → CHROM/POS/REF/ALT + EMPIRICAL_EFFECT (ES) + P_VALUE (10**-LP).
  2. Lift hg19 → hg38 with a local chain file.
  3. Map each variant to every protein window that contains it (one row per
     variant×gene) via data/sequences_protein.bed. GWAS variants have no
     pre-assigned target gene, so gene assignment is positional.
  4. Run Ensembl VEP on the unique variant×gene set and attach a coarse
     consequence class per variant against its assigned gene.

Outputs (model-independent; shared by PASTL and Enformer scoring):
  {output_dir}/gwas_manifest.tsv            positional map, no consequence
  {output_dir}/gwas_manifest_annotated.tsv  + primary_consequence / consequence_class / IMPACT

Usage:
  python build_variants.py \
      --gwas_vcf /path/to/ebi-a-GCST004744.vcf.gz \
      --sequences_bed data/sequences_protein.bed \
      --fasta /path/to/GRCh38.d1.vd1.fa \
      --chain /path/to/hg19ToHg38.over.chain.gz \
      --vep_gtf /path/to/Homo_sapiens.GRCh38.105.sorted.gtf.gz \
      --vep_fasta /path/to/hg38.fa \
      --output_dir out/predictions/variants
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from variant_common import load_windows, liftover, map_to_all_windows, finalize_manifest
from gwas_config import CONSEQUENCE_MAP, SEVERITY_ORDER


# ---------------------------------------------------------------------------
# Step 1 — parse the OpenGWAS VCF
# ---------------------------------------------------------------------------

def parse_gwas_vcf(path: str, limit: int | None) -> pd.DataFrame:
    print(f"[GWAS] reading {path}")
    raw = pd.read_csv(path, sep="\t", comment="#", header=None, low_memory=False)
    raw.columns = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER",
                   "INFO", "FORMAT", "DATA"][:raw.shape[1]]
    if limit:
        raw = raw.head(limit)
    raw = raw.dropna(subset=["CHROM", "POS"]).copy()

    fmt = str(raw["FORMAT"].iloc[0]).split(":")
    es_idx = fmt.index("ES") if "ES" in fmt else 0
    lp_idx = fmt.index("LP") if "LP" in fmt else 2
    data = raw["DATA"].astype(str).str.split(":", expand=True)

    chrom = raw["CHROM"].astype(str)
    chrom = np.where(chrom.str.startswith("chr"), chrom, "chr" + chrom)
    lp = pd.to_numeric(data[lp_idx], errors="coerce")

    var = pd.DataFrame({
        "CHROM":            chrom,
        "POS":              raw["POS"].astype(int),
        "REF":              raw["REF"].astype(str),
        "ALT":              raw["ALT"].astype(str).str.split(",").str[0],
        "EMPIRICAL_EFFECT": pd.to_numeric(data[es_idx], errors="coerce").fillna(0.0),
        "P_VALUE":          10 ** (-lp),
    })
    var = var.dropna(subset=["P_VALUE"]).reset_index(drop=True)
    print(f"  {len(var):,} variants parsed")
    return var


# ---------------------------------------------------------------------------
# Step 4 — VEP consequence annotation
# ---------------------------------------------------------------------------

def _match_key(chrom, pos, ref, alt, gene):
    c = str(chrom).replace("chr", "")
    return f"{c}_{int(pos)}_{ref}_{alt}_{gene}"


def write_vep_vcf(manifest: pd.DataFrame, vcf_path: Path):
    uniq = manifest.drop_duplicates(subset=["CHROM", "POS", "REF", "ALT", "TARGET_GENE"]).copy()
    print(f"  writing {len(uniq):,} unique variant×gene rows to {vcf_path}")
    c = uniq["CHROM"].astype(str).str.replace("chr", "", regex=False)
    pos = uniq["POS"].astype(int).astype(str)
    ref, alt, gene = uniq["REF"].astype(str), uniq["ALT"].astype(str), uniq["TARGET_GENE"].astype(str)
    vid = c + "_" + pos + "_" + ref + "_" + alt + "_" + gene    # matches _match_key
    body = c + "\t" + pos + "\t" + vid + "\t" + ref + "\t" + alt + "\t.\t.\t."
    with open(vcf_path, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n##reference=GRCh38\n")
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        fh.write("\n".join(body.tolist()) + "\n")


def run_vep(vcf_path: Path, vep_out: Path, vep_bin: str, vep_gtf: str, vep_fasta: str):
    cmd = [
        vep_bin, "--input_file", str(vcf_path), "--output_file", str(vep_out),
        "--format", "vcf", "--gtf", vep_gtf, "--fasta", vep_fasta,
        "--tab", "--canonical", "--per_gene", "--no_stats", "--force_overwrite",
        "--fields", "Uploaded_variation,Location,Allele,Gene,Feature,Consequence,IMPACT,CANONICAL,DISTANCE",
        "--fork", "8",
    ]
    print("  running VEP:\n    " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_vep(vep_out: Path) -> pd.DataFrame:
    rows, header = [], None
    with open(vep_out) as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                header = line.lstrip("#").strip().split("\t")
                continue
            rows.append(line.rstrip("\n").split("\t"))
    vep = pd.DataFrame(rows, columns=header).replace("-", np.nan)

    # VEP reports GENCODE gene ids WITH a version (ENSG...N); TARGET_GENE is
    # versionless, so strip the version before matching.
    vep["Gene"] = vep["Gene"].str.split(".").str[0]
    vep["target_from_id"] = vep["Uploaded_variation"].str.rsplit("_", n=1).str[-1]
    all_ids = set(vep["Uploaded_variation"].unique())

    matched = vep[vep["Gene"] == vep["target_from_id"]].copy()
    matched["so_terms"] = matched["Consequence"].fillna("unknown").str.split(",")
    exploded = matched.explode("so_terms")
    exploded["rank"] = exploded["so_terms"].apply(
        lambda t: SEVERITY_ORDER.index(t) if t in SEVERITY_ORDER else len(SEVERITY_ORDER))
    best = exploded.sort_values("rank").drop_duplicates(subset=["Uploaded_variation"]).copy()
    best["primary_consequence"] = best["so_terms"]
    best["consequence_class"] = best["primary_consequence"].map(CONSEQUENCE_MAP).fillna("other")

    unmatched = all_ids - set(best["Uploaded_variation"])
    unann = pd.DataFrame({
        "Uploaded_variation": list(unmatched),
        "primary_consequence": "unannotated",
        "consequence_class": "unannotated",
        "IMPACT": pd.NA,
    })
    out = pd.concat([best, unann], ignore_index=True)
    print(f"  VEP: {len(all_ids) - len(unmatched):,} annotated, {len(unmatched):,} unannotated")
    return out[["Uploaded_variation", "primary_consequence", "consequence_class", "IMPACT"]]


def annotate(manifest: pd.DataFrame, vep_df: pd.DataFrame) -> pd.DataFrame:
    manifest = manifest.copy()
    manifest["_key"] = [
        _match_key(c, p, r, a, g) for c, p, r, a, g in zip(
            manifest["CHROM"], manifest["POS"], manifest["REF"],
            manifest["ALT"], manifest["TARGET_GENE"])
    ]
    vep_df = vep_df.rename(columns={"Uploaded_variation": "_key"})
    merged = manifest.merge(vep_df, on="_key", how="left").drop(columns=["_key"])
    merged["consequence_class"] = merged["consequence_class"].fillna("unannotated")
    merged["primary_consequence"] = merged["primary_consequence"].fillna("unannotated")
    pct = 100 * (merged["consequence_class"] != "unannotated").mean()
    print(f"  annotated {pct:.1f}% of manifest rows (rest are >±5kb from the target "
          f"gene body -> no consequence for the scored gene, as expected)")
    if pct < 3:
        print("  ⚠  near-zero annotation — check gene-id versions / chr prefix / VEP GTF")
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gwas_vcf", default=None)
    parser.add_argument("--sequences_bed", required=True)
    parser.add_argument("--fasta", default=None, help="Reference FASTA (unused unless REF missing).")
    parser.add_argument("--chain", default=None, help="Local hg19→hg38 chain file.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    # VEP
    parser.add_argument("--vep_bin", default="vep")
    parser.add_argument("--vep_gtf", default=None, help="Bgzipped+tabix GENCODE/Ensembl GTF for VEP.")
    parser.add_argument("--vep_fasta", default=None, help="GRCh38 FASTA for VEP.")
    parser.add_argument("--skip_vep", action="store_true",
                        help="Skip VEP; consequence_class set to 'unannotated'.")
    parser.add_argument("--annotate_only", action="store_true",
                        help="Skip parsing/liftover/mapping; just VEP-annotate the existing "
                             "gwas_manifest.tsv in --output_dir (no re-scoring needed).")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Annotate-only: VEP an existing manifest and exit (decoupled from prediction).
    if args.annotate_only:
        if not (args.vep_gtf and args.vep_fasta):
            parser.error("--vep_gtf and --vep_fasta are required for --annotate_only")
        manifest = pd.read_csv(out_dir / "gwas_manifest.tsv", sep="\t")
        print(f"Annotate-only: {len(manifest):,} variant×gene rows from gwas_manifest.tsv")
        vcf_path, vep_out = out_dir / "gwas_for_vep.vcf", out_dir / "gwas_vep.txt"
        write_vep_vcf(manifest, vcf_path)
        run_vep(vcf_path, vep_out, args.vep_bin, args.vep_gtf, args.vep_fasta)
        annotate(manifest, parse_vep(vep_out)).to_csv(
            out_dir / "gwas_manifest_annotated.tsv", sep="\t", index=False)
        print(f"-> {out_dir / 'gwas_manifest_annotated.tsv'}")
        return

    if not (args.gwas_vcf and args.chain):
        parser.error("--gwas_vcf and --chain are required (unless --annotate_only)")

    windows = load_windows(args.sequences_bed)
    print(f"Loaded {len(windows):,} protein windows")

    var = parse_gwas_vcf(args.gwas_vcf, args.limit)
    var = liftover(var, args.chain)
    mapped = map_to_all_windows(var, windows)
    manifest = finalize_manifest(mapped, ["EMPIRICAL_EFFECT", "P_VALUE"], "LUAD_GWAS")
    print(f"Mapped to {len(manifest):,} variant×gene rows "
          f"({manifest['TARGET_GENE'].nunique():,} genes)")
    manifest.to_csv(out_dir / "gwas_manifest.tsv", sep="\t", index=False)

    if args.skip_vep:
        manifest = manifest.copy()
        manifest["primary_consequence"] = "unannotated"
        manifest["consequence_class"] = "unannotated"
        manifest["IMPACT"] = pd.NA
        annotated = manifest
    else:
        if not (args.vep_gtf and args.vep_fasta):
            parser.error("--vep_gtf and --vep_fasta are required unless --skip_vep is set")
        vcf_path = out_dir / "gwas_for_vep.vcf"
        vep_out = out_dir / "gwas_vep.txt"
        write_vep_vcf(manifest, vcf_path)
        run_vep(vcf_path, vep_out, args.vep_bin, args.vep_gtf, args.vep_fasta)
        annotated = annotate(manifest, parse_vep(vep_out))

    annotated.to_csv(out_dir / "gwas_manifest_annotated.tsv", sep="\t", index=False)
    print(f"-> {out_dir / 'gwas_manifest_annotated.tsv'}")


if __name__ == "__main__":
    main()
