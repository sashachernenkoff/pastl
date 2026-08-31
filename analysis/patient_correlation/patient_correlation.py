r"""
Patient-specific PASTL predictions and proteomics correlation.

For each CPTAC LUAD patient with germline genotype data, reconstructs a
patient-specific DNA sequence by inserting their SNVs (SNVs only, no indels)
into the reference genome, runs PASTL to predict their protein levels, and
correlates the predictions with observed Solid Tissue Normal protein levels.

Two sequences are constructed per patient × gene:
  - hom_only:  only homozygous alt (1/1) variants applied to the reference
  - het_hom:   all non-ref variants (0/1 and 1/1) applied

Their one-hot encodings are averaged (equal weight), encoding allele dosage
continuously: hom-ref -> REF one-hot, het -> 0.5*REF + 0.5*ALT, hom-alt -> ALT
one-hot. The averaged tensor is a valid continuous input to Enformer.

Per-gene predictions are saved to predictions/{symbol}.npy as they are computed,
so the job can be resumed without re-scoring completed genes.

Outputs (all in --output_dir):
  predictions/{symbol}.npy          float32 [n_patients], one per patient
  tables/patient_ids.txt            ordered patient IDs (row order matches .npy)
  tables/gene_order.txt             ordered gene symbols
  tables/per_gene_correlations.csv
  figures/patient_r_histogram.{png,pdf}
  figures/patient_r_vs_protein_level.{png,pdf}
  logs/summary.txt

Usage:
  python patient_correlation.py \
      --config model/configs/config.yaml \
      --wandb_run_id s5fo3aak --checkpoint_name best.pt --attention_head \
      --vcf /path/to/cohort.clean.norm.vcf.gz \
      --gvcf_dir /path/to/gvcfs \
      --gct /path/to/CPTAC_LUAD_discovery_proteome.gct \
            /path/to/CPTAC_LUAD_confirmatory_proteome.gct \
      --output_dir out/analysis/s5fo3aak_best/patient_correlation/
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from tqdm import tqdm
import pysam
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "model"))

import palette
from config import PASTLConfig
from dataset import CPTACDataset
from model import PASTLFinetune

_PREDICTIONS_DIR = "out/predictions"
_CHECKPOINTS_DIR = "out/model/checkpoints"
_DEFAULT_SEQUENCES_BED = str(
    Path(__file__).parent.parent.parent / "data" / "sequences_protein.bed"
)

WINDOW_SIZE = 196_608
HALF_WINDOW  = WINDOW_SIZE // 2

MIN_PATIENTS = 5   # minimum patients with non-NaN protein value to compute r


# ---------------------------------------------------------------------------
# GCT loading
# ---------------------------------------------------------------------------

def _read_gct_dims(path: str) -> tuple[int, int]:
    """Parse the GCT 1.3 dimension line.

    Format: nrow ncol nrm ncm
      nrm (parts[2]) = row metadata cols  = gene annotation cols (excl. "id")
      ncm (parts[3]) = col metadata rows  = sample metadata rows

    Returns (n_sample_meta_rows, n_gene_meta_cols).
    """
    with open(path) as fh:
        fh.readline()           # version ("#1.3" or "1.3")
        parts = fh.readline().strip().split("\t")
    if len(parts) >= 4:
        return int(parts[3]), int(parts[2])   # ncm, nrm
    return 0, 0


def load_cptac_gct_normals(paths: list[str]) -> pd.DataFrame:
    """
    Extract Solid Tissue Normal protein abundance from CPTAC GCT files.

    Returns a genes × patients DataFrame of log2 TMT ratios indexed by gene
    symbol.  Patients from multiple GCT files are concatenated; duplicate
    patient columns are averaged.
    """
    print("Loading CPTAC GCT files (Solid Tissue Normal)...")
    combined = []
    for p in paths:
        n_col_meta, n_row_meta = _read_gct_dims(p)
        df = pd.read_csv(p, sep="\t", skiprows=2, low_memory=False)

        # --- Sample metadata (first n_col_meta rows) ---
        meta = df.iloc[:n_col_meta].set_index("id")

        if "sample_type" not in meta.index:
            raise ValueError(f"'sample_type' metadata row not found in {p}")
        if "diagnosis_submitter_id" not in meta.index:
            raise ValueError(f"'diagnosis_submitter_id' metadata row not found in {p}")

        # Columns: "id", [n_row_meta annotation cols], [patient cols]
        # iloc[n_row_meta:] on the post-set_index frame skips annotation cols
        type_vals = meta.loc["sample_type"].iloc[n_row_meta:]
        id_vals   = meta.loc["diagnosis_submitter_id"].iloc[n_row_meta:]

        is_normal  = type_vals == "Solid Tissue Normal"
        clean_ids  = id_vals[is_normal.values].str.replace("-DX", "", regex=False).values

        # --- Gene data (rows after n_col_meta) ---
        data_df = df.iloc[n_col_meta:].set_index("id").iloc[:, n_row_meta:]
        data_df = data_df.loc[:, is_normal.values]
        data_df.columns = clean_ids
        data_df = data_df.apply(pd.to_numeric, errors="coerce")

        print(f"  {Path(p).name}: {data_df.shape[0]:,} genes, "
              f"{int(is_normal.sum()):,} normal samples")
        combined.append(data_df)

    result = pd.concat(combined, axis=1, join="inner")
    # Average duplicate patient columns (e.g. repeated measurements)
    result = result.T.groupby(level=0).mean().T
    print(f"  Combined: {result.shape[0]:,} genes × {result.shape[1]:,} patients\n")
    return result


# ---------------------------------------------------------------------------
# VCF helpers
# ---------------------------------------------------------------------------

def get_vcf_samples(vcf_path: str) -> list[str]:
    """Return the ordered list of sample names from the VCF header."""
    with pysam.VariantFile(vcf_path) as vcf:
        return list(vcf.header.samples)


def build_sample_map(gvcf_dir: str) -> dict[str, str]:
    """
    Build a CPT-aliquot-ID → case-ID mapping from individual gVCF headers.

    Each gVCF is named {case_id}.g.vcf.gz; the #CHROM header line's last
    column is the internal sample name (CPT aliquot ID) used in the cohort VCF.
    Returns {cpt_aliquot_id: case_id} (e.g. {'CPT0000020163': 'C3L-00001'}).
    """
    import gzip as gz
    mapping: dict[str, str] = {}
    for gvcf in sorted(Path(gvcf_dir).glob("*.g.vcf.gz")):
        case_id = gvcf.name.replace(".g.vcf.gz", "")
        try:
            with gz.open(gvcf, "rt") as fh:
                for line in fh:
                    if line.startswith("#CHROM"):
                        cpt_id = line.strip().split("\t")[-1]
                        mapping[cpt_id] = case_id
                        break
        except Exception as exc:
            print(f"  [warn] could not read {gvcf.name}: {exc}")
    return mapping


def match_patients(
    vcf_samples: list[str],
    prot_patients: list[str],
    cpt_to_case: dict[str, str],
) -> dict[str, str]:
    """
    Map proteomics case IDs → VCF sample names for the intersection.

    vcf_samples:   CPT aliquot IDs from the cohort VCF header.
    prot_patients: C3L/C3N case IDs from the proteomics GCT.
    cpt_to_case:   {CPT aliquot ID → case ID} built from per-patient gVCF headers.

    Returns {prot_case_id: vcf_sample_name}.
    """
    # Invert: case_id → CPT aliquot ID (for samples present in the cohort VCF)
    vcf_sample_set = set(vcf_samples)
    case_to_cpt = {
        case: cpt for cpt, case in cpt_to_case.items()
        if cpt in vcf_sample_set
    }
    result = {
        prot_id: case_to_cpt[prot_id]
        for prot_id in prot_patients
        if prot_id in case_to_cpt
    }
    print(f"Patient matching: {len(vcf_samples):,} VCF samples, "
          f"{len(prot_patients):,} proteomics patients, "
          f"{len(cpt_to_case):,} gVCF mappings → "
          f"{len(result):,} shared\n")
    return result


def _vcf_chrom(vcf: pysam.VariantFile, chrom: str) -> str:
    """Return the chromosome name as stored in the VCF (with or without 'chr')."""
    contigs = set(vcf.header.contigs)
    if chrom in contigs:
        return chrom
    alt = chrom.lstrip("chr") if chrom.startswith("chr") else f"chr{chrom}"
    if alt in contigs:
        return alt
    return chrom   # will likely raise on fetch; caller handles


def fetch_snvs_in_window(
    vcf_path: str,
    chrom: str,
    seq_start: int,
    seq_end: int,
    vcf_samples: list[str],
) -> dict[str, list[tuple[int, str, str, bool]]]:
    """
    Fetch biallelic SNVs in the window [seq_start, seq_end) (0-based, half-open)
    and return per-sample variant lists.

    Returns:
        dict mapping vcf_sample_name → list of (rel_pos, ref_base, alt_base, is_hom).
        rel_pos is 0-based relative to seq_start.

    Only processes records where:
      · REF and ALT are both single bases (SNV, no indels)
      · Exactly one ALT allele (biallelic)
      · Genotype is non-missing
    """
    result: dict[str, list] = {s: [] for s in vcf_samples}

    try:
        with pysam.VariantFile(vcf_path) as vcf:
            chrom_key = _vcf_chrom(vcf, chrom)
            for rec in vcf.fetch(chrom_key, seq_start, seq_end):
                # SNV filter
                if len(rec.ref) != 1:
                    continue
                alts = rec.alts
                if alts is None or len(alts) != 1 or len(alts[0]) != 1:
                    continue

                ref_base = rec.ref.upper()
                alt_base = alts[0].upper()
                rel_pos  = rec.pos - seq_start   # rec.pos is 0-based in pysam

                if rel_pos < 0 or rel_pos >= WINDOW_SIZE:
                    continue

                for sample in vcf_samples:
                    gt = rec.samples[sample]["GT"]
                    if gt is None or None in gt:
                        continue
                    n_alt = sum(1 for a in gt if a is not None and a > 0)
                    if n_alt == 0:
                        continue
                    is_hom = n_alt == len(gt)
                    result[sample].append((rel_pos, ref_base, alt_base, is_hom))

    except (ValueError, KeyError) as exc:
        print(f"  [warn] VCF fetch {chrom}:{seq_start}-{seq_end}: {exc}")

    return result


# ---------------------------------------------------------------------------
# Patient-specific one-hot encoding
# ---------------------------------------------------------------------------

def build_patient_encoding(
    ref_seq: str,
    variants: list[tuple[int, str, str, bool]],
    cpt: CPTACDataset,
    strand: str,
) -> torch.Tensor:
    """
    Build the averaged one-hot encoding for a single patient.

    Constructs two sequences from the reference:
      · hom_only: only 1/1 (homozygous alt) variants applied
      · het_hom:  all non-ref variants (0/1 and 1/1) applied

    Returns 0.5 * one_hot(hom_only) + 0.5 * one_hot(het_hom), which encodes
    allele dosage: hom-ref → pure REF one-hot, het → 0.5 REF + 0.5 ALT,
    hom-alt → pure ALT one-hot.

    Args:
        ref_seq:  + strand reference sequence, length WINDOW_SIZE.
        variants: list of (rel_pos, ref_base, alt_base, is_hom).
        cpt:      CPTACDataset instance (for _one_hot / rev_comp_trans).
        strand:   '+' or '-'.

    Returns:
        [WINDOW_SIZE, 4] float32 tensor.
    """
    seq_hom = list(ref_seq)
    seq_all = list(ref_seq)

    for rel_pos, _ref, alt, is_hom in variants:
        if is_hom:
            seq_hom[rel_pos] = alt
        seq_all[rel_pos] = alt

    s_hom = "".join(seq_hom)
    s_all = "".join(seq_all)

    if strand == "-":
        s_hom = s_hom.translate(cpt.rev_comp_trans)[::-1]
        s_all = s_all.translate(cpt.rev_comp_trans)[::-1]

    return 0.5 * cpt._one_hot(s_hom) + 0.5 * cpt._one_hot(s_all)


# ---------------------------------------------------------------------------
# Reference sequence
# ---------------------------------------------------------------------------

def fetch_ref_window(fa: Fasta, chrom: str, seq_start: int, seq_end: int) -> str:
    """Fetch + strand reference sequence, N-padded at chromosome edges."""
    chrom_key   = chrom if chrom in fa else (
        chrom.lstrip("chr") if chrom.startswith("chr") else f"chr{chrom}"
    )
    fetch_start = max(0, seq_start)
    ref_seq     = str(fa[chrom_key][fetch_start:seq_end].seq.upper())
    left_pad    = max(0, -seq_start)
    right_pad   = WINDOW_SIZE - (left_pad + len(ref_seq))
    return (("N" * left_pad) + ref_seq + ("N" * max(0, right_pad)))[:WINDOW_SIZE]


# ---------------------------------------------------------------------------
# Model forward pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_patient_predictions(
    model: torch.nn.Module,
    encodings: list[torch.Tensor],
    gene_mask: torch.Tensor,
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> np.ndarray:
    """
    Run PASTL on a list of patient-specific one-hot tensors for one gene.

    Args:
        encodings:  list of [WINDOW_SIZE, 4] float32 tensors, one per patient.
        gene_mask:  [896] gene-body mask for the gene.

    Returns:
        [n_patients] float32 numpy array of raw model predictions.
    """
    preds        = []
    amp_enabled  = amp and torch.cuda.is_available()
    gmask_base   = gene_mask.to(device).float().unsqueeze(0)   # [1, 896]

    for i in range(0, len(encodings), batch_size):
        batch  = torch.stack(encodings[i: i + batch_size]).to(device)  # [B, W, 4]
        gmask  = gmask_base.expand(len(batch), -1)                      # [B, 896]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            out = model(batch, gene_mask=gmask).squeeze(-1)             # [B]
        preds.append(out.float().cpu().numpy())

    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(cfg: PASTLConfig, checkpoint: Path, device: torch.device):
    print(f"Loading PASTL checkpoint: {checkpoint}")
    model = PASTLFinetune(cfg).to(device)
    ckpt  = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def compute_correlations(
    pred_dir: Path,
    prot_df: pd.DataFrame,
    patient_ids: list[str],
    genes: list[str],
) -> pd.DataFrame:
    """Per-gene Pearson r between patient-specific PASTL predictions and observed protein."""
    records = []
    for gene in tqdm(genes, desc="Correlating"):
        pred_file = pred_dir / f"{gene}.npy"
        if not pred_file.exists():
            records.append({"gene": gene, "n_patients": 0,
                            "pearson_r": np.nan, "p_value": np.nan,
                            "mean_obs_protein": np.nan})
            continue

        preds = np.load(pred_file).astype(float)   # [n_patients]

        if gene not in prot_df.index:
            records.append({"gene": gene, "n_patients": 0,
                            "pearson_r": np.nan, "p_value": np.nan,
                            "mean_obs_protein": np.nan})
            continue

        obs  = prot_df.loc[gene, patient_ids].values.astype(float)
        valid = np.isfinite(obs) & np.isfinite(preds)
        n     = int(valid.sum())

        if n < MIN_PATIENTS:
            records.append({"gene": gene, "n_patients": n,
                            "pearson_r": np.nan, "p_value": np.nan,
                            "mean_obs_protein": float(np.nanmean(obs))})
            continue

        r, p = pearsonr(preds[valid], obs[valid])
        records.append({
            "gene":             gene,
            "n_patients":       n,
            "pearson_r":        float(r),
            "p_value":          float(p),
            "mean_obs_protein": float(np.nanmean(obs)),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, figs_dir: Path, stem: str) -> None:
    fig.savefig(figs_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figs_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_r_histogram(corr_df: pd.DataFrame, figs_dir: Path) -> None:
    r_vals = corr_df["pearson_r"].dropna().values
    med    = float(np.median(r_vals))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axvline(0,   color="gray", linewidth=0.8, linestyle="--", zorder=1)
    ax.axvline(med, color=palette.MODEL_FILL["enformer"], linewidth=1.2,
               linestyle="--", label=f"Median r = {med:.3f}", zorder=3)
    ax.hist(r_vals, bins=40, color=palette.MODEL_FILL["pastl"],
            edgecolor="white", linewidth=0.3, zorder=2)
    ax.set_xlabel("Pearson r  (patient-specific prediction vs observed protein)")
    ax.set_ylabel("Genes")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _save(fig, figs_dir, "patient_r_histogram")


def plot_r_vs_protein_level(corr_df: pd.DataFrame, figs_dir: Path) -> None:
    """Scatter: per-gene patient r vs mean observed protein level."""
    sub = corr_df.dropna(subset=["pearson_r", "mean_obs_protein"])
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=1)
    ax.scatter(
        sub["mean_obs_protein"], sub["pearson_r"],
        color=palette.MODEL_FILL["pastl"],
        s=14, alpha=0.55, linewidths=0, zorder=2,
    )
    ax.set_xlabel("Mean observed protein (log₂ TMT ratio)")
    ax.set_ylabel("Pearson r  (patient prediction vs observed)")
    fig.tight_layout()
    _save(fig, figs_dir, "patient_r_vs_protein_level")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True,
                        help="YAML config (e.g. model/configs/config.yaml).")
    parser.add_argument("--wandb_run_id", required=True)
    parser.add_argument("--checkpoint_name", default="best.pt")
    parser.add_argument("--vcf", required=True,
                        help="Cohort germline VCF (.vcf.gz, tabix-indexed).")
    parser.add_argument("--gvcf_dir", required=True,
                        help="Directory of per-patient gVCF files ({case_id}.g.vcf.gz). "
                             "Used to map CPT aliquot IDs in the cohort VCF to C3L case IDs.")
    parser.add_argument("--gct", nargs="+", required=True,
                        help="CPTAC proteomics GCT files (discovery and confirmatory).")
    parser.add_argument("--sequences_bed", default=_DEFAULT_SEQUENCES_BED)
    parser.add_argument("--checkpoints_dir", default=_CHECKPOINTS_DIR,
                        help="Base directory containing model checkpoints")
    parser.add_argument("--predictions_dir", default=_PREDICTIONS_DIR,
                        help="Root predictions directory (not used for output, only "
                             "for locating reference predictions.pkl for the summary).")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Patients per GPU forward pass (default: 8).")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False,
                        help="bf16 autocast for the forward pass (default: fp32).")
    parser.add_argument("--use_gene_mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attention_head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gene_index", type=int, default=None,
                        help="0-based start index for array-job slicing.")
    parser.add_argument("--n_genes", type=int, default=None,
                        help="Genes to process per array task (required with --gene_index).")
    parser.add_argument("--limit_genes", type=int, default=None,
                        help="Score only the first N genes (debugging).")
    parser.add_argument("--postprocess_only", action="store_true",
                        help="Skip model loading and prediction; run correlation and "
                             "figures over already-complete predictions/ directory.")
    args = parser.parse_args()

    if args.gene_index is not None and args.n_genes is None:
        parser.error("--n_genes is required when --gene_index is set")

    # --- Config ---
    cfg = PASTLConfig()
    with open(args.config) as f:
        data = yaml.safe_load(f) or {}
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.use_gene_mask   = args.use_gene_mask
    cfg.attention_head  = args.attention_head

    palette.use_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    slug = f"{args.wandb_run_id}_{Path(args.checkpoint_name).stem}"
    precision = ("bf16 autocast" if (args.amp and torch.cuda.is_available()) else "fp32")
    print(f"Slug: {slug} | device: {device} | precision: {precision} | "
          f"batch_size: {args.batch_size}\n")

    # --- Output dirs ---
    out_dir  = Path(args.output_dir)
    pred_dir = out_dir / "predictions"
    for d in (pred_dir, out_dir / "tables", out_dir / "figures", out_dir / "logs"):
        d.mkdir(parents=True, exist_ok=True)

    # --- Proteomics ---
    prot_df = load_cptac_gct_normals(args.gct)

    # --- VCF samples ---
    vcf_samples = get_vcf_samples(args.vcf)
    print(f"VCF: {len(vcf_samples):,} samples in {args.vcf}")

    # --- CPT → case ID mapping (from per-patient gVCF headers) ---
    print(f"Building CPT→case mapping from gVCFs in {args.gvcf_dir} ...")
    cpt_to_case = build_sample_map(args.gvcf_dir)
    print(f"  {len(cpt_to_case):,} gVCF entries read\n")

    # --- Patient intersection ---
    prot_to_vcf   = match_patients(vcf_samples, list(prot_df.columns), cpt_to_case)
    shared_prot   = sorted(prot_to_vcf.keys())
    shared_vcf    = [prot_to_vcf[p] for p in shared_prot]
    n_shared      = len(shared_prot)
    if n_shared == 0:
        raise RuntimeError("No patients shared between VCF and proteomics — check ID formats.")

    prot_df = prot_df[shared_prot]

    (out_dir / "tables" / "patient_ids.txt").write_text("\n".join(shared_prot) + "\n")

    # --- Gene universe (full list always, no slicing for postprocess) ---
    bed = pd.read_csv(args.sequences_bed, sep="\t")
    if args.limit_genes:
        bed = bed.head(args.limit_genes)

    if not args.postprocess_only and args.gene_index is not None:
        lo = args.gene_index
        hi = min(lo + args.n_genes, len(bed))
        bed = bed.iloc[lo:hi].reset_index(drop=True)
        print(f"Array slice: genes[{lo}:{hi}]  ({len(bed):,} genes this task)")

    genes = bed["symbol"].tolist()
    print(f"Gene universe: {len(genes):,} genes\n")

    (out_dir / "tables" / "gene_order.txt").write_text("\n".join(genes) + "\n")

    if not args.postprocess_only:
        # --- CPTACDataset for utilities (no FASTA/cache needed here) ---
        cpt = CPTACDataset(sequences_bed=args.sequences_bed, fasta_path=None, cache_dir=None)

        # --- FASTA ---
        fa = Fasta(cfg.fasta)

        # --- Model ---
        checkpoint = Path(args.checkpoints_dir) / args.wandb_run_id / args.checkpoint_name
        model      = load_model(cfg, checkpoint, device)

        # --- Per-gene prediction loop ---
        n_done = sum(1 for g in genes if (pred_dir / f"{g}.npy").exists())
        print(f"Prediction loop: {n_done:,} / {len(genes):,} genes already complete\n")

        for gene_row in tqdm(bed.itertuples(), total=len(bed), desc="Genes"):
            symbol    = gene_row.symbol
            pred_file = pred_dir / f"{symbol}.npy"
            if pred_file.exists():
                continue

            chrom     = str(gene_row.chrom)
            seq_start = int(gene_row.start)
            seq_end   = int(gene_row.end)
            strand    = str(gene_row.strand)

            ref_seq = fetch_ref_window(fa, chrom, seq_start, seq_end)

            per_sample_vars = fetch_snvs_in_window(
                args.vcf, chrom, seq_start, seq_end, shared_vcf
            )

            encodings = [
                build_patient_encoding(ref_seq, per_sample_vars.get(vcf_id, []), cpt, strand)
                for vcf_id in shared_vcf
            ]

            gene_mask = cpt._create_gene_body_mask(symbol, seq_start, strand)
            preds     = run_patient_predictions(
                model, encodings, gene_mask, device, args.batch_size, args.amp
            )
            np.save(pred_file, preds.astype(np.float32))

        print("Prediction loop complete.\n")

    # --- Correlation ---
    print("Computing per-gene correlations...")
    corr_df = compute_correlations(pred_dir, prot_df, shared_prot, genes)
    corr_df.to_csv(out_dir / "tables" / "per_gene_correlations.csv", index=False)

    valid_r = corr_df["pearson_r"].dropna()
    n_pos   = int((valid_r > 0).sum())
    print(f"  Genes with valid r:  {len(valid_r):,} / {len(corr_df):,}")
    print(f"  Median r:            {valid_r.median():.4f}")
    print(f"  Mean r:              {valid_r.mean():.4f}")
    print(f"  r > 0:               {n_pos:,} ({100 * n_pos / len(valid_r):.1f}%)\n")

    # --- Figures ---
    plot_r_histogram(corr_df, out_dir / "figures")
    plot_r_vs_protein_level(corr_df, out_dir / "figures")

    # --- Summary log ---
    with open(out_dir / "logs" / "summary.txt", "w") as fh:
        fh.write("=== Patient Correlation Analysis ===\n")
        fh.write(f"Model slug:          {slug}\n")
        fh.write(f"VCF:                 {args.vcf}\n")
        fh.write(f"GCT files:           {args.gct}\n")
        fh.write(f"Patients (shared):   {n_shared:,}\n")
        fh.write(f"Genes:               {len(genes):,}\n")
        fh.write(f"Precision:           {precision}\n\n")
        fh.write("Per-gene patient Pearson r (Solid Tissue Normal vs patient prediction):\n")
        fh.write(f"  Valid genes:       {len(valid_r):,}\n")
        fh.write(f"  Median r:          {valid_r.median():.4f}\n")
        fh.write(f"  Mean r:            {valid_r.mean():.4f}\n")
        fh.write(f"  r > 0:             {n_pos:,} ({100 * n_pos / len(valid_r):.1f}%)\n")

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
