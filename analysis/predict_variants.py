r"""
Variant-effect prediction for the GWAS / QTL analyses.

Runs WT vs MUT forward passes for every variant in one or more manifest TSVs
(built by analysis/gwas/build_variants.py and analysis/qtl/build_variants.py)
and writes per-variant effect scores that the GWAS/QTL plot scripts consume.

Two modes:
  --mode pastl     Fine-tuned PASTL model. Score = DELTA_Z = MUT - WT predicted
                   z-scored protein abundance (the model's scalar output).
                   Minus-strand windows are reverse-complemented to match
                   training (see model/dataset.py).
  --mode enformer  Frozen base Enformer. Score = LOG2FC, the summed-log fold
                   change of the CAGE:lung track over a 4-bin TSS window
                   (Avsec et al. 2021; track 4687, bins 446:450). Always + strand,
                   matching Enformer's training convention.

Output:
  PASTL:    {predictions_dir}/{run}_{ckpt}/variants/{dataset}.tsv
  Enformer: {predictions_dir}/enformer/variants/{dataset}.tsv
Each row = all manifest columns + WT/MUT scores + DELTA_Z | LOG2FC.

The variant->gene window mapping, TSS, strand and exon coordinates come from
data/sequences_protein.bed via the manifest and model.dataset.CPTACDataset.

Usage:
  python predict_variants.py --config model/configs/config.yaml --mode pastl \
      --wandb_run_id s5fo3aak --checkpoint_name best.pt --attention_head \
      --manifest out/predictions/variants/gwas_manifest_annotated.tsv \
                 out/predictions/variants/eqtl_gtex_manifest.tsv

  python predict_variants.py --config model/configs/config.yaml --mode enformer \
      --manifest out/predictions/variants/gwas_manifest_annotated.tsv
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from pyfaidx import Fasta
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))

from config import PASTLConfig
from dataset import CPTACDataset
from model import PASTLFinetune
from utils import ensure_dir

_PREDICTIONS_DIR = "out/predictions"
_CHECKPOINTS_DIR = "out/model/checkpoints"

WINDOW_SIZE = 196_608
HALF_WINDOW = WINDOW_SIZE // 2

# Base-Enformer CAGE scoring.
CAGE_LUNG_IDX = 4687       # CAGE:lung, adult, pool1 (basenji targets_human.txt)
TSS_BIN_START = 446        # 4 central bins (~512 bp) around the TSS bin (448)
TSS_BIN_END   = 450


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
    """
    Builds a single 196,608 bp one-hot window per row, centred at the target
    gene's TSS. Reuses a CPTACDataset for the genome handle, one-hot / reverse-
    complement helpers, and the gene-body mask.

    apply_variant=False -> the reference (WT) window; identical for every variant
                           of a gene, so it is scored once per gene and cached.
    apply_variant=True  -> the ALT (MUT) window for a specific variant.

    mode='pastl'  : reverse-complements minus-strand windows and emits a
                      gene-body mask (matches training).
    mode='enformer' : always + strand, no mask.
    """

    def __init__(self, rows: pd.DataFrame, cpt: CPTACDataset, mode: str,
                 fasta_path: str, apply_variant: bool):
        self.rows = rows.reset_index(drop=True)
        self.cpt = cpt
        self.mode = mode
        self.fasta_path = fasta_path
        self.apply_variant = apply_variant
        self._fa = None   # pyfaidx handle, opened lazily per worker

    def __len__(self):
        return len(self.rows)

    def _get_fasta(self):
        """Worker-local pyfaidx handle. Opening per worker avoids the shared
        file-descriptor races that corrupt reads when one Fasta is forked."""
        if self._fa is None:
            self._fa = Fasta(self.fasta_path)
        return self._fa

    def _fetch_ref_window(self, chrom: str, seq_start: int, seq_end: int) -> str:
        """Reference + strand sequence for [seq_start, seq_end), N-padded at edges."""
        genome = self._get_fasta()
        chrom_key = chrom if chrom in genome else f"chr{chrom}"
        fetch_start = max(0, seq_start)
        ref_seq = str(genome[chrom_key][fetch_start:seq_end].seq.upper())
        left_pad  = max(0, -seq_start)
        right_pad = WINDOW_SIZE - (left_pad + len(ref_seq))
        return ("N" * left_pad) + ref_seq + ("N" * max(0, right_pad))

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]
        tss = int(row["TSS"])
        strand = str(row["STRAND"])
        chrom = str(row["CHROM"])

        seq_start = tss - HALF_WINDOW
        seq_end   = tss + HALF_WINDOW
        seq = self._fetch_ref_window(chrom, seq_start, seq_end)

        if self.apply_variant:
            # Apply the variant on the + strand genomic sequence.
            var_pos_0 = int(row["POS"]) - 1        # VCF POS is 1-based
            ref_allele = str(row["REF"])
            alt_allele = str(row["ALT"])
            rel_pos = var_pos_0 - seq_start
            if 0 <= rel_pos < len(seq) and rel_pos + len(ref_allele) <= len(seq):
                seq = seq[:rel_pos] + alt_allele + seq[rel_pos + len(ref_allele):]
                # Keep TSS centred on indels: trim / pad from the variant's side.
                len_diff = len(seq) - WINDOW_SIZE
                if len_diff > 0:
                    seq = seq[len_diff:] if rel_pos < HALF_WINDOW else seq[:WINDOW_SIZE]
                elif len_diff < 0:
                    pad = "N" * (-len_diff)
                    seq = (pad + seq) if rel_pos < HALF_WINDOW else (seq + pad)

        # Enforce exact length (pyfaidx can return ±1 bp at chromosome edges).
        seq = seq[:WINDOW_SIZE]

        # PASTL matches training by reverse-complementing minus-strand windows;
        # base Enformer always scores the + strand.
        if self.mode == "pastl" and strand == "-":
            seq = seq.translate(self.cpt.rev_comp_trans)[::-1]

        item = {"dna": self.cpt._one_hot(seq), "metadata": row.to_dict()}
        if self.mode == "pastl":
            item["gene_mask"] = self.cpt._create_gene_body_mask(
                str(row["TARGET_SYMBOL"]), seq_start, strand
            )
        return item


def build_collate_fn(mode: str):
    def collate_fn(batch):
        out = {
            "dna":      torch.stack([b["dna"] for b in batch]),
            "metadata": [b["metadata"] for b in batch],
        }
        if mode == "pastl":
            out["gene_mask"] = torch.stack([b["gene_mask"] for b in batch])
        return out
    return collate_fn


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _resume_point(out_tsv: Path) -> int:
    """Number of variant rows already written (for restart), or 0."""
    if not out_tsv.exists():
        return 0
    with open(out_tsv) as f:
        return max(0, sum(1 for _ in f) - 1)   # minus header


def _model_forward(model, dna, gmask, mode, amp_enabled):
    """One forward pass over a batch of windows.
    pastl -> [b] scalar predictions; enformer -> [b, 4] CAGE-lung TSS bins."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        if mode == "pastl":
            return model(dna, gene_mask=gmask).squeeze(-1)
        return model(dna)["human"][:, TSS_BIN_START:TSS_BIN_END, CAGE_LUNG_IDX]


def _loader(rows, cpt, mode, fasta_path, apply_variant, batch_size, num_workers):
    ds = WindowDataset(rows, cpt, mode, fasta_path, apply_variant=apply_variant)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), collate_fn=build_collate_fn(mode))


def score_dataset(label, variants, cpt, mode, model, device, batch_size,
                  num_workers, out_tsv, fasta_path, amp):
    out_tsv = Path(out_tsv)
    ensure_dir(str(out_tsv.parent))

    start_idx = _resume_point(out_tsv)
    write_mode, write_header = ("w", True)
    if start_idx > 0:
        if start_idx >= len(variants):
            print(f"  [{label}] already complete ({start_idx} rows) — skipping")
            return
        print(f"  [{label}] resuming from row {start_idx}/{len(variants)}")
        variants = variants.iloc[start_idx:]
        write_mode, write_header = ("a", False)

    amp_enabled = amp and torch.cuda.is_available()

    # --- Phase 1: score the WT (reference) window once per unique gene ---
    # WT is identical for every variant of a gene, so caching it here is what
    # cuts the forward passes ~in half (one WT/gene instead of one WT/variant).
    genes = variants.drop_duplicates("TARGET_GENE").reset_index(drop=True)
    print(f"  [{label}] {len(variants):,} variants across {len(genes):,} genes; "
          f"caching WT per gene")
    wt_map: dict = {}
    with torch.no_grad():
        for batch in tqdm(_loader(genes, cpt, mode, fasta_path, False, batch_size, num_workers),
                          desc=f"{label} WT"):
            dna = batch["dna"].to(device, non_blocking=True)
            gmask = batch["gene_mask"].to(device).float() if mode == "pastl" else None
            vals = _model_forward(model, dna, gmask, mode, amp_enabled).float().cpu()
            for k, meta in enumerate(batch["metadata"]):
                wt_map[meta["TARGET_GENE"]] = float(vals[k]) if mode == "pastl" else vals[k]

    # --- Phase 2: score the MUT window per variant, subtract the cached WT ---
    chunk, t0 = [], time.time()
    with torch.no_grad():
        mut_loader = _loader(variants, cpt, mode, fasta_path, True, batch_size, num_workers)
        for i, batch in enumerate(tqdm(mut_loader, desc=label)):
            dna = batch["dna"].to(device, non_blocking=True)
            metas = batch["metadata"]
            genes_b = [m["TARGET_GENE"] for m in metas]

            if mode == "pastl":
                gmask = batch["gene_mask"].to(device, non_blocking=True).float()
                mut_p = _model_forward(model, dna, gmask, mode, amp_enabled).float().cpu().numpy()
                for k, meta in enumerate(metas):
                    wt = wt_map[genes_b[k]]
                    meta["WT_PRED_Z"] = wt
                    meta["MUT_PRED_Z"] = mut_p[k]
                    meta["DELTA_Z"] = mut_p[k] - wt
                    chunk.append(meta)
            else:  # enformer
                mut_cage = _model_forward(model, dna, None, mode, amp_enabled).float()  # [b,4]
                wt_cage = torch.stack([wt_map[g] for g in genes_b]).to(mut_cage.device)  # [b,4]
                sar = (torch.log2(1 + mut_cage) - torch.log2(1 + wt_cage)).sum(dim=1).cpu().numpy()
                wt_sum = wt_cage.sum(dim=1).cpu().numpy()
                mut_sum = mut_cage.sum(dim=1).cpu().numpy()
                for k, meta in enumerate(metas):
                    meta["WT_CAGE"] = wt_sum[k]
                    meta["MUT_CAGE"] = mut_sum[k]
                    meta["LOG2FC"] = sar[k]
                    chunk.append(meta)

            if (i + 1) % 200 == 0 or (i + 1) == len(mut_loader):
                pd.DataFrame(chunk).to_csv(out_tsv, sep="\t", mode=write_mode,
                                           header=write_header, index=False)
                write_mode, write_header = ("a", False)
                chunk = []

    print(f"  [{label}] done in {(time.time() - t0) / 60:.1f} min -> {out_tsv}")


def merge_shards(manifests, output_dir, num_shards):
    """Concatenate per-shard outputs into the final {dataset}.tsv."""
    for m in manifests:
        name = dataset_name_from_manifest(Path(m))
        shard_dir = Path(output_dir) / "shards"
        files = sorted(shard_dir.glob(f"{name}.shard*of{num_shards:04d}.tsv"))
        if not files:
            print(f"  [{name}] no shards found in {shard_dir} — skipping merge")
            continue
        out = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)
        out_tsv = Path(output_dir) / f"{name}.tsv"
        out.to_csv(out_tsv, sep="\t", index=False)
        print(f"  [{name}] merged {len(files)} shards -> {out_tsv} ({len(out):,} rows)")


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_pastl(cfg, checkpoint, device):
    print(f"Loading PASTL checkpoint: {checkpoint}")
    model = PASTLFinetune(cfg).to(device)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_enformer(device):
    from enformer_pytorch import Enformer, from_pretrained
    if not hasattr(Enformer, "all_tied_weights_keys"):
        Enformer.all_tied_weights_keys = property(lambda self: {})
    print("Loading base Enformer (EleutherAI/enformer-official-rough)...")
    model = from_pretrained("EleutherAI/enformer-official-rough", use_tf_gamma=False)
    model.eval().to(device)
    return model


def dataset_name_from_manifest(path: Path) -> str:
    """gwas_manifest_annotated.tsv -> gwas ; eqtl_gtex_manifest.tsv -> eqtl_gtex."""
    stem = path.stem
    for suffix in ("_manifest_annotated", "_annotated", "_manifest"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["pastl", "enformer"], required=True)
    parser.add_argument("--wandb_run_id", default=None, help="Required for pastl mode.")
    parser.add_argument("--checkpoint_name", default="best.pt",
                        help="Checkpoint filename within the run dir (default: best.pt).")
    parser.add_argument("--manifest", nargs="+", required=True,
                        help="One or more manifest TSVs to score.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Windows per forward pass (WT and MUT phases both use "
                             "this). Default: 8.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False,
                        help="bf16 autocast for the forward pass (default: off = full fp32). "
                             "Pass --amp only when the ~2x speedup is worth bf16 rounding.")
    parser.add_argument("--use_gene_mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attention_head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--predictions_dir", default=_PREDICTIONS_DIR)
    parser.add_argument("--checkpoints_dir", default=_CHECKPOINTS_DIR,
                        help="Base directory containing model checkpoints")
    parser.add_argument("--output_dir", default=None,
                        help="Override the auto-constructed predictions/{slug}/variants path.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Score only the first N rows of each manifest (debugging).")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Split each manifest into this many strided shards "
                             "(for Slurm array scoring of large sets).")
    parser.add_argument("--shard", type=int, default=0,
                        help="Which shard index [0, num_shards) this task scores.")
    parser.add_argument("--merge", action="store_true",
                        help="Merge previously-written shards into {dataset}.tsv, then exit.")
    args = parser.parse_args()
    if args.num_shards > 1 and not (0 <= args.shard < args.num_shards):
        parser.error(f"--shard must be in [0, {args.num_shards})")

    # --- Config (mirror analysis/predict.py) ---
    import yaml
    cfg = PASTLConfig()
    with open(args.config) as f:
        data = yaml.safe_load(f) or {}
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.use_gene_mask = args.use_gene_mask
    cfg.attention_head = args.attention_head

    batch_size = args.batch_size or 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True   # fixed 196,608 bp input -> autotune convs
    precision = "bf16 autocast" if (args.amp and torch.cuda.is_available()) else "fp32"
    print(f"Precision: {precision} | batch={batch_size} "
          f"(WT cached once per gene, MUT per variant)")

    # --- Output dir (checkpoint-specific for PASTL, shared for Enformer) ---
    if args.mode == "pastl":
        assert args.wandb_run_id, "--wandb_run_id is required for pastl mode"
        slug = f"{args.wandb_run_id}_{Path(args.checkpoint_name).stem}"
        output_dir = args.output_dir or str(Path(args.predictions_dir) / slug / "variants")
        checkpoint = Path(args.checkpoints_dir) / args.wandb_run_id / args.checkpoint_name
    else:
        output_dir = args.output_dir or str(Path(args.predictions_dir) / "enformer" / "variants")

    # --- Merge mode: concatenate shards and exit (no model / GPU needed) ---
    if args.merge:
        merge_shards(args.manifest, output_dir, args.num_shards)
        return

    model = load_pastl(cfg, checkpoint, device) if args.mode == "pastl" else load_enformer(device)

    # --- Mask geometry from the bed (fasta opened per-worker in VariantDataset) ---
    cpt = CPTACDataset(sequences_bed=cfg.sequences_bed, fasta_path=None, cache_dir=None)

    for manifest_path in args.manifest:
        manifest_path = Path(manifest_path)
        name = dataset_name_from_manifest(manifest_path)
        variants = pd.read_csv(manifest_path, sep="\t")
        if args.limit:
            variants = variants.head(args.limit)

        if args.num_shards > 1:
            variants = variants.iloc[args.shard::args.num_shards]
            out_tsv = (Path(output_dir) / "shards" /
                       f"{name}.shard{args.shard:04d}of{args.num_shards:04d}.tsv")
            label = f"{name}[{args.shard}/{args.num_shards}]"
        else:
            out_tsv = Path(output_dir) / f"{name}.tsv"
            label = name

        print(f"\n=== Scoring '{label}': {len(variants):,} variants "
              f"({args.mode}) -> {out_tsv} ===")
        score_dataset(label, variants, cpt, args.mode, model, device, batch_size,
                      cfg.max_workers, out_tsv, cfg.fasta, args.amp)

    print("\nAll manifests scored.")


if __name__ == "__main__":
    main()
