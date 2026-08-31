r"""
Per-gene ISM inference for PASTL and Enformer.

Reads one gene from priority_genes.tsv (identified by --gene_idx, which equals
$SLURM_ARRAY_TASK_ID in the array job), generates every single-nucleotide
substitution within the gene's targeted regulatory zones, and computes:

    S = |y_ref - y_mut|

for both PASTL (scalar z-scored protein prediction) and base Enformer
(log2(1 + sum of CAGE:lung counts over 4 TSS bins, track 4687, bins 446-449)).

Region types scored by default: tss, utr5, utr3.
Pass --include_gene_body to also score the full transcribed region.

Minus-strand genes: mutations are applied in + strand genomic coordinates for
Enformer (always + strand). For PASTL, the mutation is mirrored to the
reverse-complement sequence: position L-1-rel_pos with the complement allele.

Output (one file per gene):
  {output_dir}/{gene_symbol}.h5
  Datasets: position (int32), ref_base, alt_base, region_type (str),
            pastl_ism (float32), enformer_ism (float32)
  Attributes: symbol, chrom, strand, tss, seq_start, seq_end,
              ref_pastl, ref_enformer

Usage:
  python run_ism.py --config model/configs/config.yaml \
      --wandb_run_id s5fo3aak --checkpoint_name best.pt --attention_head \
      --gene_idx $SLURM_ARRAY_TASK_ID \
      --priority_genes out/analysis/ism/priority_genes.tsv \
      --regions_tsv    out/analysis/ism/ism_regions.tsv \
      --output_dir     out/analysis/ism/scores
"""
import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import wandb
from pyfaidx import Fasta
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "model"))

from config import PASTLConfig
from dataset import CPTACDataset
from model import PASTLFinetune
from utils import ensure_dir

_PREDICTIONS_DIR = "out/predictions"
_CHECKPOINTS_DIR = "out/model/checkpoints"
_ISM_DIR = "out/analysis/ism"

WINDOW_SIZE = 196_608
HALF_WINDOW = WINDOW_SIZE // 2

# Base-Enformer CAGE scoring (matches predict_variants.py)
CAGE_LUNG_IDX = 4687
TSS_BIN_START = 446
TSS_BIN_END = 450

BASES = ["A", "C", "G", "T"]
BASE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}
COMPLEMENT_IDX = [3, 2, 1, 0]   # A(0)↔T(3), C(1)↔G(2)

ACTIVE_REGION_TYPES = {"tss", "utr5", "utr3"}


# ---------------------------------------------------------------------------
# Model loaders (mirrors predict_variants.py)
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


# ---------------------------------------------------------------------------
# Sequence helpers
# ---------------------------------------------------------------------------

def fetch_window(fasta_path, chrom, seq_start, seq_end):
    """Return the + strand reference string for [seq_start, seq_end), N-padded."""
    fa = Fasta(fasta_path)
    chrom_key = chrom if chrom in fa else f"chr{chrom}"
    fetch_start = max(0, seq_start)
    seq = str(fa[chrom_key][fetch_start:seq_end].seq.upper())
    left_pad = max(0, -seq_start)
    right_pad = WINDOW_SIZE - (left_pad + len(seq))
    return ("N" * left_pad) + seq + ("N" * max(0, right_pad))


def one_hot(seq):
    """Encode a DNA string to [L, 4] float32. N maps to an all-zero row."""
    tokens = torch.tensor([BASE_MAP.get(b, 4) for b in seq], dtype=torch.long)
    return torch.nn.functional.one_hot(tokens, num_classes=5)[:, :4].float()


def rc_one_hot(dna_oh):
    """Reverse-complement a [L, 4] one-hot tensor: reverse then complement columns."""
    return dna_oh.flip(0)[:, [3, 2, 1, 0]]


# ---------------------------------------------------------------------------
# Reference scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_reference(pastl_model, enformer_model, dna_plus, dna_pastl,
                    gene_mask, device, amp):
    """Score the reference sequence with both models; returns (y_pastl, y_enformer)."""
    dna_plus_b = dna_plus.unsqueeze(0).to(device)
    dna_pastl_b = dna_pastl.unsqueeze(0).to(device)
    gmask_b = gene_mask.unsqueeze(0).to(device) if gene_mask is not None else None

    amp_ctx = torch.autocast(
        device_type="cuda", dtype=torch.bfloat16,
        enabled=(amp and torch.cuda.is_available())
    )
    with amp_ctx:
        y_pastl = pastl_model(dna_pastl_b, gene_mask=gmask_b).squeeze(-1).float().item()
        cage_raw = enformer_model(dna_plus_b)["human"][
            0, TSS_BIN_START:TSS_BIN_END, CAGE_LUNG_IDX
        ].float()
        y_enformer = torch.log2(1.0 + cage_raw).sum().item()

    return y_pastl, y_enformer


# ---------------------------------------------------------------------------
# HDF5 streaming writer
# ---------------------------------------------------------------------------

class ISMWriter:
    """Appends ISM mutation records to a resizable HDF5 file."""

    _STR_DT = h5py.string_dtype()

    def __init__(self, path, gene_meta):
        self._f = h5py.File(path, "w")
        for k, v in gene_meta.items():
            self._f.attrs[k] = str(v)
        kw = dict(shape=(0,), maxshape=(None,), chunks=True)
        self._pos   = self._f.create_dataset("position",      dtype="int32",        **kw)
        self._ref   = self._f.create_dataset("ref_base",      dtype=self._STR_DT,   **kw)
        self._alt   = self._f.create_dataset("alt_base",      dtype=self._STR_DT,   **kw)
        self._rtype = self._f.create_dataset("region_type",   dtype=self._STR_DT,   **kw)
        self._tlp   = self._f.create_dataset("pastl_ism",    dtype="float32",      **kw)
        self._enf   = self._f.create_dataset("enformer_ism",  dtype="float32",      **kw)
        self._n = 0

    def append(self, positions, ref_bases, alt_bases, region_types,
               pastl_scores, enformer_scores):
        k = len(positions)
        if k == 0:
            return
        new_n = self._n + k
        for ds in (self._pos, self._ref, self._alt, self._rtype, self._tlp, self._enf):
            ds.resize(new_n, axis=0)
        self._pos[self._n:new_n]   = np.array(positions,      dtype=np.int32)
        self._ref[self._n:new_n]   = ref_bases
        self._alt[self._n:new_n]   = alt_bases
        self._rtype[self._n:new_n] = region_types
        self._tlp[self._n:new_n]   = np.array(pastl_scores,  dtype=np.float32)
        self._enf[self._n:new_n]   = np.array(enformer_scores, dtype=np.float32)
        self._n = new_n

    def close(self):
        self._f.close()


# ---------------------------------------------------------------------------
# ISM inference loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def _flush_batch(buf, dna_plus, dna_pastl_ref, gene_mask, strand,
                 pastl_model, enformer_model, ref_pastl, ref_enformer,
                 device, amp, scores_by_region, writer):
    """
    Forward both models on a batch of mutations; append results to writer.
    buf: list of (gpos, rel_pos, ref_idx, alt_idx, region_type)
    """
    B = len(buf)

    # Build + strand batch (Enformer always uses + strand).
    dna_plus_batch = dna_plus.unsqueeze(0).expand(B, -1, -1).clone()
    for k, (gpos, rel_pos, ref_idx, alt_idx, rtype) in enumerate(buf):
        dna_plus_batch[k, rel_pos, :] = 0.0
        dna_plus_batch[k, rel_pos, alt_idx] = 1.0

    # Build PASTL batch: for minus-strand genes apply the mutation at the
    # reverse-complemented position with the complemented allele.
    if strand == "-":
        dna_pastl_batch = dna_pastl_ref.unsqueeze(0).expand(B, -1, -1).clone()
        for k, (gpos, rel_pos, ref_idx, alt_idx, rtype) in enumerate(buf):
            rc_pos = WINDOW_SIZE - 1 - rel_pos
            rc_alt = COMPLEMENT_IDX[alt_idx]
            dna_pastl_batch[k, rc_pos, :] = 0.0
            dna_pastl_batch[k, rc_pos, rc_alt] = 1.0
    else:
        dna_pastl_batch = dna_plus_batch

    dna_plus_b   = dna_plus_batch.to(device, non_blocking=True)
    dna_pastl_b = dna_pastl_batch.to(device, non_blocking=True)
    gmask_b = (
        gene_mask.unsqueeze(0).expand(B, -1).to(device, non_blocking=True)
        if gene_mask is not None else None
    )

    amp_ctx = torch.autocast(
        device_type="cuda", dtype=torch.bfloat16,
        enabled=(amp and torch.cuda.is_available())
    )
    with amp_ctx:
        pastl_mut = pastl_model(dna_pastl_b, gene_mask=gmask_b).squeeze(-1).float()
        cage_mut = enformer_model(dna_plus_b)["human"][
            :, TSS_BIN_START:TSS_BIN_END, CAGE_LUNG_IDX
        ].float()
        enf_mut = torch.log2(1.0 + cage_mut).sum(dim=1)

    tlp_scores = torch.abs(pastl_mut - ref_pastl).cpu().numpy()
    enf_scores = torch.abs(enf_mut - ref_enformer).cpu().numpy()

    pos_out, ref_out, alt_out, rtype_out, tlp_out, enf_out = [], [], [], [], [], []
    for k, (gpos, rel_pos, ref_idx, alt_idx, rtype) in enumerate(buf):
        pos_out.append(gpos)
        ref_out.append(BASES[ref_idx])
        alt_out.append(BASES[alt_idx])
        rtype_out.append(rtype)
        tlp_out.append(float(tlp_scores[k]))
        enf_out.append(float(enf_scores[k]))
        scores_by_region[rtype]["pastl"].append(float(tlp_scores[k]))
        scores_by_region[rtype]["enformer"].append(float(enf_scores[k]))

    writer.append(pos_out, ref_out, alt_out, rtype_out, tlp_out, enf_out)


def run_gene_ism(symbol, gene_rows, fasta_path, pastl_model, enformer_model,
                 cpt, device, amp, batch_size, output_dir,
                 include_gene_body, wandb_run, limit_positions=None):
    active_types = ACTIVE_REGION_TYPES | ({"gene_body"} if include_gene_body else set())
    regions_df = gene_rows[gene_rows["region_type"].isin(active_types)].copy()

    if regions_df.empty:
        print(f"  [{symbol}] no regions to mutate — skipping")
        return

    chrom = str(regions_df["chrom"].iloc[0])
    strand = str(regions_df["strand"].iloc[0])
    tss = int(regions_df["tss"].iloc[0])
    seq_start = tss - HALF_WINDOW
    seq_end = tss + HALF_WINDOW

    out_path = Path(output_dir) / f"{symbol}.h5"
    if out_path.exists():
        print(f"  [{symbol}] output already exists — skipping")
        return

    # Reference sequence (+ strand) and one-hot encodings
    seq_ref = fetch_window(fasta_path, chrom, seq_start, seq_end)[:WINDOW_SIZE]
    dna_plus = one_hot(seq_ref)
    dna_pastl_ref = rc_one_hot(dna_plus) if strand == "-" else dna_plus

    gene_mask = cpt._create_gene_body_mask(symbol, seq_start, strand)

    ref_pastl, ref_enformer = score_reference(
        pastl_model, enformer_model, dna_plus, dna_pastl_ref, gene_mask, device, amp
    )
    print(f"  [{symbol}] ref: pastl={ref_pastl:.4f}  enformer={ref_enformer:.4f}")

    # Build position lists per region_type (relative window index + genomic pos)
    positions_by_region = {}
    for _, row in regions_df.iterrows():
        rtype = str(row["region_type"])
        r_start, r_end = int(row["start"]), int(row["end"])
        rel_gpos = [
            (r - seq_start, r)
            for r in range(r_start, r_end)
            if 0 <= r - seq_start < WINDOW_SIZE
        ]
        positions_by_region.setdefault(rtype, []).extend(rel_gpos)

    if limit_positions is not None:
        positions_by_region = {
            rt: pos[:limit_positions] for rt, pos in positions_by_region.items()
        }

    total_pos = sum(len(v) for v in positions_by_region.values())
    print(f"  [{symbol}] {total_pos:,} positions across "
          f"{list(positions_by_region.keys())} -> ~{total_pos * 3:,} mutations")

    scores_by_region = {rt: {"pastl": [], "enformer": []} for rt in positions_by_region}
    writer = ISMWriter(out_path, {
        "symbol":       symbol,
        "chrom":        chrom,
        "strand":       strand,
        "tss":          tss,
        "seq_start":    seq_start,
        "seq_end":      seq_end,
        "ref_pastl":   ref_pastl,
        "ref_enformer": ref_enformer,
    })

    buf = []
    n_done = 0
    t0 = time.time()
    pbar = tqdm(desc=symbol, unit="mut")

    for rtype, rel_gpos_list in positions_by_region.items():
        for rel_pos, gpos in rel_gpos_list:
            ref_vec = dna_plus[rel_pos]
            ref_idx = int(ref_vec.argmax().item())
            if ref_vec.max().item() < 0.5:
                continue   # N base — skip
            for alt_idx in range(4):
                if alt_idx == ref_idx:
                    continue
                buf.append((gpos, rel_pos, ref_idx, alt_idx, rtype))
                if len(buf) == batch_size:
                    _flush_batch(
                        buf, dna_plus, dna_pastl_ref, gene_mask, strand,
                        pastl_model, enformer_model, ref_pastl, ref_enformer,
                        device, amp, scores_by_region, writer,
                    )
                    n_done += batch_size
                    pbar.update(batch_size)
                    buf = []

    if buf:   # final partial batch
        _flush_batch(
            buf, dna_plus, dna_pastl_ref, gene_mask, strand,
            pastl_model, enformer_model, ref_pastl, ref_enformer,
            device, amp, scores_by_region, writer,
        )
        n_done += len(buf)
        pbar.update(len(buf))

    pbar.close()
    writer.close()

    elapsed = time.time() - t0
    throughput = n_done / elapsed if elapsed > 0 else 0.0
    print(f"  [{symbol}] {n_done:,} mutations in {elapsed / 60:.1f} min "
          f"({throughput:.0f} mut/s) -> {out_path}")

    # W&B logging: per-region score distributions and throughput
    log_dict = {
        "gene":                   symbol,
        "n_mutations":            n_done,
        "throughput_mut_per_s":   throughput,
        "elapsed_min":            elapsed / 60,
        "ref_pastl":             ref_pastl,
        "ref_enformer":           ref_enformer,
    }
    for rtype, sc in scores_by_region.items():
        if sc["pastl"]:
            log_dict[f"{rtype}/mean_pastl_ism"]   = float(np.mean(sc["pastl"]))
            log_dict[f"{rtype}/max_pastl_ism"]    = float(np.max(sc["pastl"]))
            log_dict[f"{rtype}/mean_enformer_ism"] = float(np.mean(sc["enformer"]))
            log_dict[f"{rtype}/max_enformer_ism"]  = float(np.max(sc["enformer"]))
            log_dict[f"{rtype}/pastl_hist"]       = wandb.Histogram(sc["pastl"])
            log_dict[f"{rtype}/enformer_hist"]     = wandb.Histogram(sc["enformer"])
    wandb_run.log(log_dict)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True,
                        help="Path to config.yaml config")
    parser.add_argument("--wandb_run_id", required=True,
                        help="PASTL W&B run ID (e.g. s5fo3aak)")
    parser.add_argument("--checkpoint_name", default="best.pt",
                        help="Checkpoint filename within the run directory (default: best.pt)")
    parser.add_argument("--attention_head",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_gene_mask",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gene_idx", type=int, required=True,
                        help="Row index in priority_genes.tsv (= $SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--priority_genes", required=True,
                        help="Path to priority_genes.tsv")
    parser.add_argument("--regions_tsv", required=True,
                        help="Path to ism_regions.tsv (from build_ism_regions.py)")
    parser.add_argument("--output_dir", default=f"{_ISM_DIR}/scores",
                        help="Directory to write per-gene .h5 files")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Mutations per forward pass (default: 4)")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False,
                        help="bf16 autocast (default: off = full fp32)")
    parser.add_argument("--include_gene_body",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Also mutate positions in the gene body region (default: off)")
    parser.add_argument("--limit_positions", type=int, default=None,
                        help="Cap positions per region (e.g. 2 for a login-node smoke test)")
    parser.add_argument("--checkpoints_dir", default=_CHECKPOINTS_DIR,
                        help="Base directory containing model checkpoints")
    args = parser.parse_args()

    import yaml
    cfg = PASTLConfig()
    with open(args.config) as f:
        data = yaml.safe_load(f) or {}
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.use_gene_mask = args.use_gene_mask
    cfg.attention_head = args.attention_head

    # --- Identify this task's gene ---
    priority = pd.read_csv(args.priority_genes, sep="\t")
    if args.gene_idx >= len(priority):
        raise ValueError(
            f"--gene_idx {args.gene_idx} is out of range (n_genes={len(priority)})"
        )
    symbol = priority.iloc[args.gene_idx]["symbol"]
    group = priority.iloc[args.gene_idx]["group"]
    print(f"Gene {args.gene_idx}: {symbol} ({group})")

    regions = pd.read_csv(args.regions_tsv, sep="\t")
    gene_rows = regions[regions["symbol"] == symbol]
    if gene_rows.empty:
        print(f"[{symbol}] not found in ism_regions.tsv — nothing to do")
        sys.exit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True   # fixed 196,608 bp input -> autotune convs

    checkpoint = Path(args.checkpoints_dir) / args.wandb_run_id / args.checkpoint_name
    pastl_model = load_pastl(cfg, checkpoint, device)
    enformer_model = load_enformer(device)

    # CPTACDataset for the gene-body mask helper (no FASTA/cache needed)
    cpt = CPTACDataset(sequences_bed=cfg.sequences_bed, fasta_path=None, cache_dir=None)

    ensure_dir(args.output_dir)

    wandb_run = wandb.init(
        project="pastl-ism",
        name=f"ism_{symbol}",
        group="ism_array",
        config={
            "wandb_run_id":    args.wandb_run_id,
            "checkpoint":      args.checkpoint_name,
            "gene_idx":        args.gene_idx,
            "symbol":          symbol,
            "group":           group,
            "batch_size":      args.batch_size,
            "amp":             args.amp,
            "include_gene_body": args.include_gene_body,
        },
    )

    run_gene_ism(
        symbol=symbol,
        gene_rows=gene_rows,
        fasta_path=cfg.fasta,
        pastl_model=pastl_model,
        enformer_model=enformer_model,
        cpt=cpt,
        device=device,
        amp=(args.amp and torch.cuda.is_available()),
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        include_gene_body=args.include_gene_body,
        limit_positions=args.limit_positions,
        wandb_run=wandb_run,
    )

    wandb_run.finish()


if __name__ == "__main__":
    main()
