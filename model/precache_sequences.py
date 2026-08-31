r"""
Pre-build and cache all protein sequences before training.

Reads sequences_protein.bed for the protein list and their exact genomic
windows, fetches each window from the reference FASTA, one-hot encodes it, and
writes the result to {cache_dir}/sequences.h5. Training then reads from this
cache instead of the FASTA on every batch. Cache keys are protein symbols;
values are [196608, 4] uint8 arrays. Already-cached proteins are skipped.

Usage:
    python precache_sequences.py --config model/configs/config.yaml

    python precache_sequences.py \
        --bed   /path/to/sequences_protein.bed \
        --fasta /path/to/hg38.fa \
        --cache /path/to/cache/
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py
import numpy as np
import pandas as pd
from pyfaidx import Fasta

sys.path.insert(0, str(Path(__file__).parent))

WINDOW_SIZE = 196_608

_BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
_RC_TRANS  = str.maketrans('ACGTN', 'TGCAN')


def _one_hot(seq: str) -> np.ndarray:
    """One-hot encode a DNA string → [L, 4] uint8. N/unknown → all-zeros row."""
    arr = np.zeros((len(seq), 4), dtype=np.uint8)
    for i, base in enumerate(seq):
        idx = _BASE_MAP.get(base)
        if idx is not None:
            arr[i, idx] = 1
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-cache PASTL protein sequences")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--bed",    type=str, default=None, help="sequences_protein.bed path")
    parser.add_argument("--fasta",  type=str, default=None, help="Reference FASTA path")
    parser.add_argument("--cache",  type=str, default=None, help="Cache directory")
    args = parser.parse_args()

    bed_path   = args.bed
    fasta_path = args.fasta
    cache_dir  = args.cache

    if args.config:
        from config import load_config  # type: ignore
        cfg        = load_config(args.config)
        bed_path   = bed_path   or cfg.sequences_bed
        fasta_path = fasta_path or cfg.fasta
        cache_dir  = cache_dir  or cfg.cache_dir

    for name, val in [("bed", bed_path), ("fasta", fasta_path), ("cache", cache_dir)]:
        if not val:
            parser.error(f"--{name} is required (or provide --config)")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    h5_path = cache_path / "sequences.h5"

    bed = pd.read_csv(bed_path, sep='\t')
    proteins = list(bed['symbol'])
    print(f"Proteins to cache: {len(proteins)}")

    already_cached: set[str] = set()
    if h5_path.exists():
        try:
            with h5py.File(h5_path, "r") as f:
                already_cached = set(f.keys())
        except Exception:
            pass
    print(f"Already cached:    {len(already_cached)}")

    genome = Fasta(fasta_path)
    bed_idx = bed.set_index('symbol')

    t0 = time.time()
    built   = 0
    skipped = 0

    with h5py.File(h5_path, "a") as f:
        for i, sym in enumerate(proteins):
            if sym in f:
                skipped += 1
                continue

            row    = bed_idx.loc[sym]
            chrom  = str(row['chrom'])
            start  = int(row['start'])
            end    = int(row['end'])
            strand = str(row['strand'])

            chrom_key = chrom if chrom in genome else f"chr{chrom}"

            fetch_start = max(0, start)
            ref_seq     = str(genome[chrom_key][fetch_start:end].seq.upper())

            left_pad_len  = max(0, -start)
            right_pad_len = WINDOW_SIZE - (left_pad_len + len(ref_seq))
            final_seq     = ('N' * left_pad_len) + ref_seq + ('N' * right_pad_len)

            if strand == '-':
                final_seq = final_seq.translate(_RC_TRANS)[::-1]

            arr = _one_hot(final_seq)
            f.create_dataset(sym, data=arr, compression="gzip", compression_opts=1)
            built += 1

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate    = built / max(elapsed, 1e-6)
                eta     = (len(proteins) - i - 1) / max(rate, 1e-6)
                print(
                    f"  [{i+1:,}/{len(proteins):,}]  "
                    f"built={built}  skipped={skipped}  "
                    f"rate={rate:.1f}/s  eta={eta/60:.1f}min",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Built:   {built}")
    print(f"  Skipped: {skipped} (already cached)")
    print(f"  Cache:   {h5_path}")


if __name__ == "__main__":
    main()
