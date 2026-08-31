"""
PyTorch Dataset for PASTL fine-tuning.

Loads protein windows and precomputed targets from sequences_protein.bed.
Sequences are served from an HDF5 cache built by precache_sequences.py,
with an optional FASTA fallback for cache misses.
"""

import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from pyfaidx import Fasta


WINDOW_SIZE = 196_608


class CPTACDataset(Dataset):
    def __init__(self, sequences_bed, fasta_path=None, cache_dir=None):
        print("\nInitializing dataset")

        self._seq_h5_path = Path(cache_dir) / "sequences.h5" if cache_dir else None
        self._seq_h5: h5py.File | None = None  # opened lazily per worker

        # Enformer geometry
        self.window_size = WINDOW_SIZE
        self.half_window = WINDOW_SIZE // 2

        # Gene-body mask geometry
        self.crop_bp  = 40_960
        self.bin_size = 128
        self.num_bins = 896

        # Load protein info from BED
        bed = pd.read_csv(sequences_bed, sep='\t')

        self.instances: list[str]       = list(bed['symbol'])
        self.split_of:  dict[str, str]  = dict(zip(bed['symbol'], bed['split']))
        self.meta_dict: dict[str, dict] = {}
        self.gene_boundaries: dict[str, tuple | None] = {}

        for _, row in bed.iterrows():
            sym = row['symbol']
            self.meta_dict[sym] = {
                'Chrom':  str(row['chrom']),
                'start':  int(row['start']),
                'end':    int(row['end']),
                'Strand': str(row['strand']),
                'target': float(row['target_value']),
            }
            try:
                starts = [int(x) for x in str(row['exonstarts']).split(',') if x]
                ends   = [int(x) for x in str(row['exonends']).split(',') if x]
                self.gene_boundaries[sym] = (min(starts), max(ends)) if starts and ends else None
            except (ValueError, AttributeError):
                self.gene_boundaries[sym] = None

        # FASTA handle for cache-miss fallback
        self.genome = Fasta(fasta_path) if fasta_path else None
        self.base_map       = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        self.rev_comp_trans = str.maketrans('ACGTN', 'TGCAN')

        print(f"Loaded {len(self.instances):,} proteins  "
              f"(train={sum(1 for s in self.split_of.values() if s=='train')}  "
              f"val={sum(1 for s in self.split_of.values() if s=='valid')}  "
              f"test={sum(1 for s in self.split_of.values() if s=='test')})")

    # ------------------------------------------------------------------
    # Sequence loading
    # ------------------------------------------------------------------

    def _get_h5(self) -> h5py.File | None:
        """Return a worker-local HDF5 handle, opening it on first access."""
        if self._seq_h5_path is None or not self._seq_h5_path.exists():
            return None
        if self._seq_h5 is None:
            self._seq_h5 = h5py.File(self._seq_h5_path, "r")
        return self._seq_h5

    def _load_cached_seq(self, prot_symbol: str):
        """Return one-hot tensor [WINDOW_SIZE, 4] float32 from HDF5 cache, or None."""
        try:
            f = self._get_h5()
            if f is not None and prot_symbol in f:
                return torch.from_numpy(f[prot_symbol][:].astype(np.float32))
        except Exception:
            pass
        return None

    def _fetch_from_fasta(self, prot_symbol: str, meta: dict) -> torch.Tensor:
        """Fetch sequence from FASTA and one-hot encode it."""
        if self.genome is None:
            raise RuntimeError(
                f"No sequence cache hit for {prot_symbol} and no FASTA path provided."
            )
        chrom  = meta['Chrom']
        start  = meta['start']
        end    = meta['end']
        strand = meta['Strand']

        chrom_key   = chrom if chrom in self.genome else f"chr{chrom}"
        fetch_start = max(0, start)
        ref_seq     = str(self.genome[chrom_key][fetch_start:end].seq.upper())

        left_pad_len  = max(0, -start)
        right_pad_len = WINDOW_SIZE - (left_pad_len + len(ref_seq))
        final_seq     = ('N' * left_pad_len) + ref_seq + ('N' * right_pad_len)

        if strand == '-':
            final_seq = final_seq.translate(self.rev_comp_trans)[::-1]

        return self._one_hot(final_seq)

    def _one_hot(self, seq: str) -> torch.Tensor:
        tokens = torch.tensor([self.base_map.get(base, 4) for base in seq], dtype=torch.long)
        return torch.nn.functional.one_hot(tokens, num_classes=5)[:, :4].float()

    # ------------------------------------------------------------------
    # Gene-body mask
    # ------------------------------------------------------------------

    def _create_gene_body_mask(self, prot_symbol: str, seq_start: int, strand: str) -> torch.Tensor:
        """Unmasks the transcribed region from first exonStart to last exonEnd."""
        mask       = torch.zeros(self.num_bins, dtype=torch.float32)
        boundaries = self.gene_boundaries.get(prot_symbol)

        if boundaries is not None:
            gene_start, gene_end = boundaries
            rel_s = gene_start - seq_start
            rel_e = gene_end   - seq_start
            bin_s = max(0,              int(np.floor((rel_s - self.crop_bp) / self.bin_size)))
            bin_e = min(self.num_bins - 1, int(np.floor((rel_e - self.crop_bp) / self.bin_size)))
            if bin_s <= bin_e and bin_e >= 0 and bin_s < self.num_bins:
                mask[bin_s: bin_e + 1] = 1.0
        else:
            mask += 1.0

        if strand == '-':
            mask = torch.flip(mask, dims=[0])

        return mask

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> dict:
        sym  = self.instances[idx]
        meta = self.meta_dict[sym]

        dna = self._load_cached_seq(sym)
        if dna is None:
            dna = self._fetch_from_fasta(sym, meta)

        return {
            'prot_symbol': sym,
            'dna':         dna,
            'base_start':  meta['start'],
            'strand':      meta['Strand'],
            'label':       torch.tensor(meta['target'], dtype=torch.float32),
        }


def build_collate_fn(dataset, use_gene_mask=False, mode='train'):
    """
    Returns a collate function for DataLoader.

    Args:
        dataset:       The CPTACDataset instance.
        use_gene_mask: If True, compute and include the gene-body mask in the batch.
        mode:          'train', 'val', or 'test' (reserved for future use).
    """
    def collate_fn(batch):
        dnas, labels, masks, prots = [], [], [], []

        for item in batch:
            dnas.append(item['dna'])
            labels.append(item['label'].item())
            prots.append(item['prot_symbol'])
            if use_gene_mask:
                masks.append(dataset._create_gene_body_mask(
                    item['prot_symbol'], item['base_start'], item['strand']
                ))

        batch_out = {
            'dna':   torch.stack(dnas),
            'label': torch.tensor(labels, dtype=torch.float32),
            'prots': prots,
        }
        if use_gene_mask:
            batch_out['gene_mask'] = torch.stack(masks)
        return batch_out

    return collate_fn
