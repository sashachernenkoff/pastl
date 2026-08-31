r"""
Weight-correlation analysis for PASTL models with the attention head.

Extracts per-gene spatial attention weights from a trained checkpoint,
annotates each gene's Enformer output window with genomic features
(5UTR, CDS, 3UTR, PLS, ELS, CTCF), computes Cohen's d effect sizes to
characterize which features the model attends to, and produces a global
violin plot and per-gene example track plots.

Intermediate annotations are cached in <output_dir>/tables/ and reused on
subsequent runs.  The model must have been trained with --attention_head.

Usage:
    python weight_correlation_analysis.py \
        --wandb_run_id m7h1162n --checkpoint best.pt \
        --config /path/to/config.yaml \
        --gtf /path/to/gencode.v36.annotation.gtf.gz \
        --encode /path/to/ENCFF252OTD.bed.gz \
        --output_dir out/analysis/m7h1162n_best/weight_corr/
"""

import argparse
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "model"))

from config import load_config
from dataset import CPTACDataset, build_collate_fn
from model import PASTLFinetune
import palette
from utils import ensure_dir

_CHECKPOINTS_DIR = "out/model/checkpoints"

NUM_BINS      = 896
BIN_SIZE      = 128
OUTPUT_RADIUS = (NUM_BINS * BIN_SIZE) // 2   # 57,344 bp
SEQ_HALF      = 196_608 // 2                  # 98,304 bp

FEATURE_MAP = {
    'CDS':  ('CDSStarts',  'CDSEnds'),
    'PLS':  ('PLS_Starts', 'PLS_Ends'),
    'ELS':  ('ELS_Starts', 'ELS_Ends'),
    '5UTR': ('5UTRStarts', '5UTREnds'),
    '3UTR': ('3UTRStarts', '3UTREnds'),
    'CTCF': ('CTCF_Starts', 'CTCF_Ends'),
}

_BOLD  = palette.PALETTES["bold"]
_MUTED = palette.PALETTES["muted"]

# Violin bodies — muted palette, order matches violin x-axis: CDS PLS ELS 5UTR 3UTR CTCF
VIOLIN_COLORS = {
    'CDS':  _MUTED["cat1"],
    'PLS':  _MUTED["cat2"],
    'ELS':  _MUTED["cat3"],
    '5UTR': _MUTED["cat4"],
    '3UTR': _MUTED["cat5"],
    'CTCF': _MUTED["other"],
}

# Feature track patches — muted palette, per-feature semantic colour assignment
TRACK_COLORS = {
    'CTCF': _MUTED["cat1"],   # blue
    'ELS':  _MUTED["cat3"],   # green
    'PLS':  _MUTED["cat5"],   # gold
    '3UTR': _MUTED["cat4"],   # purple
    'CDS':  _MUTED["cat2"],   # red
    '5UTR': _MUTED["other"],  # grey
}

WINDOW_BG      = "#EBEBEB"   # light grey shaded output window
ATTN_FILL      = _BOLD["cat1"]  # bold blue for attention weight trace

FEATURE_TRACKS = [
    ('CTCF', 'CTCF_Starts', 'CTCF_Ends', 1),
    ('ELS',  'ELS_Starts',  'ELS_Ends',  2),
    ('PLS',  'PLS_Starts',  'PLS_Ends',  3),
    ('3UTR', '3UTRStarts',  '3UTREnds',  4),
    ('CDS',  'CDSStarts',   'CDSEnds',   5),
    ('5UTR', '5UTRStarts',  '5UTREnds',  6),
]


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def _fmt_intervals(intervals):
    if not intervals:
        return pd.NA, pd.NA
    intervals = sorted(intervals)
    return ",".join(str(x[0]) for x in intervals), ",".join(str(x[1]) for x in intervals)


def _fmt_intervals_df(df_subset):
    if df_subset.empty:
        return pd.NA, pd.NA
    return ",".join(df_subset['start'].astype(str)), ",".join(df_subset['end'].astype(str))


def build_annotation(bed_df: pd.DataFrame, gtf_path: str, encode_path: str) -> pd.DataFrame:
    """
    Annotate genes with UTR/CDS intervals (from GTF) and cCRE intervals
    (from ENCODE), matched to each gene's Enformer sequence window.
    Uses txid from the bed to select the canonical transcript.
    """
    target_tx_map = dict(zip(bed_df['Symbol'], bed_df['TxID']))

    # --- GTF: separate 5UTR / CDS / 3UTR ---
    print(f"Parsing GTF: {gtf_path} ...")
    transcript_features: dict = defaultdict(lambda: {'UTR': [], 'CDS': []})
    with gzip.open(gtf_path, 'rt') as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            fields = line.split('\t')
            feature = fields[2]
            if feature not in ('UTR', 'CDS'):
                continue
            attr = {item.strip().split(' ')[0]: item.strip().split(' ')[1].strip('"')
                    for item in fields[8].strip().split(';')
                    if len(item.strip().split(' ')) == 2}
            symbol = attr.get('gene_name')
            tx_id  = attr.get('transcript_id', '').split('.')[0]
            if symbol not in target_tx_map or tx_id != target_tx_map[symbol]:
                continue
            transcript_features[symbol][feature].append((int(fields[3]), int(fields[4])))

    feat_cols: dict[str, list] = {k: [] for k in
                                   ('5UTRStarts', '5UTREnds', 'CDSStarts', 'CDSEnds',
                                    '3UTRStarts', '3UTREnds')}
    for _, row in bed_df.iterrows():
        symbol = row['Symbol']
        strand = row['Strand']
        f_dict = transcript_features.get(symbol, {'UTR': [], 'CDS': []})
        cds_ivals = f_dict['CDS']
        utr_ivals = f_dict['UTR']
        all_5utr: set = set()
        all_3utr: set = set()
        if cds_ivals and utr_ivals:
            cds_min = min(x[0] for x in cds_ivals)
            cds_max = max(x[1] for x in cds_ivals)
            for u_s, u_e in utr_ivals:
                if strand == '+':
                    (all_5utr if u_e <= cds_min else all_3utr).add((u_s, u_e))
                else:
                    (all_5utr if u_s >= cds_max else all_3utr).add((u_s, u_e))
        s5, e5 = _fmt_intervals(all_5utr)
        sc, ec = _fmt_intervals(set(cds_ivals))
        s3, e3 = _fmt_intervals(all_3utr)
        feat_cols['5UTRStarts'].append(s5);  feat_cols['5UTREnds'].append(e5)
        feat_cols['CDSStarts'].append(sc);   feat_cols['CDSEnds'].append(ec)
        feat_cols['3UTRStarts'].append(s3);  feat_cols['3UTREnds'].append(e3)

    for col, data in feat_cols.items():
        bed_df[col] = data

    # --- ENCODE: PLS / ELS / CTCF ---
    print(f"Parsing ENCODE cCREs: {encode_path} ...")
    encode_df = pd.read_csv(encode_path, sep='\t', header=None,
                             usecols=[0, 1, 2, 9],
                             names=['chrom', 'start', 'end', 'cCRE_group'])
    pls_df  = encode_df[encode_df['cCRE_group'] == 'PLS']
    els_df  = encode_df[encode_df['cCRE_group'].isin(['pELS', 'dELS'])]
    ctcf_df = encode_df[encode_df['cCRE_group'].str.contains('CTCF', na=False)]

    enc_cols: dict[str, list] = {k: [] for k in
                                  ('PLS_Starts', 'PLS_Ends', 'ELS_Starts', 'ELS_Ends',
                                   'CTCF_Starts', 'CTCF_Ends')}
    for _, row in tqdm(bed_df.iterrows(), total=len(bed_df), desc="Mapping cCREs"):
        chrom  = row['Chrom']
        tss    = row['TSS']
        strand = row['Strand']
        seq_start = tss - (SEQ_HALF - 1) if strand == '-' else tss - SEQ_HALF
        seq_end   = seq_start + 196_608

        def _map(src: pd.DataFrame):
            sub = src[(src['chrom'] == chrom) &
                      (src['start'] >= seq_start) &
                      (src['end']   <= seq_end)]
            return _fmt_intervals_df(sub)

        sp, ep = _map(pls_df);   enc_cols['PLS_Starts'].append(sp);  enc_cols['PLS_Ends'].append(ep)
        se, ee = _map(els_df);   enc_cols['ELS_Starts'].append(se);  enc_cols['ELS_Ends'].append(ee)
        sc2, ec2 = _map(ctcf_df); enc_cols['CTCF_Starts'].append(sc2); enc_cols['CTCF_Ends'].append(ec2)

    for col, data in enc_cols.items():
        bed_df[col] = data

    return bed_df


# ---------------------------------------------------------------------------
# Spatial-weight extraction
# ---------------------------------------------------------------------------

def extract_spatial_weights(cfg, checkpoint_path: str, ds: CPTACDataset,
                             batch_size: int, splits: list[str]) -> dict[str, np.ndarray]:
    """
    Capture the per-bin attention weights from model.softmax via a forward hook.
    Returns {symbol: np.ndarray of shape (NUM_BINS,)}.
    Requires the model to have been trained with attention_head=True.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {checkpoint_path} on {device}")

    model = PASTLFinetune(cfg).to(device)
    ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if not cfg.attention_head:
        raise ValueError(
            "This analysis requires a model trained with --attention_head. "
            "The loaded checkpoint uses the simple linear head (no spatial weights)."
        )

    captured_weights: list[np.ndarray] = []
    captured_prots:   list[str]        = []

    base_model = model._orig_mod if hasattr(model, '_orig_mod') else model

    def _hook(module, input, output):
        # output: [B, 896, 1] -> extend list with [B, 896] arrays
        captured_weights.extend(output.detach().cpu().numpy().squeeze(-1))

    handle = base_model.softmax.register_forward_hook(_hook)

    idx = [i for i, s in enumerate(ds.instances) if ds.split_of[s] in splits]
    print(f"Extracting weights for {len(idx)} genes (splits: {splits}) ...")
    loader = DataLoader(
        Subset(ds, idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, cfg.max_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=build_collate_fn(ds, use_gene_mask=cfg.use_gene_mask, mode="test"),
    )

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting weights"):
            dna = batch['dna'].to(device)
            captured_prots.extend(batch['prots'])
            model(dna, gene_mask=None)  # no mask: capture unmasked spatial weights

    handle.remove()
    return dict(zip(captured_prots, captured_weights))


# ---------------------------------------------------------------------------
# Effect-size computation
# ---------------------------------------------------------------------------

def _get_binary_mask(row, start_col: str, end_col: str) -> np.ndarray:
    """896-length binary mask for a genomic feature, oriented 5'->3'."""
    mask = np.zeros(NUM_BINS)
    if pd.isna(row[start_col]) or pd.isna(row[end_col]):
        return mask
    tss          = row['TSS']
    window_start = tss - OUTPUT_RADIUS
    for s, e in zip([int(x) for x in str(row[start_col]).split(',')],
                    [int(x) for x in str(row[end_col]).split(',')]):
        rel_s = s - window_start
        rel_e = e - window_start
        bin_s = max(0, int(np.floor(rel_s / BIN_SIZE)))
        bin_e = min(NUM_BINS - 1, int(np.ceil(rel_e / BIN_SIZE)))
        if bin_s <= bin_e:
            mask[bin_s:bin_e + 1] = 1
    if row['Strand'] == '-':
        mask = mask[::-1]
    return mask


def _cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled = ((n1 - 1) * np.var(g1, ddof=1) + (n2 - 1) * np.var(g2, ddof=1)) / (n1 + n2 - 2)
    return 0.0 if pooled == 0 else (np.mean(g1) - np.mean(g2)) / np.sqrt(pooled)


def _empirical_p(weights: np.ndarray, mask: np.ndarray,
                 observed_d: float, num_permutations: int) -> float:
    shifts = np.random.randint(1, len(mask), size=num_permutations)
    count  = sum(
        1 for shift in shifts
        if (lambda m: abs(_cohens_d(weights[m == 1], weights[m == 0])) >= abs(observed_d))(np.roll(mask, shift))
    )
    return (count + 1) / (num_permutations + 1)


def compute_effect_sizes(ann_df: pd.DataFrame, weights_map: dict[str, np.ndarray],
                         num_permutations: int) -> pd.DataFrame:
    results = []
    valid   = set(weights_map)
    for _, row in tqdm(ann_df.iterrows(), total=len(ann_df), desc="Computing effect sizes"):
        symbol = row['Symbol']
        if symbol not in valid or pd.isna(row['CDSStarts']):
            continue
        weights = weights_map[symbol].astype(float)
        if np.all(weights == 0):
            continue
        entry = {'Symbol': symbol}
        for feat_label, (s_col, e_col) in FEATURE_MAP.items():
            mask = _get_binary_mask(row, s_col, e_col)
            tgt  = weights[mask == 1]
            bg   = weights[mask == 0]
            if len(tgt) > 1 and len(bg) > 1:
                d = _cohens_d(tgt, bg)
                entry[f'EffectSize_{feat_label}'] = d
                entry[f'PValue_{feat_label}'] = (
                    _empirical_p(weights, mask, d, num_permutations)
                    if not np.isnan(d) else np.nan
                )
            else:
                entry[f'EffectSize_{feat_label}'] = np.nan
                entry[f'PValue_{feat_label}']     = np.nan
        results.append(entry)
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_violin(results_df: pd.DataFrame, output_path: str):
    plot_df = (results_df
               .melt(id_vars='Symbol', var_name='Feature', value_name='EffectSize')
               .pipe(lambda d: d[d['Feature'].str.startswith('EffectSize_')].copy()))
    plot_df['Feature'] = plot_df['Feature'].str.replace('EffectSize_', '', regex=False)
    plot_df = plot_df[(plot_df['EffectSize'] >= -3) & (plot_df['EffectSize'] <= 5)]

    palette.use_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(
        data=plot_df, x='Feature', y='EffectSize',
        palette=VIOLIN_COLORS,
        order=['CDS', 'PLS', 'ELS', '5UTR', '3UTR', 'CTCF'],
        inner='box', width=0.6, ax=ax, zorder=2,
    )
    from matplotlib.collections import PolyCollection
    for collection in ax.collections:
        if isinstance(collection, PolyCollection):
            collection.set_edgecolor("black")
            collection.set_linewidth(0.4)
    ax.axhline(0, color="#888888", linestyle='--', linewidth=0.4, zorder=1)
    ax.set_title("Global Distribution of Feature-Weight Effect Sizes")
    ax.set_ylabel("Cohen's d")
    ax.set_xlabel("Genomic Feature Category")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved violin plot -> {output_path}")


def _auto_select_targets(ann_df: pd.DataFrame, weights_map: dict[str, np.ndarray],
                         n: int = 20) -> list[str]:
    """Top-N symbols by CDS enrichment among genes with a clear attention peak."""
    results = []
    for _, row in ann_df.iterrows():
        symbol = row['Symbol']
        if symbol not in weights_map or pd.isna(row['CDSStarts']):
            continue
        w = weights_map[symbol].astype(float)
        if np.sum(w) == 0:
            continue
        mask      = _get_binary_mask(row, 'CDSStarts', 'CDSEnds')
        max_w     = np.max(w)
        snr       = max_w / (np.mean(w) + 1e-9)
        cds_enr   = np.sum(w * mask) / np.sum(w)
        results.append({'Symbol': symbol, 'MaxW': max_w, 'SNR': snr, 'CDS_Enrich': cds_enr})
    ranked = (pd.DataFrame(results)
              .query("MaxW >= 0.012 and SNR >= 10.0")
              .sort_values('CDS_Enrich', ascending=False))
    top = ranked['Symbol'].head(n).tolist()
    print(f"Auto-selected {len(top)} example symbols by CDS enrichment.")
    return top


def _parse_coords(starts_str, ends_str) -> list[tuple[int, int]]:
    if pd.isna(starts_str) or pd.isna(ends_str):
        return []
    return list(zip([int(x) for x in str(starts_str).split(',')],
                    [int(x) for x in str(ends_str).split(',')]))


def plot_tracks(symbols: list[str], ann_df: pd.DataFrame,
                weights_map: dict[str, np.ndarray], output_dir: str):
    ann_indexed   = ann_df.set_index('Symbol')
    window_radius = SEQ_HALF  # display full Enformer input window

    for symbol in symbols:
        if symbol not in ann_indexed.index or symbol not in weights_map:
            print(f"Skipping {symbol} — data missing.")
            continue

        row     = ann_indexed.loc[symbol]
        weights = weights_map[symbol].astype(float)
        tss     = row['TSS']
        chrom   = row['Chrom']
        strand  = row['Strand']

        # Reverse weights for minus-strand so display is 5'->3' (left to right)
        if strand == '-':
            weights = weights[::-1]

        window_start = tss - window_radius
        window_end   = tss + window_radius

        palette.use_style()
        fig, (ax_w, ax_f) = plt.subplots(
            nrows=2, ncols=1, figsize=(10, 2.4),
            gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.1},
        )

        bin_coords = np.linspace(
            tss - OUTPUT_RADIUS + BIN_SIZE // 2,
            tss + OUTPUT_RADIUS - BIN_SIZE // 2,
            NUM_BINS,
        )
        ax_w.plot(bin_coords, weights, color="black", linewidth=0.5, zorder=3)
        ax_w.axvspan(tss - OUTPUT_RADIUS, tss + OUTPUT_RADIUS, color=WINDOW_BG, zorder=1)
        ax_w.set_title(f"{symbol} Spatial Attention Weights", fontweight='bold', pad=10)
        ax_w.set_ylabel("Attention")

        for label, s_col, e_col, y_pos in FEATURE_TRACKS:
            for s, e in _parse_coords(row[s_col], row[e_col]):
                if e < window_start or s > window_end:
                    continue
                actual_w  = e - s
                min_w     = BIN_SIZE * 4
                width     = max(actual_w, min_w)
                x0        = s if actual_w >= min_w else (s + e) / 2 - min_w / 2
                ax_f.add_patch(patches.Rectangle(
                    (x0, y_pos - 0.4), width, 0.8,
                    facecolor=TRACK_COLORS[label], zorder=3, linewidth=0,
                ))
        ax_f.axvspan(tss - OUTPUT_RADIUS, tss + OUTPUT_RADIUS, color=WINDOW_BG, zorder=1)

        for ax in (ax_w, ax_f):
            xlim = (window_end, window_start) if strand == '-' else (window_start, window_end)
            ax.set_xlim(*xlim)

        peak = round(float(weights.max()), 2)
        ax_w.set_yticks([0, peak])
        ax_w.set_ylim(bottom=0)
        ax_w.set_xticks([])
        ax_f.set_ylim(0.5, 6.5)
        ax_f.set_yticks([t[3] for t in FEATURE_TRACKS])
        ax_f.set_yticklabels([t[0] for t in FEATURE_TRACKS])
        ax_f.set_xticks([])
        pos_label = f"{chrom}:{window_start:,}-{window_end:,}"
        ax_f.set_xlabel(pos_label)
        ax_f.set_ylabel("Features")

        plt.tight_layout()
        out_path = Path(output_dir) / f"{symbol}_corr_map.png"
        plt.savefig(str(out_path), bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_run_id",      required=True,
                        help="WandB run ID; locates the checkpoint directory")
    parser.add_argument("--checkpoint",        default="best.pt",
                        help="Checkpoint filename within the run directory (default: best.pt)")
    parser.add_argument("--config",            required=True,
                        help="Path to model YAML config")
    parser.add_argument("--gtf",               required=True,
                        help="GENCODE GTF file (gzipped) for CDS/UTR annotation")
    parser.add_argument("--encode",            required=True,
                        help="ENCODE cCRE BED file (gzipped) for PLS/ELS/CTCF annotation")
    parser.add_argument("--sequences_bed",     default=None,
                        help="Override the sequences_protein.bed path from the config")
    parser.add_argument("--output_dir",        required=True,
                        help="Root output directory; tables/ and figures/ created inside")
    parser.add_argument("--batch_size",        type=int, default=8)
    parser.add_argument("--splits",            nargs='+',
                        default=["train", "valid", "test"],
                        help="Dataset splits to extract weights from (default: all)")
    parser.add_argument("--target_symbols",    nargs='*', default=None,
                        help="Symbols for example track plots; auto-selected if omitted")
    parser.add_argument("--num_example_plots", type=int, default=20,
                        help="Number of auto-selected example plots (default: 20)")
    parser.add_argument("--num_permutations",  type=int, default=1000,
                        help="Permutations for empirical p-value (default: 1000)")
    parser.add_argument("--checkpoints_dir",   default=_CHECKPOINTS_DIR,
                        help="Base directory containing model checkpoints")
    args = parser.parse_args()

    palette.use_style()
    cfg = load_config(args.config)
    if args.sequences_bed:
        cfg.sequences_bed = args.sequences_bed

    checkpoint_path = str(Path(args.checkpoints_dir) / args.wandb_run_id / args.checkpoint)

    tables_dir  = Path(args.output_dir) / "tables"
    figures_dir = Path(args.output_dir) / "figures"
    ensure_dir(str(tables_dir))
    ensure_dir(str(figures_dir))

    # ── Gene metadata from sequences_protein.bed ──────────────────────────
    print(f"Loading gene metadata from {cfg.sequences_bed} ...")
    bed_raw = pd.read_csv(cfg.sequences_bed, sep='\t')
    bed_df  = pd.DataFrame({
        'Symbol': bed_raw['symbol'],
        'Chrom':  bed_raw['chrom'],
        'Strand': bed_raw['strand'],
        'TxID':   bed_raw['txid'].astype(str).str.split('.').str[0],  # strip version suffix
        'TSS':    ((bed_raw['start'] + bed_raw['end']) // 2).astype(int),
    })

    # ── Annotation (cached) ────────────────────────────────────────────────
    ann_path = tables_dir / "weight_corr_annotation.tsv"
    if ann_path.exists():
        print(f"Loading cached annotation from {ann_path}")
        ann_df = pd.read_csv(ann_path, sep='\t')
    else:
        ann_df = build_annotation(bed_df.copy(), args.gtf, args.encode)
        ann_df.to_csv(str(ann_path), sep='\t', index=False)
        print(f"Saved annotation -> {ann_path}")

    # ── Extract spatial weights from checkpoint (cached) ──────────────────
    weights_path = tables_dir / "spatial_weights.pkl"
    if weights_path.exists():
        print(f"Loading cached spatial weights from {weights_path}")
        weights_map = pd.read_pickle(weights_path)
    else:
        ds = CPTACDataset(
            sequences_bed=cfg.sequences_bed,
            fasta_path=cfg.fasta,
            cache_dir=cfg.cache_dir,
        )
        weights_map = extract_spatial_weights(cfg, checkpoint_path, ds,
                                              args.batch_size, args.splits)
        pd.to_pickle(weights_map, weights_path)
        print(f"Saved spatial weights -> {weights_path}")

    # ── Compute Cohen's d effect sizes (cached) ───────────────────────────
    results_path = tables_dir / "weight_feature_effect_sizes.tsv"
    if results_path.exists():
        print(f"Loading cached effect sizes from {results_path}")
        results_df = pd.read_csv(results_path, sep='\t')
    else:
        results_df = compute_effect_sizes(ann_df, weights_map, args.num_permutations)
        results_df.to_csv(str(results_path), sep='\t', index=False)
        print(f"Saved effect sizes -> {results_path}")

    print("\n% of genes with significant effect sizes (p < 0.05):")
    for feat in FEATURE_MAP:
        p_col = f'PValue_{feat}'
        sig   = (results_df[p_col] < 0.05).sum()
        total = results_df[p_col].notna().sum()
        print(f"  {feat}: {sig}/{total} ({sig / max(total, 1):.1%})")

    print("\nMedian Cohen's d per feature:")
    print(results_df[[f'EffectSize_{f}' for f in FEATURE_MAP]].median().to_string())

    # ── Violin plot ────────────────────────────────────────────────────────
    plot_violin(results_df, str(figures_dir / "global_effect_size.png"))

    # ── Example track plots ────────────────────────────────────────────────
    target_symbols = (args.target_symbols
                      or _auto_select_targets(ann_df, weights_map, args.num_example_plots))
    print(f"\nGenerating track plots for: {target_symbols}")
    plot_tracks(target_symbols, ann_df, weights_map, str(figures_dir))


if __name__ == "__main__":
    main()
