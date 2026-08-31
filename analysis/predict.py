r"""
Generate protein-abundance predictions from a trained PASTL checkpoint or the
frozen Enformer backbone.

Modes:
  --mode pastl     Run a fine-tuned PASTL checkpoint. Scores the test split to
                   {predictions_dir}/{run_id}_{ckpt_stem}/predictions.pkl, or all
                   splits to predictions_all.pkl with --all_splits.
  --mode enformer  Run the frozen Enformer trunk as a feature extractor and write
                   pooled 3072-dim embeddings (central 3 bins and gene-mask pooled)
                   per split to {predictions_dir}/enformer/embeddings_{split}.pkl.

Usage:
  python analysis/predict.py --config model/configs/config.yaml --mode pastl \
      --wandb_run_id <run_id> --checkpoint_name best.pt \
      --checkpoints_dir out/model/checkpoints --predictions_dir out/predictions

  python analysis/predict.py --config model/configs/config.yaml --mode enformer \
      --predictions_dir out/predictions
"""
import argparse
import sys
import os
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))

from config import PASTLConfig
from dataset import CPTACDataset, build_collate_fn
from model import PASTLFinetune
from utils import ensure_dir

_PREDICTIONS_DIR = "out/predictions"
_CHECKPOINTS_DIR = "out/model/checkpoints"

def predict_loader(cfg, ds, indices, batch_size, mode):
    return DataLoader(
        Subset(ds, indices),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.max_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=build_collate_fn(ds, use_gene_mask=cfg.use_gene_mask, mode=mode),
    )

def extract_enformer(cfg, ds, batch_size, output_dir):
    """
    Extract Enformer 3072-dim embeddings (central bin only).
    """
    from enformer_pytorch import Enformer
    # Monkey-patch to fix compatibility with newer transformers versions (e.g. 5.14+)
    if not hasattr(Enformer, 'all_tied_weights_keys'):
        Enformer.all_tied_weights_keys = property(lambda self: {})
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Enformer base model on {device}...")
    
    backbone = Enformer.from_pretrained('EleutherAI/enformer-official-rough', use_tf_gamma=False)
    backbone.eval()
    backbone.to(device)

    for split in ["train", "valid", "test"]:
        idx = [i for i, s in enumerate(ds.instances) if ds.split_of[s] == split]
        if not idx:
            continue
            
        print(f"\nProcessing {split} split ({len(idx)} sequences)...")
        loader = predict_loader(cfg, ds, idx, batch_size, mode="test")

        all_ids = []
        all_labels = []
        all_embeddings_3bin = []
        all_embeddings_mask = []

        with torch.no_grad():
            for batch in tqdm(loader):
                dna = batch['dna'].to(device).float()
                gene_mask = batch['gene_mask'].to(device).float() if 'gene_mask' in batch else None
                label = batch['label'].numpy()

                # Pass through Enformer trunk in full fp32 precision
                _, embeddings = backbone(dna, return_embeddings=True)  # [B, 896, 3072]

                all_ids.extend(batch['prots'])
                all_labels.extend(label)
                
                # Pre-calculate BOTH pooling strategies to save 110GB of disk space
                
                # Strategy 1: Central 3 bins
                center_idx = embeddings.shape[1] // 2
                pool_3bin = embeddings[:, center_idx-1:center_idx+2, :].mean(dim=1).cpu().numpy()
                all_embeddings_3bin.extend(pool_3bin)
                
                # Strategy 2: Gene Mask (if available)
                if gene_mask is not None:
                    mask = gene_mask.unsqueeze(-1)
                    pool_mask = (embeddings * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
                    all_embeddings_mask.extend(pool_mask.cpu().numpy())
                else:
                    # If mask is disabled or missing, just pad with zeros
                    all_embeddings_mask.extend(np.zeros_like(pool_3bin))

        df = pd.DataFrame({
            "protein_id": all_ids,
            "true_label": all_labels,
            "embedding_3bin": all_embeddings_3bin,
            "embedding_genemask": all_embeddings_mask
        })
        
        out_file = Path(output_dir) / f"embeddings_{split}.pkl"
        ensure_dir(str(out_file.parent))
        df.to_pickle(out_file)
        print(f"Saved {split} embeddings to {out_file}")

def _run_model_on_indices(model, cfg, ds, indices, batch_size, device, split_label):
    loader = predict_loader(cfg, ds, indices, batch_size, mode="test")
    all_ids, all_labels, all_preds, all_splits = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=split_label):
            dna = batch['dna'].to(device)
            gene_mask = batch['gene_mask'].to(device).float() if 'gene_mask' in batch else None
            label = batch['label'].numpy()
            output = model(dna, gene_mask=gene_mask).squeeze(-1)
            all_ids.extend(batch['prots'])
            all_labels.extend(label)
            all_preds.extend(output.float().cpu().numpy())
            all_splits.extend([split_label] * len(batch['prots']))

    return all_ids, all_labels, all_preds, all_splits

def predict_pastl(cfg, checkpoint, ds, batch_size, output_dir):
    """
    Run the fine-tuned PASTL model on the test split.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading PASTL checkpoint: {checkpoint} on {device}")

    model = PASTLFinetune(cfg).to(device)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_idx = [i for i, s in enumerate(ds.instances) if ds.split_of[s] == 'test']
    print(f"\nProcessing TEST split ({len(test_idx)} sequences)...")
    all_ids, all_labels, all_preds, _ = _run_model_on_indices(
        model, cfg, ds, test_idx, batch_size, device, "test"
    )

    df = pd.DataFrame({
        "protein_id": all_ids,
        "true_label": all_labels,
        "prediction": all_preds
    })

    out_file = Path(output_dir) / "predictions.pkl"
    ensure_dir(str(out_file.parent))
    df.to_pickle(out_file)
    print(f"Saved test predictions to {out_file}")

    r2 = r2_score(all_labels, all_preds)
    pearson = pearsonr(all_labels, all_preds)[0]
    print(f"\nTest R²:       {r2:.4f}")
    print(f"Test Pearson r: {pearson:.4f}")

def predict_pastl_all(cfg, checkpoint, ds, batch_size, output_dir):
    """
    Run the fine-tuned PASTL model on all splits (train, valid, test)
    and save to predictions_all.pkl.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading PASTL checkpoint: {checkpoint} on {device}")

    model = PASTLFinetune(cfg).to(device)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_ids, all_labels, all_preds, all_splits = [], [], [], []

    for split in ["train", "valid", "test"]:
        idx = [i for i, s in enumerate(ds.instances) if ds.split_of[s] == split]
        if not idx:
            continue
        print(f"\nProcessing {split} split ({len(idx)} sequences)...")
        ids, labels, preds, splits = _run_model_on_indices(
            model, cfg, ds, idx, batch_size, device, split
        )
        all_ids.extend(ids)
        all_labels.extend(labels)
        all_preds.extend(preds)
        all_splits.extend(splits)

    df = pd.DataFrame({
        "protein_id": all_ids,
        "split": all_splits,
        "true_label": all_labels,
        "prediction": all_preds
    })

    out_file = Path(output_dir) / "predictions_all.pkl"
    ensure_dir(str(out_file.parent))
    df.to_pickle(out_file)
    print(f"\nSaved all-split predictions ({len(df):,} proteins) to {out_file}")

    # Report per-split metrics
    for split in ["train", "valid", "test"]:
        sub = df[df['split'] == split]
        if len(sub) == 0:
            continue
        r2 = r2_score(sub['true_label'], sub['prediction'])
        pearson = pearsonr(sub['true_label'], sub['prediction'])[0]
        print(f"{split:>5}  R²={r2:.4f}  Pearson r={pearson:.4f}  n={len(sub)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["pastl", "enformer"], required=True)
    parser.add_argument("--wandb_run_id", default=None, help="WandB run ID; required for pastl mode")
    parser.add_argument("--checkpoint_name", default="best.pt", help="Checkpoint filename within the run directory (default: best.pt)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--use_gene_mask", action=argparse.BooleanOptionalAction, default=True, help="Use gene mask for pooling (default: True)")
    parser.add_argument("--attention_head", action=argparse.BooleanOptionalAction, default=False, help="Use attention head (default: False)")
    parser.add_argument("--all_splits", action="store_true", default=False, help="Predict all splits (train+valid+test) and save to predictions_all.pkl")
    parser.add_argument("--predictions_dir", default=_PREDICTIONS_DIR, help="Base directory for prediction outputs")
    parser.add_argument("--checkpoints_dir", default=_CHECKPOINTS_DIR, help="Base directory containing model checkpoints")

    args = parser.parse_args()

    cfg = PASTLConfig() # Parse from the provided config yaml
    import yaml
    with open(args.config) as f:
        data = yaml.safe_load(f) or {}
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    
    cfg.batch_size = args.batch_size
    cfg.use_gene_mask = args.use_gene_mask
    cfg.attention_head = args.attention_head

    if args.mode == "pastl":
        assert args.wandb_run_id, "--wandb_run_id is required for pastl mode"
        checkpoint = Path(args.checkpoints_dir) / args.wandb_run_id / args.checkpoint_name
        slug = f"{args.wandb_run_id}_{Path(args.checkpoint_name).stem}"
        output_dir = Path(args.predictions_dir) / slug
    else:
        output_dir = Path(args.predictions_dir) / "enformer"

    print(f"Initializing Dataset...")
    ds = CPTACDataset(
        sequences_bed=cfg.sequences_bed,
        fasta_path=cfg.fasta,
        cache_dir=cfg.cache_dir,
    )

    if args.mode == "enformer":
        extract_enformer(cfg, ds, args.batch_size, output_dir)
    elif args.all_splits:
        predict_pastl_all(cfg, checkpoint, ds, args.batch_size, output_dir)
    else:
        predict_pastl(cfg, checkpoint, ds, args.batch_size, output_dir)

if __name__ == "__main__":
    main()
