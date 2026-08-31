"""
PASTL fine-tuning training script.

Usage:
    python train.py                                                     # defaults
    python train.py --config model/configs/config.yaml                 # YAML config
    python train.py --config model/configs/config.yaml --trunk_lr 5e-7 # + CLI override
    python train.py --resume_from /path/to/best.pt                     # resume
"""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

from config import PASTLConfig, parse_args
from dataset import CPTACDataset, build_collate_fn
from evaluate import compute_all_metrics, evaluate_test_set, plot_training_curves
from losses import build_criterion
from model import PASTLFinetune
from utils import count_parameters, ensure_dir


# ---------------------------------------------------------------------------
# Cooperative SIGTERM shutdown (Slurm pre-wall signal)
# ---------------------------------------------------------------------------

_SHOULD_STOP = False


def _handle_sigterm(signum, frame) -> None:
    global _SHOULD_STOP
    _SHOULD_STOP = True
    print("\n[SIGTERM received — will stop after this epoch]", flush=True)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def make_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs:  int,
    min_lr:        float,
    base_lr:       float,
) -> LambdaLR:
    """Linear warmup then cosine annealing, floored at min_lr / base_lr."""
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(min_lr / base_lr, cosine)

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path:        Path,
    model:       torch.nn.Module,
    optimizer:   torch.optim.Optimizer,
    scheduler:   LambdaLR,
    epoch:       int,
    best_val_loss: float,
    wandb_run_id:  str,
    train_losses:  list[float],
    val_losses:    list[float],
    val_r2_scores: list[float],
) -> None:
    torch.save(
        {
            "epoch":                epoch,
            "best_val_loss":        best_val_loss,
            "wandb_run_id":         wandb_run_id,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_losses":         train_losses,
            "val_losses":           val_losses,
            "val_r2_scores":        val_r2_scores,
        },
        path,
    )


def load_checkpoint(
    path:      str | Path,
    model:     torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    device:    torch.device,
) -> tuple[int, float, str, list, list, list]:
    """Loads checkpoint and returns (start_epoch, best_val_loss, wandb_run_id,
    train_losses, val_losses, val_r2_scores)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return (
        ckpt["epoch"] + 1,
        ckpt["best_val_loss"],
        ckpt.get("wandb_run_id", wandb.util.generate_id()),
        ckpt.get("train_losses", []),
        ckpt.get("val_losses", []),
        ckpt.get("val_r2_scores", []),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(cfg: PASTLConfig) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("Error: CUDA unavailable. Exiting.")
    device = torch.device("cuda")

    signal.signal(signal.SIGTERM, _handle_sigterm)

    # TF32 on Hopper
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True

    num_cores = min(cfg.max_workers, os.cpu_count() or cfg.max_workers)

    # --- Dataset ---
    full_dataset = CPTACDataset(
        sequences_bed=cfg.sequences_bed,
        fasta_path=cfg.fasta,
        cache_dir=cfg.cache_dir,
    )

    train_idx = [i for i, s in enumerate(full_dataset.instances) if full_dataset.split_of[s] == 'train']
    val_idx   = [i for i, s in enumerate(full_dataset.instances) if full_dataset.split_of[s] == 'valid']
    test_idx  = [i for i, s in enumerate(full_dataset.instances) if full_dataset.split_of[s] == 'test']
    print(f"\nData split:  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    # --- DataLoaders ---
    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=num_cores,
        pin_memory=True,
        prefetch_factor=2,
    )
    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        shuffle=True, drop_last=True,
        collate_fn=build_collate_fn(full_dataset, use_gene_mask=cfg.use_gene_mask, mode='train'),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_idx),
        shuffle=False,
        collate_fn=build_collate_fn(full_dataset, use_gene_mask=cfg.use_gene_mask, mode='val'),
        **loader_kwargs,
    )
    test_loader = DataLoader(
        Subset(full_dataset, test_idx),
        shuffle=False,
        collate_fn=build_collate_fn(full_dataset, use_gene_mask=cfg.use_gene_mask, mode='test'),
        **loader_kwargs,
    )

    # --- Model ---
    model = PASTLFinetune(cfg=cfg).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"\nModel parameters:")
    print(f"  Total:     {total_params:,}")
    print(f"  Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")

    if cfg.compile_model and hasattr(torch, 'compile'):
        model = torch.compile(model)

    # --- Optimizer (differential LRs) ---
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    
    if cfg.freeze_trunk_epochs == 0:
        raw_model.unfreeze_trunk()
        
    optimizer = torch.optim.AdamW(
        [
            {'params': raw_model.trunk_parameters(), 'lr': cfg.trunk_lr, 'name': 'trunk'},
            {'params': raw_model.head_parameters(),  'lr': cfg.head_lr,  'name': 'head'},
        ],
        weight_decay=cfg.weight_decay,
    )

    scheduler = make_cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=cfg.warmup_epochs,
        total_epochs=cfg.epochs,
        min_lr=cfg.min_lr,
        base_lr=cfg.trunk_lr,   # denominator for the multiplier; head LR scales proportionally
    )

    criterion   = build_criterion()

    # --- History ---
    train_losses:  list[float] = []
    val_losses:    list[float] = []
    val_r2_scores: list[float] = []
    best_val_loss  = float('inf')
    start_epoch    = 0
    run_id         = wandb.util.generate_id()

    # --- Resume ---
    if cfg.resume_from and os.path.exists(cfg.resume_from):
        print(f"\nResuming from: {cfg.resume_from}")
        start_epoch, best_val_loss, run_id, train_losses, val_losses, val_r2_scores = load_checkpoint(
            cfg.resume_from, model, optimizer, scheduler, device,
        )
        print(f"  Starting at epoch {start_epoch}, best val loss so far: {best_val_loss:.4f}")

    # --- W&B ---
    run = wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        id=run_id,
        resume="allow",
        save_code=True,
        config=vars(cfg),
    )
    run_id = run.id

    ckpt_dir = Path(cfg.out_dir) / "model" / "checkpoints" / run_id
    ensure_dir(str(ckpt_dir))

    latest_path = ckpt_dir / "latest.pt"
    best_path   = ckpt_dir / "best.pt"

    # Log initial metadata
    wandb.log({"n_trainable_params": trainable_params, "n_total_params": total_params}, step=0)

    # --- Training loop ---
    print("\nStarting fine-tuning...")
    for epoch in range(start_epoch, cfg.epochs):
        if epoch == cfg.freeze_trunk_epochs and cfg.freeze_trunk_epochs > 0:
            print(f"\n  [Unfreezing Enformer trunk at epoch {epoch}]")
            raw_model.unfreeze_trunk()
            
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            dna       = batch['dna'].to(device, non_blocking=True)
            gene_mask = batch['gene_mask'].to(device, non_blocking=True).float() if 'gene_mask' in batch else None
            label     = batch['label'].to(device, non_blocking=True).float()

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = model(dna, gene_mask=gene_mask).squeeze(-1)
                loss   = criterion(output, label.view(-1))

            (loss / cfg.accumulation_steps).backward()
            train_loss += loss.item()

            if (i + 1) % cfg.accumulation_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation ---
        model.eval()
        val_loss  = 0.0
        all_preds  = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                dna       = batch['dna'].to(device, non_blocking=True)
                gene_mask = batch['gene_mask'].to(device, non_blocking=True).float() if 'gene_mask' in batch else None
                label     = batch['label'].to(device, non_blocking=True).float()

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    output = model(dna, gene_mask=gene_mask).squeeze(-1)
                    loss   = criterion(output, label.view(-1))

                val_loss   += loss.item()
                all_preds.extend(output.float().cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        avg_val_loss    = val_loss / len(val_loader)
        val_metrics     = compute_all_metrics(all_labels, all_preds)
        current_val_r2  = val_metrics["r2"]

        val_losses.append(avg_val_loss)
        val_r2_scores.append(current_val_r2)

        wandb.log(
            {
                "epoch":        epoch + 1,
                "train/loss":   avg_train_loss,
                "val/loss":     avg_val_loss,
                "val/r2":       current_val_r2,
                "val/pearson":  val_metrics["pearson"],
                "val/spearman": val_metrics["spearman"],
                "lr/trunk":     optimizer.param_groups[0]['lr'],
                "lr/head":      optimizer.param_groups[1]['lr'],
            },
            step=epoch + 1,
        )

        print(
            f"  Epoch [{epoch + 1}/{cfg.epochs}]  "
            f"train={avg_train_loss:.4f}  val={avg_val_loss:.4f}  "
            f"R²={current_val_r2:.4f}  "
            f"lr_trunk={optimizer.param_groups[0]['lr']:.2e}"
        )

        scheduler.step()

        # Always overwrite latest
        save_checkpoint(
            latest_path, model=model, optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, best_val_loss=best_val_loss, wandb_run_id=run_id,
            train_losses=train_losses, val_losses=val_losses, val_r2_scores=val_r2_scores,
        )

        # Save an explicit checkpoint for this specific epoch
        epoch_path = ckpt_dir / f"epoch_{epoch + 1:03d}.pt"
        save_checkpoint(
            epoch_path, model=model, optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, best_val_loss=best_val_loss, wandb_run_id=run_id,
            train_losses=train_losses, val_losses=val_losses, val_r2_scores=val_r2_scores,
        )

        # Save best separately
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(
                best_path, model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, best_val_loss=best_val_loss, wandb_run_id=run_id,
                train_losses=train_losses, val_losses=val_losses, val_r2_scores=val_r2_scores,
            )
            wandb.save(str(best_path))
            print(f"    ✓ New best checkpoint saved ({avg_val_loss:.4f})")

        # Cooperative shutdown on SIGTERM
        if _SHOULD_STOP:
            print(f"[Stopping cleanly at epoch {epoch + 1}]", flush=True)
            break


    print(f"\n→ Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoints: {ckpt_dir}")

    # --- Post-training ---
    plot_training_curves(train_losses, val_losses, val_r2_scores, str(ckpt_dir))

    # Final test evaluation using the best checkpoint
    print("\nLoading best checkpoint for test evaluation...")
    load_checkpoint(best_path, model, optimizer, scheduler, device)
    test_metrics = evaluate_test_set(model, test_loader, device, str(ckpt_dir), cfg.bottleneck_dim)

    wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
    wandb.finish()


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
