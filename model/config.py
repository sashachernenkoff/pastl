"""
Centralized configuration for PASTL fine-tuning.

All hyperparameters, paths, and training settings live in PASTLConfig.
Load at runtime from a YAML file with optional CLI overrides:

    python train.py --config model/configs/config.yaml [--trunk_lr 1e-6 --epochs 50 ...]

Environment variables (optional; override the path defaults):
    PROJECT_DIR   = .
    DATA_DIR      = /path/to/data
    OUT_DIR       = out
    CACHE_DIR     = cache
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


# ---------------------------------------------------------------------------
# Path constants (env-var driven with sane defaults)
# ---------------------------------------------------------------------------

_PROJECT_DIR  = os.getenv("PROJECT_DIR",    ".")
_DATA_DIR     = os.getenv("DATA_DIR",       "/path/to/data")
_OUT_DIR      = os.getenv("OUT_DIR",        "out")
_CACHE_DIR    = os.getenv("CACHE_DIR",      "cache")


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class PASTLConfig:
    # --- Paths ---
    project_dir:  str = _PROJECT_DIR
    data_dir:     str = _DATA_DIR
    out_dir:      str = _OUT_DIR
    cache_dir:    str = _CACHE_DIR

    # Derived paths — set to empty string to use the defaults below
    gct_discovery:    str = ""
    gct_confirmatory: str = ""
    metadata:         str = ""
    fasta:            str = ""
    sequences_bed:    str = ""

    def __post_init__(self):
        if not self.gct_discovery:
            self.gct_discovery = os.path.join(self.data_dir, "CPTAC_protein/CPTAC_LUAD_discovery_proteome.gct")
        if not self.gct_confirmatory:
            self.gct_confirmatory = os.path.join(self.data_dir, "CPTAC_protein/CPTAC_LUAD_confirmatory_proteome.gct")
        if not self.fasta:
            self.fasta = os.path.join(self.data_dir, "GRCh38.d1.vd1.fa")
        if not self.sequences_bed:
            self.sequences_bed = os.path.join(self.project_dir, "data", "sequences_protein.bed")

    @property
    def gct_files(self) -> List[str]:
        return [self.gct_discovery, self.gct_confirmatory]

    # --- Model ---
    enformer_dim:       int   = 3072
    attention_head:     bool  = True
    bottleneck_dim:     int   = 256
    hidden_dim:         int   = 64
    bottleneck_dropout: float = 0.60
    head_dropout:       float = 0.15
    use_gene_mask:      bool  = True

    # --- Optimizer ---
    trunk_lr:     float = 1e-6
    head_lr:      float = 1.5e-5
    weight_decay: float = 1e-2

    # --- Schedule ---
    epochs:              int   = 50
    warmup_epochs:       int   = 5
    freeze_trunk_epochs: int   = 10
    min_lr:              float = 1e-8

    # --- Batching ---
    batch_size:         int = 4
    accumulation_steps: int = 16   # effective batch size = batch_size * accumulation_steps

    # --- Regularisation ---
    max_grad_norm: float = 0.2

    # --- Data ---
    window_size: int   = 196_608

    # --- System ---
    max_workers:       int  = 7
    compile_model:     bool = True
    use_checkpointing: bool = True

    # --- Logging ---
    wandb_project:  str           = "pastl-finetune"
    wandb_run_name: Optional[str] = None

    # --- Resume ---
    resume_from: Optional[str] = None


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _coerce(value, default):
    """Cast value to match the type of default (guards against YAML string surprises)."""
    if value is None or default is None or isinstance(default, (bool, list)):
        return value
    if isinstance(default, float):
        return float(value)
    if isinstance(default, int):
        return int(value)
    return value


def load_config(config_path: Optional[str] = None, **overrides) -> PASTLConfig:
    """
    Build a PASTLConfig, optionally loading a YAML file and applying CLI overrides.

    Priority (highest first): CLI overrides > YAML values > dataclass defaults.
    """
    cfg = PASTLConfig()
    defaults = {f.name: f.default for f in dataclasses.fields(cfg)}

    if config_path:
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, _coerce(v, defaults.get(k)))

    for k, v in overrides.items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, _coerce(v, defaults.get(k)))

    # Re-run __post_init__ to recompute derived paths after overrides
    cfg.__post_init__()
    return cfg


def parse_args() -> PASTLConfig:
    parser = argparse.ArgumentParser(description="PASTL fine-tuning")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    # Paths
    parser.add_argument("--project_dir",    type=str, default=None)
    parser.add_argument("--data_dir",       type=str, default=None)
    parser.add_argument("--out_dir",        type=str, default=None)
    parser.add_argument("--cache_dir",      type=str, default=None)
    parser.add_argument("--fasta",          type=str, default=None)
    parser.add_argument("--sequences_bed",  type=str, default=None)

    # Model
    parser.add_argument("--enformer_dim", type=int, default=None)
    parser.add_argument("--attention_head",     action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--bottleneck_dim",     type=int,   default=None)
    parser.add_argument("--hidden_dim",         type=int,   default=None)
    parser.add_argument("--bottleneck_dropout", type=float, default=None)
    parser.add_argument("--head_dropout",       type=float, default=None)
    parser.add_argument("--use_gene_mask", action=argparse.BooleanOptionalAction, default=None)

    # Optimizer / schedule
    parser.add_argument("--trunk_lr",     type=float, default=None)
    parser.add_argument("--head_lr",      type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--epochs",              type=int,   default=None)
    parser.add_argument("--warmup_epochs",       type=int,   default=None)
    parser.add_argument("--freeze_trunk_epochs", type=int,   default=None)
    parser.add_argument("--min_lr",              type=float, default=None)

    # Batching / regularization
    parser.add_argument("--batch_size",         type=int,   default=None)
    parser.add_argument("--accumulation_steps", type=int,   default=None)
    parser.add_argument("--max_grad_norm",      type=float, default=None)

    # Data
    parser.add_argument("--window_size",       type=int,   default=None)

    # System
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--compile_model", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable or disable torch.compile()")

    # Logging
    parser.add_argument("--wandb_project",  type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)

    # Resume
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint file to resume training from")

    args = parser.parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    return load_config(args.config, **overrides)
