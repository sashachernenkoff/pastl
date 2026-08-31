"""
PASTL: fine-tunes the Enformer trunk to predict a single protein-abundance
z-score from one-hot DNA.

The frozen Enformer trunk maps [B, 196608, 4] -> [B, 896, 3072] embeddings; its
upper transformer blocks and final pointwise layer are unfrozen in the second
training stage. Two interchangeable heads (cfg.attention_head) reduce the 896
bins to a scalar:

  attention_head=True (default): gene-body-masked spatial attention pooling. A
    3072 -> bottleneck projection (Linear/ReLU/Dropout), a per-bin attention
    score (bottleneck -> hidden Tanh -> 1) softmaxed over bins with a gene-body
    mask restricting attention to bins overlapping the target transcript, a
    mask-renormalized weighted sum over bins, dropout, and a linear read-out.

  attention_head=False (ablation): extract the central bin (index 448) and apply
    a single linear layer to a scalar.
"""

import torch
import torch.nn as nn
from enformer_pytorch import Enformer

# Monkey-patch to fix compatibility with newer transformers versions (e.g. 5.14+)
if not hasattr(Enformer, 'all_tied_weights_keys'):
    Enformer.all_tied_weights_keys = property(lambda self: {})

from enformer_pytorch import from_pretrained

from config import PASTLConfig


class PASTLFinetune(nn.Module):
    def __init__(self, cfg: PASTLConfig):
        super().__init__()
        self.enformer_dim = 3072  # standard Enformer output dims
        self.attention_head = getattr(cfg, 'attention_head', False)
        
        # --- Enformer trunk ---
        self.enformer = from_pretrained(
            'EleutherAI/enformer-official-rough',
            use_tf_gamma=False
        )

        # Enable activation checkpointing to save VRAM across 11 transformer blocks
        if hasattr(self.enformer, 'use_checkpointing'):
            self.enformer.use_checkpointing = True

        # Freeze everything first
        for param in self.enformer.parameters():
            param.requires_grad = False
            
        if self.attention_head:
            # --- Complex Projection Head ---
            self.pointwise = nn.Sequential(
                nn.Linear(self.enformer_dim, cfg.bottleneck_dim),
                nn.ReLU(),
                nn.Dropout(cfg.bottleneck_dropout)
            )
            self.bin_weights = nn.Sequential(
                nn.Linear(cfg.bottleneck_dim, cfg.hidden_dim),
                nn.Tanh(),
                nn.Linear(cfg.hidden_dim, 1)
            )
            self.softmax = nn.Softmax(dim=1)
            self.final_dropout = nn.Dropout(cfg.head_dropout)
            self.output_head = nn.Linear(cfg.bottleneck_dim, 1)
            self._init_weights()
        else:
            # --- Simple Linear Head ---
            # Single linear layer directly mapped from Enformer dim (3072) to scalar (1)
            self.output_head = nn.Linear(self.enformer_dim, 1)

    def unfreeze_trunk(self):
        """Unfreeze the upper transformer blocks and final pointwise layer."""
        for name, param in self.enformer.named_parameters():
            if 'transformer' in name or 'final_pointwise' in name:
                param.requires_grad = True

    def _init_weights(self):
        """Initialize head weights for regression stability."""
        for m in self.pointwise:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        for m in self.bin_weights:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.xavier_normal_(self.output_head.weight)
        nn.init.constant_(self.output_head.bias, 0)

    def trunk_parameters(self):
        """Returns parameters for the Enformer trunk (transformer + pointwise) that are subject to finetuning."""
        return [p for n, p in self.enformer.named_parameters() if 'transformer' in n or 'final_pointwise' in n]

    def head_parameters(self):
        """Returns parameters for the prediction head."""
        if self.attention_head:
            params = []
            for module in [self.pointwise, self.bin_weights, self.output_head, self.final_dropout]:
                params.extend([p for p in module.parameters() if p.requires_grad])
            return params
        return self.output_head.parameters()

    def forward(self, dna_seq, return_features=False, gene_mask=None, **kwargs):
        """
        Forward pass.
        Args:
            dna_seq: [B, L, 4] tensor of one-hot encoded DNA sequences.
            gene_mask: [B, 896] gene-body mask used by the attention head.
        Returns:
            [B, 1] tensor of predicted z-scores.
        """
        # Enformer trunk -> embeddings [B, 896, 3072]
        _, embeddings = self.enformer(dna_seq, return_embeddings=True)

        if self.attention_head:
            embeddings = self.pointwise(embeddings)
            weights = self.bin_weights(embeddings)
            if gene_mask is not None:
                is_intergenic = (gene_mask.unsqueeze(-1) == 0.0)
                weights = weights.masked_fill(is_intergenic, -1e9)
            weights = self.softmax(weights)
            if gene_mask is not None:
                weights = weights * gene_mask.unsqueeze(-1)
                weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
            latent_feats = torch.sum(embeddings * weights, dim=1)
            prediction = self.output_head(self.final_dropout(latent_feats))
        else:
            # Extract the central spatial bin
            # Enformer outputs 896 bins. The exact centre is bin index 448 (0-indexed).
            center_bin_idx = embeddings.shape[1] // 2
            
            # Shape: [B, 3072]
            latent_feats = embeddings[:, center_bin_idx, :]

            # Project to a scalar
            prediction = self.output_head(latent_feats)

        if return_features:
            return prediction, latent_feats

        return prediction
