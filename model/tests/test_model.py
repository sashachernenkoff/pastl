"""Smoke tests for the PASTLFinetune model: forward/backward pass and gradient flow."""

import sys
from pathlib import Path
import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PASTLConfig
from model import PASTLFinetune


def test_single_batch_forward_backward():
    """Smoke test to ensure the model can process a batch and backpropagate."""
    cfg = PASTLConfig()
    
    # We don't need a GPU for a quick smoke test if we just want to test shapes
    # but we can let it run on GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = PASTLFinetune(cfg).to(device)
    
    # Create a dummy batch (B=2, L=196608, C=4)
    B = 2
    dummy_dna = torch.zeros((B, 196608, 4), device=device)
    dummy_label = torch.tensor([1.0, -1.0], device=device)
    
    # Use AMP for realistic memory testing if on GPU
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32):
        out = model(dummy_dna).squeeze(-1)
        loss = torch.nn.functional.huber_loss(out, dummy_label)
    
    assert out.shape == (B,)
    
    # Backward pass
    loss.backward()
    
    # Verify gradients are flowing to the unfrozen head
    assert model.output_head.weight.grad is not None
    
    # Verify frozen trunk (CNN) has no gradients
    for name, param in model.enformer.named_parameters():
        if 'transformer' not in name and 'final_pointwise' not in name:
            assert param.grad is None
