"""Unit tests for model.losses.build_criterion."""

import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from losses import build_criterion


def test_build_criterion():
    criterion = build_criterion()
    assert isinstance(criterion, nn.HuberLoss)
    
    # Test standard output
    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([1.0, 2.0])
    loss = criterion(pred, target)
    assert loss.item() == 0.0
    
    # Test penalization
    pred_bad = torch.tensor([1.0, 5.0])
    loss_bad = criterion(pred_bad, target)
    assert loss_bad.item() > 0.0
