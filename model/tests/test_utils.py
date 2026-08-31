"""Unit tests for model.utils (ensure_dir, count_parameters)."""

import sys
from pathlib import Path
import os
import tempfile
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import ensure_dir, count_parameters


def test_ensure_dir():
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "new_dir", "nested")
        ensure_dir(target)
        assert os.path.exists(target)
        
        # Test idempotency (shouldn't crash if it already exists)
        ensure_dir(target)
        assert os.path.exists(target)


def test_count_parameters():
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.Linear(5, 2)
    )
    
    total, trainable = count_parameters(model)
    expected = (10 * 5 + 5) + (5 * 2 + 2)
    
    assert total == expected
    assert trainable == expected
    
    # Freeze the first layer
    model[0].weight.requires_grad = False
    total, trainable = count_parameters(model)
    
    assert total == expected
    assert trainable == expected - (10 * 5)
