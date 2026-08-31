"""
Loss functions for PASTL fine-tuning.
"""

import torch.nn as nn


def build_criterion():
    return nn.MSELoss()
