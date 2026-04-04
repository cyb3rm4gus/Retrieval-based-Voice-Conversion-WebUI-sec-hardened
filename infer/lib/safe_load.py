"""Centralized torch.load wrapper for security-hardened RVC.

All model/checkpoint loading goes through this module.
weights_only=True is enforced to prevent pickle deserialization attacks
(CWE-502) from malicious .pth files. All RVC checkpoints contain only
safe types (dicts, OrderedDicts, tensors, strings, ints, floats, lists).
"""

import torch


def safe_torch_load(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=True)
