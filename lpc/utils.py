import numpy as np
import torch
import torch.distributed as dist


def load_checkpoint(filename):
    """Load a checkpoint from file."""
    checkpoint = torch.load(filename, map_location='cpu')
    return checkpoint


def save_checkpoint(state, filename):
    """Save a checkpoint to file."""
    torch.save(state, filename)

