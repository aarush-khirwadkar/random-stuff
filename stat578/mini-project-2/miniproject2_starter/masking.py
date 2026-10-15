"""The masking distribution pi: WHERE do we hide patches, and HOW MANY?

This is the central design object of the project. Every function here returns the
same three tensors, so the training code never needs to know which pattern was used:

    ids_keep    (B, n_keep)  indices of the VISIBLE patches, per image
    ids_restore (B, N)       inverse permutation, used to put patches back in order
    mask        (B, N)       0 = visible, 1 = hidden

Fixing n_keep = round(N * (1 - ratio)) for every image keeps the tensors rectangular,
which is what lets the encoder run on visible patches only (the MAE trick).
"""
import torch


def _from_scores(scores, ratio):
    """Keep the n_keep patches with the SMALLEST scores.

    Every masking pattern below is just a different way of assigning scores:
    give a patch a low score to make it visible, a high score to hide it.
    """
    B, N = scores.shape
    n_keep = max(1, int(round(N * (1.0 - ratio))))
    ids_shuffle = torch.argsort(scores, dim=1)          # ascending
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :n_keep]

    mask = torch.ones(B, N, device=scores.device)
    mask[:, :n_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)  # back to patch order
    return ids_keep, ids_restore, mask


def random_mask(B, grid, ratio, device):
    """Uniformly random patches, independently for each image (MAE's choice)."""
    N = grid * grid
    return _from_scores(torch.rand(B, N, device=device), ratio)


def block_mask(B, grid, ratio, device, block=2):
    """Contiguous square blocks, as in BEiT's blockwise masking.

    Patches are scored by the block they belong to, so whole blocks tend to be
    hidden together. A small jitter breaks ties inside a block.
    """
    n_blocks = grid // block
    blk = torch.rand(B, n_blocks, n_blocks, device=device)
    scores = blk.repeat_interleave(block, dim=1).repeat_interleave(block, dim=2)
    scores = scores + 0.01 * torch.rand(B, grid, grid, device=device)
    return _from_scores(scores.reshape(B, -1), ratio)


def grid_mask(B, grid, ratio, device):
    """Checkerboard-like: hidden patches are spread as evenly as possible.

    This is the OPPOSITE extreme from block masking -- every hidden patch keeps
    visible neighbours, so interpolation remains easy. Useful as a control.
    """
    N = grid * grid
    rows = torch.arange(grid, device=device).view(-1, 1)
    cols = torch.arange(grid, device=device).view(1, -1)
    base = ((rows + cols) % 2).float().reshape(1, N)      # 0/1 checkerboard
    scores = base + 0.5 * torch.rand(B, N, device=device)
    return _from_scores(scores, ratio)


PATTERNS = {"random": random_mask, "block": block_mask, "grid": grid_mask}


def make_mask(pattern, B, grid, ratio, device):
    if pattern not in PATTERNS:
        raise ValueError(f"unknown mask pattern {pattern!r}; choose from {list(PATTERNS)}")
    return PATTERNS[pattern](B, grid, ratio, device)
