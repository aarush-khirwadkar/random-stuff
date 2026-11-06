"""The masking distribution pi: WHAT is hidden, and WHERE the loss is applied.

Every function returns the same four tensors, so the training code never needs to
know which pattern was used:

    ids_keep    (B, n_keep)  indices of the VISIBLE patches, per image
    ids_restore (B, N)       inverse permutation, used to put patches back in order
    mask        (B, N)       0 = visible, 1 = hidden      (what the encoder does not see)
    loss_mask   (B, N)       1 = a prediction is scored here

For random / block / grid masking (Mini-Project II) loss_mask == mask: every hidden
patch is predicted. Multi-block masking (I-JEPA) is different: the hidden set is
"everything outside one large context block, plus the target blocks", but the loss is
scored ONLY on the target blocks. Patches that are neither context nor target are
simply dropped -- they are hidden, but nothing is predicted there.

Fixing n_keep = round(N * (1 - ratio)) for every image keeps the tensors rectangular,
which is what lets the encoder run on visible patches only (the MAE trick).
"""
import torch


def _from_scores(scores, ratio):
    """Keep the n_keep patches with the SMALLEST scores.

    Every masking pattern is just a different way of assigning scores: give a patch
    a low score to make it visible, a high score to hide it.
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
    ids_keep, ids_restore, mask = _from_scores(torch.rand(B, N, device=device), ratio)
    return ids_keep, ids_restore, mask, mask.clone()


def block_mask(B, grid, ratio, device, block=2):
    """Contiguous square blocks, as in BEiT's blockwise masking."""
    n_blocks = grid // block
    blk = torch.rand(B, n_blocks, n_blocks, device=device)
    scores = blk.repeat_interleave(block, dim=1).repeat_interleave(block, dim=2)
    scores = scores + 0.01 * torch.rand(B, grid, grid, device=device)
    ids_keep, ids_restore, mask = _from_scores(scores.reshape(B, -1), ratio)
    return ids_keep, ids_restore, mask, mask.clone()


def grid_mask(B, grid, ratio, device):
    """Checkerboard-like: every hidden patch keeps visible neighbours (a control)."""
    N = grid * grid
    rows = torch.arange(grid, device=device).view(-1, 1)
    cols = torch.arange(grid, device=device).view(1, -1)
    base = ((rows + cols) % 2).float().reshape(1, N)
    scores = base + 0.5 * torch.rand(B, N, device=device)
    ids_keep, ids_restore, mask = _from_scores(scores, ratio)
    return ids_keep, ids_restore, mask, mask.clone()


def _rects(B, grid, scale, aspect, device):
    """Sample B rectangles on a grid x grid lattice; return a (B, grid, grid) 0/1 map.

    scale  = (lo, hi): area as a fraction of the grid, sampled uniformly
    aspect = (lo, hi): height / width, sampled uniformly
    """
    N = grid * grid
    s = torch.empty(B, device=device).uniform_(*scale) * N
    r = torch.empty(B, device=device).uniform_(*aspect)
    h = (s * r).sqrt().round().clamp(1, grid).long()
    w = (s / r).sqrt().round().clamp(1, grid).long()
    top = (torch.rand(B, device=device) * (grid - h + 1).float()).floor().long()
    left = (torch.rand(B, device=device) * (grid - w + 1).float()).floor().long()
    rows = torch.arange(grid, device=device).view(1, grid, 1)
    cols = torch.arange(grid, device=device).view(1, 1, grid)
    in_r = (rows >= top.view(B, 1, 1)) & (rows < (top + h).view(B, 1, 1))
    in_c = (cols >= left.view(B, 1, 1)) & (cols < (left + w).view(B, 1, 1))
    return (in_r & in_c).float()


def multiblock_mask(B, grid, ratio, device, n_targets=4, target_scale=(0.15, 0.20),
                    target_aspect=(0.75, 1.5), context_scale=(0.85, 1.0)):
    """I-JEPA's multi-block masking, on a small grid.

    Targets: n_targets rectangles, each covering target_scale of the image (they may
             overlap). The loss is scored on their union.
    Context: ONE large rectangle (context_scale of the image, unit aspect ratio) with
             every target patch removed from it. The encoder sees the context only.

    Scores: context-minus-targets < outside-everything < targets, with random jitter
    inside each group, so the n_keep visible patches are drawn from the context first
    and target patches are the LAST thing that could ever become visible. With the
    default ratio (0.75, n_keep = 16 of 64) they never are.
    """
    N = grid * grid
    tgt = torch.zeros(B, grid, grid, device=device)
    for _ in range(n_targets):
        tgt = torch.maximum(tgt, _rects(B, grid, target_scale, target_aspect, device))
    ctx = _rects(B, grid, context_scale, (1.0, 1.0), device)
    ctx_only = ctx * (1 - tgt)

    scores = torch.rand(B, grid, grid, device=device)
    scores = scores + 1.0 * (1 - ctx_only) + 1.0 * tgt      # 0-1 ctx, 1-2 outside, 2-3 tgt
    ids_keep, ids_restore, mask = _from_scores(scores.reshape(B, -1), ratio)
    loss_mask = tgt.reshape(B, -1) * mask                     # targets that are hidden
    return ids_keep, ids_restore, mask, loss_mask


PATTERNS = {"random": random_mask, "block": block_mask, "grid": grid_mask,
            "multiblock": multiblock_mask}


def make_mask(pattern, B, grid, ratio, device):
    if pattern not in PATTERNS:
        raise ValueError(f"unknown mask pattern {pattern!r}; choose from {list(PATTERNS)}")
    return PATTERNS[pattern](B, grid, ratio, device)


def hidden_ids(ids_restore, n_keep):
    """Indices of the HIDDEN patches, per image (rectangular: N - n_keep each)."""
    ids_shuffle = torch.argsort(ids_restore, dim=1)
    return ids_shuffle[:, n_keep:]
