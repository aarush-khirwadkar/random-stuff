"""Diagnostics for masked image modeling.

Two of these carry over from Mini-Project I (effective rank, output std). The
rest are specific to models that produce representations at TWO levels: one
vector per image (CLS) and one per patch.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def effective_rank(feats):
    """exp(entropy of normalized squared singular values) of an (n, d) batch.

    Small values signal dimensional collapse. Scale-invariant, so it will NOT
    detect a constant representation -- use output_std for that.
    Returns (effective_rank, singular_values).
    """
    feats = feats - feats.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(feats.float())
    ev = s ** 2
    p = ev / ev.sum().clamp_min(1e-12)
    entropy = -(p * p.clamp_min(1e-12).log()).sum()
    return torch.exp(entropy).item(), s.detach().cpu()


@torch.no_grad()
def output_std(z):
    """Mean per-dimension std of L2-normalized features; ~0 means complete collapse."""
    z = F.normalize(z, dim=-1)
    return z.std(dim=0).mean().item()


@torch.no_grad()
def cls_patch_similarity(cls, patches):
    """Mean cosine similarity between the CLS vector and its own patch tokens.

    This is the quantity DINOv3 reports rising over long training: as it grows,
    patch features drift toward the global vector and lose locality.
        cls     (B, d)
        patches (B, N, d)
    """
    c = F.normalize(cls, dim=-1).unsqueeze(1)          # (B, 1, d)
    p = F.normalize(patches, dim=-1)                   # (B, N, d)
    return (c * p).sum(-1).mean().item()


@torch.no_grad()
def patch_locality(patches, grid, k=4):
    """How spatially close are a patch's most similar patches, within one image?

    For every patch we take its k most similar OTHER patches and average their
    grid distance. Small values mean patch features still encode local content;
    the value drifts up if patches all start to look alike.
    Returns the mean distance, normalized by the largest possible distance.
    """
    B, N, d = patches.shape
    p = F.normalize(patches, dim=-1)
    sim = torch.bmm(p, p.transpose(1, 2))                       # (B, N, N)
    sim = sim - torch.eye(N, device=p.device).unsqueeze(0) * 10  # drop self-match
    idx = sim.topk(k, dim=-1).indices                            # (B, N, k)

    rows = torch.arange(N, device=p.device) // grid
    cols = torch.arange(N, device=p.device) % grid
    r0, c0 = rows.view(1, N, 1), cols.view(1, N, 1)
    r1, c1 = rows[idx], cols[idx]
    dist = ((r1 - r0).abs() + (c1 - c0).abs()).float()           # grid L1 distance
    return (dist.mean() / (2 * (grid - 1))).item()


@torch.no_grad()
def masked_recon_error(model, imgs, pattern="random", ratio=0.75):
    """Loss of the current model on a held-out batch, at the given masking design."""
    was_training = model.training
    model.eval()
    loss, _, _ = model(imgs, pattern=pattern, ratio=ratio)
    if was_training:
        model.train()
    return loss.item()
