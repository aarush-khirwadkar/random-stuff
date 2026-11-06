"""Diagnostics for predictive latent-space models.

Four carry over from Mini-Projects I and II (effective rank, output std, CLS-patch
similarity, patch locality). Two are new and specific to methods whose TARGET is
produced by a network rather than read off the data:

  target_drift     how much the teacher's description of a fixed batch changed since
                   the last epoch. Zero for a frozen/fixed teacher; positive for EMA.
                   "The right-hand side of the loss moves."
  st_agreement     cosine similarity between the student's and the teacher's patch
                   features on the same full images. Near 1 means the two have become
                   closely coupled -- the target is self-generated.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def effective_rank(feats):
    """exp(entropy of normalized squared singular values) of an (n, d) batch.
    Scale-invariant: cannot see a constant representation -- use output_std for that."""
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
    """Mean cosine similarity between the CLS vector and its own patch tokens."""
    c = F.normalize(cls, dim=-1).unsqueeze(1)
    p = F.normalize(patches, dim=-1)
    return (c * p).sum(-1).mean().item()


@torch.no_grad()
def patch_locality(patches, grid, k=4):
    """Mean grid distance from each patch to its k most similar patches (normalised).
    Small: patch features encode local content. Drifting up: patches interchangeable."""
    B, N, d = patches.shape
    p = F.normalize(patches, dim=-1)
    sim = torch.bmm(p, p.transpose(1, 2))
    sim = sim - torch.eye(N, device=p.device).unsqueeze(0) * 10
    idx = sim.topk(k, dim=-1).indices
    rows = torch.arange(N, device=p.device) // grid
    cols = torch.arange(N, device=p.device) % grid
    r0, c0 = rows.view(1, N, 1), cols.view(1, N, 1)
    r1, c1 = rows[idx], cols[idx]
    dist = ((r1 - r0).abs() + (c1 - c0).abs()).float()
    return (dist.mean() / (2 * (grid - 1))).item()


@torch.no_grad()
def target_drift(prev_patches, cur_patches):
    """1 - mean cosine between the teacher's patch features now and one epoch ago,
    on the same images. 0 = the target did not move."""
    if prev_patches is None:
        return float("nan")
    a = F.normalize(prev_patches, dim=-1)
    b = F.normalize(cur_patches, dim=-1)
    return (1 - (a * b).sum(-1)).mean().item()


@torch.no_grad()
def student_teacher_agreement(student_patches, teacher_patches):
    """Mean cosine between student and teacher patch features on full images.
    Rises toward 1 as the two networks become coupled."""
    a = F.normalize(student_patches, dim=-1)
    b = F.normalize(teacher_patches, dim=-1)
    return (a * b).sum(-1).mean().item()
