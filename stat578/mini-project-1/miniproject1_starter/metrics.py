"""Representation diagnostics: alignment, uniformity, and effective rank.

alignment / uniformity follow Wang & Isola (2020); effective rank follows the
entropy-of-singular-values definition used in Lecture 3.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def alignment(z1, z2):
    """Mean squared distance between L2-normalized positive pairs. Lower = more aligned."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    return ((z1 - z2) ** 2).sum(dim=1).mean().item()


@torch.no_grad()
def uniformity(z, t=2.0):
    """log E exp(-t * ||zi - zj||^2) over L2-normalized features. Lower = more uniform."""
    z = F.normalize(z, dim=1)
    sq_pdist = torch.pdist(z, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log().item()


@torch.no_grad()
def output_std(z):
    """Mean over dimensions of the per-dimension std of L2-normalized features.
    ~1/sqrt(d) for a healthy representation, ~0 under COMPLETE collapse (constant output).
    This is the collapse signal plotted in the SimSiam paper; note that effective rank
    (below) is scale-invariant and instead detects DIMENSIONAL collapse."""
    z = F.normalize(z, dim=1)
    return z.std(dim=0).mean().item()


@torch.no_grad()
def effective_rank(feats):
    """Effective rank of a feature batch (n, d): exp(entropy of normalized singular values).
    Returns (effective_rank, singular_values). A small value signals dimensional collapse."""
    feats = feats - feats.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(feats.float())          # singular values of centered features
    ev = s ** 2
    p = ev / ev.sum().clamp_min(1e-12)
    entropy = -(p * p.clamp_min(1e-12).log()).sum()
    return torch.exp(entropy).item(), s.detach().cpu()
