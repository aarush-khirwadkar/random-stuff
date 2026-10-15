"""Masked image modeling with two choices of target Y.

Both variants share one encoder, one decoder and one masking module. The ONLY
difference is what sits on the right-hand side of the loss:

    target = "pixel"    Y = the patch's raw (normalized) pixels      -- MAE
    target = "feature"  Y = an EMA teacher's patch representation    -- iBOT-style

Holding everything else fixed is what makes the comparison a controlled experiment.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import masking
from vit import Encoder, Decoder, patchify


class MaskedModel(nn.Module):
    def __init__(self, target="pixel", img=32, patch=4, dim=192, depth=6, heads=3,
                 dec_dim=96, dec_depth=2, momentum=0.996, center=True, center_m=0.9):
        super().__init__()
        assert target in ("pixel", "feature")
        self.target = target
        self.patch = patch
        self.encoder = Encoder(img, patch, dim, depth, heads)
        self.grid = self.encoder.grid
        self.n_patches = self.encoder.n_patches

        out_dim = patch * patch * 3 if target == "pixel" else dim
        self.decoder = Decoder(self.n_patches, dim, dec_dim, dec_depth, heads, out_dim)

        # The teacher exists only for the feature target. It is an EMA copy of the
        # encoder and is never touched by gradients.
        self.momentum = momentum
        if target == "feature":
            self.teacher = copy.deepcopy(self.encoder)
            for p in self.teacher.parameters():
                p.requires_grad = False
            # Predicting your own features admits a constant solution. Centering the
            # teacher -- subtracting a running mean, as in DINO -- discourages every
            # patch from mapping to the same vector. Disable it with --no_center to
            # see what happens without any anti-collapse device.
            self.use_center = center
            self.center_m = center_m
            self.register_buffer("center", torch.zeros(1, 1, dim))
        else:
            self.teacher = None
            self.use_center = False

    @torch.no_grad()
    def update_teacher(self):
        """EMA update; a no-op for the pixel target, which has no teacher."""
        if self.teacher is None:
            return
        m = self.momentum
        for ps, pt in zip(self.encoder.parameters(), self.teacher.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1 - m)
        for bs, bt in zip(self.encoder.buffers(), self.teacher.buffers()):
            bt.data.copy_(bs.data)

    @torch.no_grad()
    def _pixel_target(self, imgs, norm_pix=True):
        y = patchify(imgs, self.patch)
        if norm_pix:                       # per-patch standardisation (as in MAE)
            mu = y.mean(dim=-1, keepdim=True)
            var = y.var(dim=-1, keepdim=True)
            y = (y - mu) / (var + 1e-6).sqrt()
        return y

    @torch.no_grad()
    def _feature_target(self, imgs):
        _, patches = self.teacher.features(imgs)             # (B, N, dim)
        if self.use_center:
            patches = patches - self.center
            batch_mean = patches.mean(dim=(0, 1), keepdim=True)
            self.center.mul_(self.center_m).add_(batch_mean, alpha=1 - self.center_m)
        return F.normalize(patches, dim=-1)

    def forward(self, imgs, pattern="random", ratio=0.75, norm_pix=True):
        B = imgs.shape[0]
        ids_keep, ids_restore, mask = masking.make_mask(
            pattern, B, self.grid, ratio, imgs.device)

        latent = self.encoder(imgs, ids_keep=ids_keep)       # visible patches only
        pred = self.decoder(latent, ids_restore)             # (B, N, out_dim)

        if self.target == "pixel":
            y = self._pixel_target(imgs, norm_pix)
            # mean squared error over the patch's pixel dimensions
            per_patch = (pred - y).pow(2).mean(dim=-1)       # (B, N)
        else:
            y = self._feature_target(imgs)
            pred = F.normalize(pred, dim=-1)
            # squared distance between unit vectors = 2 - 2*cos. Summing over the
            # feature dimension (rather than averaging) keeps the gradient scale
            # comparable to the pixel branch; this is the BYOL/DINO-style loss.
            per_patch = 2 - 2 * (pred * y).sum(dim=-1)       # (B, N)
        loss = (per_patch * mask).sum() / mask.sum().clamp_min(1.0)
        return loss, pred, mask

    @torch.no_grad()
    def features(self, imgs):
        return self.encoder.features(imgs)


def build_model(args):
    return MaskedModel(target=args.target, dim=args.dim, depth=args.depth,
                       dec_dim=args.dec_dim, dec_depth=args.dec_depth,
                       momentum=args.momentum,
                       center=not getattr(args, "no_center", False))
