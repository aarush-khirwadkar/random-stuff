"""Predictive modeling of hidden patches, with every design axis of Week 10 as a flag.

One model class. Everything is held fixed except what you change:

    --target      pixel | feature | code      IN WHAT SPACE is the target?
    --teacher     ema | frozen | fixed        WHO constructs the target (feature/code)?
    --teacher_sees full | targets            does the teacher see the whole image, or
                                              only the hidden patches it must describe?
    --pred_depth  k                           an OUTSIDE predictor with k blocks
    --mask_in_encoder                         ... or mask tokens INSIDE the encoder and
                                              a linear head (iBOT/BEiT-style)
    --pattern / --ratio                       WHAT is hidden (masking.py)
    --no_center                               remove the anti-collapse device (feature)

Targets
-------
  pixel    Y = per-patch standardised pixels                       (MAE)
  feature  Y = a teacher's patch representation, centered + L2-normalised
               loss = 2 - 2 cos(pred, Y)                            (I-JEPA-style)
  code     Y = the index of the nearest entry of an online codebook fitted to the
               teacher's patch representations; loss = cross-entropy over K codes
                                                                    (BEiT/HuBERT-style,
                                                                     with an online
                                                                     tokenizer)
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import masking
from vit import Encoder, Predictor, LinearHead, patchify


class Codebook(nn.Module):
    """An online tokenizer: K prototypes fitted to teacher features by EMA k-means.

    assign(x): nearest prototype (cosine) for each row of x -- the discrete target.
    update(x): move prototypes toward the features assigned to them (EMA), and
               re-seed any prototype that has gone unused ("dead codes"), which is
               this target's anti-collapse device: a codebook that used one code would
               make the prediction task trivial.
    """
    def __init__(self, K=256, dim=192, momentum=0.99, dead_threshold=1e-3):
        super().__init__()
        self.K, self.m, self.dead = K, momentum, dead_threshold
        self.register_buffer("codes", F.normalize(torch.randn(K, dim), dim=-1))
        self.register_buffer("usage", torch.full((K,), 1.0 / K))
        self.register_buffer("ema_sum", torch.zeros(K, dim))
        self.register_buffer("ema_cnt", torch.zeros(K))

    @torch.no_grad()
    def assign(self, x):                                    # x: (M, dim), any scale
        x = F.normalize(x, dim=-1)
        return (x @ self.codes.t()).argmax(dim=-1)          # (M,)

    @torch.no_grad()
    def update(self, x):
        x = F.normalize(x, dim=-1)
        idx = self.assign(x)
        onehot = F.one_hot(idx, self.K).float()             # (M, K)
        cnt = onehot.sum(0)
        summ = onehot.t() @ x
        self.ema_cnt.mul_(self.m).add_(cnt, alpha=1 - self.m)
        self.ema_sum.mul_(self.m).add_(summ, alpha=1 - self.m)
        self.usage.mul_(self.m).add_(cnt / cnt.sum().clamp_min(1), alpha=1 - self.m)
        alive = self.ema_cnt > 1e-3
        new = F.normalize(self.ema_sum[alive] / self.ema_cnt[alive].unsqueeze(-1), dim=-1)
        self.codes[alive] = new
        dead = self.usage < self.dead / self.K              # essentially never used
        n_dead = int(dead.sum())
        if n_dead > 0:                                      # re-seed from the batch
            pick = torch.randint(0, x.shape[0], (n_dead,), device=x.device)
            self.codes[dead] = x[pick]
            self.usage[dead] = 1.0 / self.K
            self.ema_cnt[dead] = 1.0
            self.ema_sum[dead] = x[pick]

    @torch.no_grad()
    def perplexity(self):
        """exp(entropy of code usage): K means all codes used equally, 1 means one code."""
        p = self.usage / self.usage.sum().clamp_min(1e-12)
        return torch.exp(-(p * p.clamp_min(1e-12).log()).sum()).item()


class PredictiveModel(nn.Module):
    def __init__(self, target="feature", teacher="ema", teacher_sees="full",
                 pred_depth=2, mask_in_encoder=False, img=32, patch=4, dim=192, depth=6,
                 heads=3, pred_dim=96, momentum=0.996, center=True, center_m=0.9,
                 codebook_size=256, teacher_ckpt=None):
        super().__init__()
        assert target in ("pixel", "feature", "code")
        assert teacher in ("ema", "frozen", "fixed")
        assert teacher_sees in ("full", "targets")
        self.target, self.teacher_kind, self.teacher_sees = target, teacher, teacher_sees
        self.mask_in_encoder = mask_in_encoder
        self.patch, self.momentum = patch, momentum

        self.encoder = Encoder(img, patch, dim, depth, heads)
        self.grid, self.n_patches = self.encoder.grid, self.encoder.n_patches

        out_dim = {"pixel": patch * patch * 3, "feature": dim, "code": codebook_size}[target]
        if mask_in_encoder:
            self.predictor = LinearHead(dim, out_dim)
        else:
            self.predictor = Predictor(self.n_patches, dim, pred_dim, pred_depth, heads, out_dim)

        # ---- the teacher: who constructs the target -------------------------------
        self.teacher = None
        if target in ("feature", "code"):
            self.teacher = copy.deepcopy(self.encoder)
            if teacher == "fixed":
                assert teacher_ckpt is not None, "--teacher fixed needs --teacher_ckpt"
                ck = torch.load(teacher_ckpt, map_location="cpu")
                self.teacher.load_state_dict(ck["encoder"])
            for p in self.teacher.parameters():
                p.requires_grad = False
            # 'ema'    : teacher tracks the student  (online, moving target)
            # 'frozen' : teacher = random init, never updated (a frozen random tokenizer)
            # 'fixed'  : teacher = a pretrained encoder, never updated (semantic but stale)

        # ---- anti-collapse device for the feature target: DINO-style centering -----
        self.use_center = center and target == "feature"
        self.center_m = center_m
        self.register_buffer("center", torch.zeros(1, 1, dim))

        # ---- online tokenizer for the code target -----------------------------------
        self.codebook = Codebook(codebook_size, dim) if target == "code" else None

    # ------------------------------------------------------------------ teacher ----
    @torch.no_grad()
    def update_teacher(self):
        """EMA update; a no-op for pixel targets and for frozen / fixed teachers."""
        if self.teacher is None or self.teacher_kind != "ema":
            return
        m = self.momentum
        for ps, pt in zip(self.encoder.parameters(), self.teacher.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1 - m)
        for bs, bt in zip(self.encoder.buffers(), self.teacher.buffers()):
            bt.data.copy_(bs.data)

    @torch.no_grad()
    def teacher_patches(self, imgs, ids_restore, n_keep):
        """Teacher patch representations at every position, (B, N, dim).

        teacher_sees='full'    : encode the whole image, read off all N positions --
                                 each target is CONTEXTUALISED by the rest of the image.
        teacher_sees='targets' : encode only the hidden patches (no visible context);
                                 visible positions get zeros (they carry no loss).
        """
        B = imgs.shape[0]
        if self.teacher_sees == "full":
            _, patches = self.teacher.features(imgs)
            return patches
        ids_hid = masking.hidden_ids(ids_restore, n_keep)          # (B, N - n_keep)
        t = self.teacher(imgs, ids_keep=ids_hid)[:, 1:, :]          # (B, N - n_keep, dim)
        out = torch.zeros(B, self.n_patches, t.shape[-1], device=imgs.device)
        out.scatter_(1, ids_hid.unsqueeze(-1).expand(-1, -1, t.shape[-1]), t)
        return out

    # ------------------------------------------------------------------ targets ----
    @torch.no_grad()
    def _pixel_target(self, imgs, norm_pix=True):
        y = patchify(imgs, self.patch)
        if norm_pix:
            mu = y.mean(dim=-1, keepdim=True)
            var = y.var(dim=-1, keepdim=True)
            y = (y - mu) / (var + 1e-6).sqrt()
        return y

    @torch.no_grad()
    def _feature_target(self, patches, loss_mask):
        if self.use_center:
            patches = patches - self.center
            sel = patches[loss_mask.bool()]                 # update on scored positions
            self.center.mul_(self.center_m).add_(sel.mean(0).view(1, 1, -1),
                                                 alpha=1 - self.center_m)
        return F.normalize(patches, dim=-1)

    @torch.no_grad()
    def _code_target(self, patches, loss_mask):
        sel = patches[loss_mask.bool()]                     # (M, dim)
        if self.training:
            self.codebook.update(sel)
        return self.codebook.assign(patches.reshape(-1, patches.shape[-1])).view(
            patches.shape[0], patches.shape[1])             # (B, N) long

    # ------------------------------------------------------------------ forward ----
    def forward(self, imgs, pattern="multiblock", ratio=0.6, norm_pix=True):
        B = imgs.shape[0]
        ids_keep, ids_restore, mask, loss_mask = masking.make_mask(
            pattern, B, self.grid, ratio, imgs.device)
        n_keep = ids_keep.shape[1]

        if self.mask_in_encoder:
            latent = self.encoder(imgs, mask=mask)          # all positions, mask tokens
        else:
            latent = self.encoder(imgs, ids_keep=ids_keep)  # visible patches only
        pred = self.predictor(latent, ids_restore)          # (B, N, out_dim)

        if self.target == "pixel":
            y = self._pixel_target(imgs, norm_pix)
            per_patch = (pred - y).pow(2).mean(dim=-1)
        else:
            tp = self.teacher_patches(imgs, ids_restore, n_keep)
            if self.target == "feature":
                y = self._feature_target(tp, loss_mask)
                pred = F.normalize(pred, dim=-1)
                per_patch = 2 - 2 * (pred * y).sum(dim=-1)  # squared distance on the sphere
            else:
                y = self._code_target(tp, loss_mask)
                per_patch = F.cross_entropy(pred.transpose(1, 2), y, reduction="none")
        loss = (per_patch * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)
        return loss, pred, mask, loss_mask

    @torch.no_grad()
    def features(self, imgs):
        return self.encoder.features(imgs)


def build_model(args):
    return PredictiveModel(
        target=args.target, teacher=args.teacher, teacher_sees=args.teacher_sees,
        pred_depth=args.pred_depth, mask_in_encoder=args.mask_in_encoder,
        dim=args.dim, depth=args.depth, pred_dim=args.pred_dim, momentum=args.momentum,
        center=not args.no_center, codebook_size=args.codebook_size,
        teacher_ckpt=args.teacher_ckpt)
