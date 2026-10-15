"""A small Vision Transformer, and the asymmetric encoder/decoder MAE uses.

Deliberately compact: 32x32 CIFAR images, 4x4 patches, so grid = 8 and N = 64
patches. The encoder is the thing we keep; the decoder is thrown away after
pretraining.
"""
import math
import torch
import torch.nn as nn


def sincos_pos_embed(n_pos, dim, device=None):
    """Fixed sine-cosine position embeddings (no parameters to learn)."""
    pos = torch.arange(n_pos, dtype=torch.float, device=device).unsqueeze(1)
    i = torch.arange(dim // 2, dtype=torch.float, device=device).unsqueeze(0)
    angle = pos / torch.pow(10000.0, 2 * i / dim)
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=1)


class Block(nn.Module):
    """One pre-norm Transformer block: attention, then MLP."""
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.n2(x))
        return x


class PatchEmbed(nn.Module):
    """Cut the image into patches and linearly embed each one."""
    def __init__(self, img=32, patch=4, in_ch=3, dim=192):
        super().__init__()
        assert img % patch == 0
        self.grid = img // patch
        self.n_patches = self.grid ** 2
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)      # (B, N, dim)


def patchify(imgs, patch):
    """(B,3,H,W) -> (B, N, patch*patch*3): the raw pixel target for each patch."""
    B, C, H, W = imgs.shape
    g = H // patch
    x = imgs.reshape(B, C, g, patch, g, patch)
    x = torch.einsum("bcgphq->bghqpc", x)                   # (B, g, g, patch, patch, C)
    return x.reshape(B, g * g, patch * patch * C)


def unpatchify(patches, patch, grid, C=3):
    """Inverse of patchify, used only for visualising reconstructions."""
    B = patches.shape[0]
    x = patches.reshape(B, grid, grid, patch, patch, C)
    x = torch.einsum("bghqpc->bcgphq", x)
    return x.reshape(B, C, grid * patch, grid * patch)


class Encoder(nn.Module):
    """ViT encoder. Can run on ALL patches, or on a subset (the MAE trick)."""
    def __init__(self, img=32, patch=4, dim=192, depth=6, heads=3):
        super().__init__()
        self.embed = PatchEmbed(img, patch, 3, dim)
        self.grid = self.embed.grid
        self.n_patches = self.embed.n_patches
        self.dim = dim
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.register_buffer("pos", sincos_pos_embed(self.n_patches + 1, dim).unsqueeze(0))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x, ids_keep=None):
        """ids_keep=None processes every patch; otherwise only the visible ones."""
        B = x.shape[0]
        t = self.embed(x) + self.pos[:, 1:, :]
        if ids_keep is not None:
            idx = ids_keep.unsqueeze(-1).expand(-1, -1, self.dim)
            t = torch.gather(t, dim=1, index=idx)           # keep visible patches only
        cls = (self.cls + self.pos[:, :1, :]).expand(B, -1, -1)
        t = torch.cat([cls, t], dim=1)
        for b in self.blocks:
            t = b(t)
        return self.norm(t)                                 # (B, 1 + n_kept, dim)

    @torch.no_grad()
    def features(self, x):
        """Frozen-feature extraction: CLS vector and patch tokens for a full image."""
        t = self.forward(x, ids_keep=None)
        return t[:, 0], t[:, 1:]


class Decoder(nn.Module):
    """Small decoder used only during pretraining, then discarded.

    out_dim decides WHAT is predicted:
      patch*patch*3  -> raw pixels          (the MAE target)
      encoder dim    -> teacher features    (the iBOT-style target)
    """
    def __init__(self, n_patches, enc_dim=192, dim=96, depth=2, heads=3, out_dim=48):
        super().__init__()
        self.proj = nn.Linear(enc_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.register_buffer("pos", sincos_pos_embed(n_patches + 1, dim).unsqueeze(0))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, latent, ids_restore):
        B, N = ids_restore.shape
        x = self.proj(latent)                               # (B, 1 + n_kept, dim)
        n_kept = x.shape[1] - 1
        pad = self.mask_token.expand(B, N - n_kept, -1)     # mask tokens enter HERE
        seq = torch.cat([x[:, 1:, :], pad], dim=1)
        idx = ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        seq = torch.gather(seq, dim=1, index=idx)           # restore patch order
        seq = torch.cat([x[:, :1, :], seq], dim=1) + self.pos
        for b in self.blocks:
            seq = b(seq)
        return self.head(self.norm(seq))[:, 1:, :]          # drop CLS -> (B, N, out_dim)
