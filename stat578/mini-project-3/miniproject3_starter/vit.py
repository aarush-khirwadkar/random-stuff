"""A small Vision Transformer: the encoder we keep, and the predictor we discard.

Deliberately compact: 32x32 CIFAR images, 4x4 patches, so grid = 8 and N = 64
patches. Two places where a prediction for a hidden patch can be formed:

  * OUTSIDE the encoder (MAE, I-JEPA): the encoder sees visible patches only; mask
    tokens enter a separate Predictor, whose depth is a flag. The encoder is never
    shown a symbol it will not see downstream.
  * INSIDE the encoder (BEiT, iBOT): the encoder itself receives a mask token at
    every hidden position, and a linear head reads the prediction off its output.
    (Encoder.forward with mask_in_encoder=True.)
"""
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
    x = torch.einsum("bcgphq->bghqpc", x)
    return x.reshape(B, g * g, patch * patch * C)


def unpatchify(patches, patch, grid, C=3):
    """Inverse of patchify, used only for visualising reconstructions."""
    B = patches.shape[0]
    x = patches.reshape(B, grid, grid, patch, patch, C)
    x = torch.einsum("bghqpc->bcgphq", x)
    return x.reshape(B, C, grid * patch, grid * patch)


class Encoder(nn.Module):
    """ViT encoder. Runs on ALL patches, on a visible SUBSET, or on all patches with
    hidden ones replaced by a mask token (mask_in_encoder)."""
    def __init__(self, img=32, patch=4, dim=192, depth=6, heads=3):
        super().__init__()
        self.embed = PatchEmbed(img, patch, 3, dim)
        self.grid = self.embed.grid
        self.n_patches = self.embed.n_patches
        self.dim = dim
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))   # used only inside-mode
        self.register_buffer("pos", sincos_pos_embed(self.n_patches + 1, dim).unsqueeze(0))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x, ids_keep=None, mask=None):
        """ids_keep=None, mask=None : process every patch (evaluation).
           ids_keep given            : process the visible patches only (outside-mode).
           mask given (B,N, 1=hidden): process every position, hidden ones replaced by
                                       the mask token (inside-mode)."""
        B = x.shape[0]
        t = self.embed(x)
        if mask is not None:
            m = mask.unsqueeze(-1)
            t = t * (1 - m) + self.mask_token.expand(B, self.n_patches, -1) * m
        t = t + self.pos[:, 1:, :]
        if ids_keep is not None:
            idx = ids_keep.unsqueeze(-1).expand(-1, -1, self.dim)
            t = torch.gather(t, dim=1, index=idx)           # keep visible patches only
        cls = (self.cls + self.pos[:, :1, :]).expand(B, -1, -1)
        t = torch.cat([cls, t], dim=1)
        for b in self.blocks:
            t = b(t)
        return self.norm(t)                                 # (B, 1 + n_tokens, dim)

    @torch.no_grad()
    def features(self, x):
        """Frozen-feature extraction on a FULL image: CLS vector and patch tokens."""
        t = self.forward(x)
        return t[:, 0], t[:, 1:]


class Predictor(nn.Module):
    """Small Transformer that turns (context features + mask tokens with positions)
    into a prediction at every patch position. Used only in pretraining, then discarded.

    out_dim decides WHAT is predicted:
      patch*patch*3  -> raw pixels           (target = pixel)
      encoder dim    -> teacher features     (target = feature)
      codebook size  -> logits over codes    (target = code)
    """
    def __init__(self, n_patches, enc_dim=192, dim=96, depth=2, heads=3, out_dim=48):
        super().__init__()
        assert depth >= 1, "an outside predictor needs at least one block to read context"
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


class LinearHead(nn.Module):
    """Inside-mode 'predictor': a linear map on the encoder's own outputs."""
    def __init__(self, enc_dim, out_dim):
        super().__init__()
        self.head = nn.Linear(enc_dim, out_dim)

    def forward(self, latent, ids_restore=None):
        return self.head(latent[:, 1:, :])                  # (B, N, out_dim)
