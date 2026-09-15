"""Self-supervised methods: SimCLR (contrastive) and BYOL / SimSiam (non-contrastive).

Each module's forward(x1, x2) returns the training loss for a pair of views.

Ablation flags:
    SimSiam.use_stopgrad  -- if False, gradients flow through the target branch (expect collapse)
    use_predictor         -- if False, the predictor is replaced by identity (expect collapse)

SimSiam and BYOL both use a stop-gradient, but they realize it differently, so the
--no_stopgrad ablation only makes sense for SimSiam:
  * SimSiam's target is the online projector output with an explicit .detach().
    Removing that detach (use_stopgrad=False) genuinely lets gradients flow through
    the target branch -- a real ablation that collapses.
  * BYOL's target is a separate EMA network whose parameters have requires_grad=False.
    Gradients computed through it are discarded regardless, so there is nothing to
    "turn off." BYOL therefore does not accept --no_stopgrad; use SimSiam to see
    stop-gradient removal collapse.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import cifar_encoder, Projector, Predictor


# ----------------------------- SimCLR ---------------------------------------
def nt_xent(z1, z2, temperature=0.5):
    """Normalized temperature-scaled cross entropy over a batch of 2n views."""
    n = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)      # (2n, d)
    sim = z @ z.t() / temperature                           # (2n, 2n)
    sim.fill_diagonal_(float("-inf"))                       # mask self-similarity
    idx = torch.arange(2 * n, device=z.device)
    targets = (idx + n) % (2 * n)                           # positive of i is i+n (mod 2n)
    return F.cross_entropy(sim, targets)


class SimCLR(nn.Module):
    def __init__(self, proj_dim=128, temperature=0.5):
        super().__init__()
        self.encoder, feat = cifar_encoder()
        self.projector = Projector(feat, 512, proj_dim)
        self.temperature = temperature

    def forward(self, x1, x2):
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        return nt_xent(z1, z2, self.temperature)

    def features(self, x):
        return self.encoder(x)


# --------------------------- shared helper ----------------------------------
def neg_cos(p, z):
    """Negative cosine similarity (mean over the batch)."""
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()


# ----------------------------- SimSiam --------------------------------------
class SimSiam(nn.Module):
    def __init__(self, proj_dim=128, use_stopgrad=True, use_predictor=True):
        super().__init__()
        self.encoder, feat = cifar_encoder()
        self.projector = Projector(feat, 512, proj_dim)
        self.predictor = Predictor(proj_dim, 512) if use_predictor else nn.Identity()
        self.use_stopgrad = use_stopgrad

    def forward(self, x1, x2):
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        p1, p2 = self.predictor(z1), self.predictor(z2)
        t1 = z1.detach() if self.use_stopgrad else z1
        t2 = z2.detach() if self.use_stopgrad else z2
        return 0.5 * neg_cos(p1, t2) + 0.5 * neg_cos(p2, t1)

    def features(self, x):
        return self.encoder(x)


# ------------------------------- BYOL ---------------------------------------
class BYOL(nn.Module):
    def __init__(self, proj_dim=128, momentum=0.99, use_predictor=True):
        super().__init__()
        self.encoder, feat = cifar_encoder()
        self.projector = Projector(feat, 512, proj_dim)
        self.predictor = Predictor(proj_dim, 512) if use_predictor else nn.Identity()
        self.momentum = momentum
        # Target network: an EMA copy of the online encoder+projector.  The target is
        # ALWAYS a stop-gradient branch -- its parameters are updated only by the EMA
        # in update_target(), never by backprop.  (This is why we do not expose a
        # --no_stopgrad ablation for BYOL: flowing gradients into requires_grad=False
        # target params would simply discard them, so it would not be a real ablation.
        # Use SimSiam --no_stopgrad if you want to *see* stop-gradient removal collapse.)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_projector = copy.deepcopy(self.projector)
        for p in list(self.target_encoder.parameters()) + list(self.target_projector.parameters()):
            p.requires_grad = False

    @torch.no_grad()
    def update_target(self):
        m = self.momentum
        for po, pt in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            pt.data.mul_(m).add_(po.data, alpha=1 - m)
        for po, pt in zip(self.projector.parameters(), self.target_projector.parameters()):
            pt.data.mul_(m).add_(po.data, alpha=1 - m)

    @torch.no_grad()
    def _target(self, x):
        return self.target_projector(self.target_encoder(x))

    def forward(self, x1, x2):
        p1 = self.predictor(self.projector(self.encoder(x1)))
        p2 = self.predictor(self.projector(self.encoder(x2)))
        t1, t2 = self._target(x1), self._target(x2)   # always stop-gradient
        return 0.5 * neg_cos(p1, t2) + 0.5 * neg_cos(p2, t1)

    def features(self, x):
        return self.encoder(x)


def build_model(args):
    if args.method == "simclr":
        return SimCLR(temperature=args.temperature)
    if args.method == "simsiam":
        return SimSiam(use_stopgrad=not args.no_stopgrad, use_predictor=not args.no_predictor)
    if args.method == "byol":
        # BYOL's target is always a stop-gradient EMA branch; only --no_predictor applies.
        if args.no_stopgrad:
            raise SystemExit(
                "--no_stopgrad is not supported for BYOL: the target is an EMA branch with "
                "requires_grad=False, so there is no gradient to remove. Use "
                "'--method simsiam --no_stopgrad' to demonstrate stop-gradient removal."
            )
        return BYOL(momentum=args.momentum, use_predictor=not args.no_predictor)
    raise ValueError(f"unknown method: {args.method}")
