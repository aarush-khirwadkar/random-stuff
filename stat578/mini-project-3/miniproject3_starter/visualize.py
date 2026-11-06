"""Two pictures worth having before you train anything.

1. The masks. For --pattern multiblock the picture has three colours: context
   (visible), targets (hidden AND scored), dropped (hidden, not scored).
       python visualize.py masks --pattern multiblock --out masks.png
       python visualize.py masks --pattern random     --out masks_random.png

2. Reconstructions, for a --target pixel checkpoint only (a feature or code target
   predicts representations, not pixels, so there is nothing to look at).
       python visualize.py recon --ckpt runs/pix/encoder.pt --out recon.png
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import data
import masking
from methods import PredictiveModel
from vit import patchify, unpatchify


def show_masks(args):
    grid = 8
    ids_keep, ids_restore, mask, loss_mask = masking.make_mask(
        args.pattern, args.n, grid, args.ratio, "cpu")
    # 0 = context (visible), 1 = dropped (hidden, unscored), 2 = target (hidden, scored)
    code = mask + loss_mask
    fig, axes = plt.subplots(1, args.n, figsize=(1.6 * args.n, 1.9))
    cmap = matplotlib.colors.ListedColormap(["#F5B48A", "#D9D9D9", "#8FBF9F"])
    for i, ax in enumerate(np.atleast_1d(axes)):
        ax.imshow(code[i].view(grid, grid), cmap=cmap, vmin=0, vmax=2)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{args.pattern} @ ratio {args.ratio}: orange = context (visible), "
                 f"grey = dropped, green = target (scored)", fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print("saved", args.out)


@torch.no_grad()
def show_recon(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    a = ck.get("args", {})
    if a.get("target", "pixel") != "pixel":
        raise SystemExit("recon applies to --target pixel checkpoints only.")
    model = PredictiveModel(target="pixel", pred_depth=a.get("pred_depth", 2),
                            mask_in_encoder=a.get("mask_in_encoder", False),
                            dim=a.get("dim", 192), depth=a.get("depth", 6),
                            pred_dim=a.get("pred_dim", 96)).to(device)
    model.encoder.load_state_dict(ck["encoder"])
    model.predictor.load_state_dict(ck["predictor"])
    model.eval()

    _, test_ds = data.get_eval_datasets(args.data)
    imgs = next(iter(DataLoader(test_ds, batch_size=args.n, shuffle=True)))[0].to(device)
    torch.manual_seed(args.seed)
    _, pred, mask, loss_mask = model(imgs, pattern=args.pattern, ratio=args.ratio,
                                     norm_pix=not a.get("no_norm_pix", False))
    patch, grid = model.patch, model.grid
    y = patchify(imgs, patch)
    if not a.get("no_norm_pix", False):     # undo per-patch standardisation
        mu, var = y.mean(-1, keepdim=True), y.var(-1, keepdim=True)
        pred = pred * (var + 1e-6).sqrt() + mu
    m = mask.unsqueeze(-1)
    masked = unpatchify(y * (1 - m), patch, grid)
    filled = unpatchify(y * (1 - m) + pred * m, patch, grid)
    rows = [data.denormalize(imgs), data.denormalize(masked), data.denormalize(filled)]
    fig, axes = plt.subplots(3, args.n, figsize=(1.4 * args.n, 4.4))
    for r, row in enumerate(rows):
        for i in range(args.n):
            axes[r, i].imshow(row[i].permute(1, 2, 0).cpu().numpy())
            axes[r, i].set_xticks([]); axes[r, i].set_yticks([])
    for r, name in enumerate(["original", "masked", "prediction"]):
        axes[r, 0].set_ylabel(name, fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print("saved", args.out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["masks", "recon"])
    p.add_argument("--ckpt", default=None)
    p.add_argument("--pattern", choices=["random", "block", "grid", "multiblock"],
                   default="multiblock")
    p.add_argument("--ratio", type=float, default=0.6)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="fig.png")
    p.add_argument("--data", default="./data")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    if args.what == "masks":
        show_masks(args)
    else:
        assert args.ckpt, "recon needs --ckpt"
        show_recon(args)


if __name__ == "__main__":
    main()
