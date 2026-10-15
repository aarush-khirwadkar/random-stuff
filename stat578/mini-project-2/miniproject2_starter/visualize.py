"""Save a grid of original / masked / reconstructed images.

Only meaningful for --target pixel: a feature-target model predicts teacher
representations, not pixels, so there is nothing to look at.

Example
-------
python visualize.py --ckpt runs/mae75/encoder.pt --ratio 0.75 --out recon75.png
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import data
from methods import MaskedModel
from vit import patchify, unpatchify


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--pattern", choices=["random", "block", "grid"], default="random")
    p.add_argument("--ratio", type=float, default=0.75)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--out", default="recon.png")
    p.add_argument("--data", default="./data")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    a = ck.get("args", {})
    if a.get("target", "pixel") != "pixel":
        raise SystemExit("visualize.py only applies to --target pixel checkpoints.")

    model = MaskedModel(target="pixel", dim=a.get("dim", 192), depth=a.get("depth", 6),
                        dec_dim=a.get("dec_dim", 96),
                        dec_depth=a.get("dec_depth", 2)).to(device)
    model.encoder.load_state_dict(ck["encoder"])
    if "decoder" in ck:
        model.decoder.load_state_dict(ck["decoder"])
    else:
        raise SystemExit("this checkpoint has no decoder; re-run train.py to save one.")
    model.eval()

    _, test_ds = data.get_eval_datasets(args.data)
    imgs = next(iter(DataLoader(test_ds, batch_size=args.n, shuffle=True)))[0].to(device)

    norm_pix = not a.get("no_norm_pix", False)
    loss, pred, mask = model(imgs, pattern=args.pattern, ratio=args.ratio,
                             norm_pix=norm_pix)
    patch, grid = model.patch, model.grid

    y = patchify(imgs, patch)
    if norm_pix:                                            # undo the per-patch norm
        mu = y.mean(dim=-1, keepdim=True)
        sd = (y.var(dim=-1, keepdim=True) + 1e-6).sqrt()
        recon = pred * sd + mu
    else:
        recon = pred
    keep = (1 - mask).unsqueeze(-1)                          # 1 where visible
    pasted = y * keep + recon * (1 - keep)                   # keep visible patches as-is

    orig_img = data.denormalize(imgs)
    masked_img = data.denormalize(unpatchify(y * keep, patch, grid))
    recon_img = data.denormalize(unpatchify(pasted, patch, grid))

    fig, axes = plt.subplots(3, args.n, figsize=(1.3 * args.n, 4.4))
    for j in range(args.n):
        for i, (im, name) in enumerate([(orig_img, "original"),
                                        (masked_img, "masked"),
                                        (recon_img, "reconstruction")]):
            ax = axes[i, j]
            ax.imshow(im[j].permute(1, 2, 0).cpu().numpy())
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(name, fontsize=8)
                ax.set_title(name, fontsize=8, loc="left")
    plt.tight_layout()
    plt.savefig(args.out, dpi=140)
    print(f"masked loss = {loss.item():.4f};  saved {args.out}")


if __name__ == "__main__":
    main()
