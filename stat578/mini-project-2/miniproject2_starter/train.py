"""Masked-image-modeling pretraining on CIFAR-10.

Examples
--------
python train.py --target pixel   --ratio 0.75 --pattern random --out runs/mae75
python train.py --target pixel   --ratio 0.15 --pattern random --out runs/mae15
python train.py --target pixel   --ratio 0.75 --pattern block  --out runs/mae75_block
python train.py --target feature --ratio 0.75 --pattern random --out runs/feat75
python train.py --target feature --no_center --out runs/feat_nocenter   # expect collapse

The two design objects under study are exposed as flags:
    --ratio / --pattern    the masking distribution pi
    --target               what the model must predict, Y
"""
import argparse
import csv
import os
import torch
from torch.utils.data import DataLoader

import data
import metrics
from methods import build_model


@torch.no_grad()
def probe(model, imgs, grid):
    """Frozen-feature diagnostics on one held-out batch."""
    model.eval()
    cls, patches = model.features(imgs)
    erank, _ = metrics.effective_rank(cls)
    out = dict(
        cls_erank=erank,
        cls_std=metrics.output_std(cls),
        cls_patch_sim=metrics.cls_patch_similarity(cls, patches),
        patch_locality=metrics.patch_locality(patches, grid),
    )
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["pixel", "feature"], default="pixel")
    p.add_argument("--pattern", choices=["random", "block", "grid"], default="random")
    p.add_argument("--ratio", type=float, default=0.75)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--dec_dim", type=int, default=96)
    p.add_argument("--dec_depth", type=int, default=2)
    p.add_argument("--momentum", type=float, default=0.996)
    p.add_argument("--no_center", action="store_true",
                   help="ablation (feature target only): remove teacher centering")
    p.add_argument("--no_norm_pix", action="store_true",
                   help="ablation (pixel target only): predict raw pixels instead of "
                        "per-patch standardised ones")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/exp")
    p.add_argument("--data", default="./data")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| target:", args.target,
          "| pattern:", args.pattern, "| ratio:", args.ratio)

    ds = data.get_pretrain_dataset(args.data, train=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True, pin_memory=True)
    probe_loader = DataLoader(ds, batch_size=512, shuffle=True,
                              num_workers=2, drop_last=True)
    probe_imgs = next(iter(probe_loader))[0].to(device)

    model = build_model(args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd,
                            betas=(0.9, 0.95))

    def lr_at(epoch):
        """Linear warmup, then cosine decay -- the usual ViT schedule."""
        import math
        if epoch < args.warmup:
            return args.lr * (epoch + 1) / args.warmup
        t = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * t))

    with open(os.path.join(args.out, "metrics.csv"), "w", newline="") as log:
        w = csv.writer(log)
        w.writerow(["epoch", "loss", "cls_erank", "cls_std",
                    "cls_patch_sim", "patch_locality"])
        for epoch in range(args.epochs):
            for g in opt.param_groups:
                g["lr"] = lr_at(epoch)
            model.train()
            total, nb = 0.0, 0
            for imgs, _ in loader:
                imgs = imgs.to(device, non_blocking=True)
                loss, _, _ = model(imgs, pattern=args.pattern, ratio=args.ratio,
                                   norm_pix=not args.no_norm_pix)
                opt.zero_grad()
                loss.backward()
                opt.step()
                model.update_teacher()
                total += loss.item()
                nb += 1

            d = probe(model, probe_imgs, model.grid)
            w.writerow([epoch, total / nb, d["cls_erank"], d["cls_std"],
                        d["cls_patch_sim"], d["patch_locality"]])
            log.flush()
            if epoch % 5 == 0 or epoch == args.epochs - 1:
                print(f"epoch {epoch:3d}  loss={total/nb:.4f}  "
                      f"cls_erank={d['cls_erank']:.1f}  "
                      f"cls_patch_sim={d['cls_patch_sim']:.3f}  "
                      f"locality={d['patch_locality']:.3f}")

    torch.save({"encoder": model.encoder.state_dict(),
                "decoder": model.decoder.state_dict(),
                "args": vars(args)},
               os.path.join(args.out, "encoder.pt"))
    print("saved encoder to", os.path.join(args.out, "encoder.pt"))


if __name__ == "__main__":
    main()
