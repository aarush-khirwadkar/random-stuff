"""Predictive latent-space pretraining on CIFAR-10.

Examples (the project's runs; all at --epochs 100)
--------------------------------------------------
# Part 1: same masking, two target spaces
python train.py --target pixel   --out runs/pix
python train.py --target feature --out runs/feat

# Part 2a: the third target space -- an online discrete code
python train.py --target code    --out runs/code

# Part 2b: who constructs the target?
python train.py --target feature --teacher frozen                             --out runs/feat_frozen
python train.py --target feature --teacher fixed --teacher_ckpt runs/feat/encoder.pt --out runs/feat_fixed

# Part 2c: does the teacher see the whole image?
python train.py --target feature --teacher_sees targets --out runs/feat_targets

# Part 2d: where is the prediction made?
python train.py --target feature --mask_in_encoder --out runs/feat_inside

# Part 2e: what is hidden?
python train.py --target feature --pattern random --out runs/feat_random

Defaults: multi-block masking at ratio 0.6 (26 of 64 patches visible), an EMA teacher that sees the full image,
an outside predictor with 2 blocks, centering on.
"""
import argparse
import csv
import math
import os
import torch
from torch.utils.data import DataLoader, Subset

import data
import metrics
from methods import build_model


@torch.no_grad()
def probe(model, imgs, grid, prev_teacher):
    """Frozen-feature diagnostics on one held-out batch of FULL images."""
    model.eval()
    cls, patches = model.features(imgs)
    erank, _ = metrics.effective_rank(cls)
    out = dict(
        cls_erank=erank,
        cls_std=metrics.output_std(cls),
        cls_patch_sim=metrics.cls_patch_similarity(cls, patches),
        patch_locality=metrics.patch_locality(patches, grid),
        target_drift=float("nan"),
        st_agreement=float("nan"),
        code_perplexity=float("nan"),
    )
    cur_teacher = None
    if model.teacher is not None:
        _, cur_teacher = model.teacher.features(imgs)
        out["target_drift"] = metrics.target_drift(prev_teacher, cur_teacher)
        out["st_agreement"] = metrics.student_teacher_agreement(patches, cur_teacher)
    if model.codebook is not None:
        out["code_perplexity"] = model.codebook.perplexity()
    model.train()
    return out, cur_teacher


COLUMNS = ["epoch", "loss", "cls_erank", "cls_std", "cls_patch_sim", "patch_locality",
           "target_drift", "st_agreement", "code_perplexity"]


def main():
    p = argparse.ArgumentParser()
    # the design axes
    p.add_argument("--target", choices=["pixel", "feature", "code"], default="feature")
    p.add_argument("--teacher", choices=["ema", "frozen", "fixed"], default="ema")
    p.add_argument("--teacher_ckpt", default=None, help="encoder.pt for --teacher fixed")
    p.add_argument("--teacher_sees", choices=["full", "targets"], default="full")
    p.add_argument("--pred_depth", type=int, default=2, help="blocks in the outside predictor")
    p.add_argument("--mask_in_encoder", action="store_true",
                   help="mask tokens inside the encoder + linear head (no outside predictor)")
    p.add_argument("--pattern", choices=["random", "block", "grid", "multiblock"],
                   default="multiblock")
    p.add_argument("--ratio", type=float, default=0.6)
    p.add_argument("--no_center", action="store_true",
                   help="ablation (feature target): remove teacher centering")
    p.add_argument("--codebook_size", type=int, default=256)
    p.add_argument("--no_norm_pix", action="store_true")
    # architecture / optimisation (held fixed across the project)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--pred_dim", type=int, default=96)
    p.add_argument("--momentum", type=float, default=0.996)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/exp")
    p.add_argument("--data", default="./data")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0,
                   help="debugging: use only the first N training images")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | target: {args.target} | teacher: {args.teacher} "
          f"sees {args.teacher_sees} | predictor: "
          f"{'inside (linear head)' if args.mask_in_encoder else f'outside, depth {args.pred_depth}'} "
          f"| mask: {args.pattern} @ {args.ratio}")

    ds = data.get_pretrain_dataset(args.data, train=True)
    if args.limit > 0:
        ds = Subset(ds, list(range(args.limit)))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True, pin_memory=True)
    probe_loader = DataLoader(ds, batch_size=min(512, len(ds)), shuffle=True,
                              num_workers=2, drop_last=True)
    probe_imgs = next(iter(probe_loader))[0].to(device)

    model = build_model(args).to(device)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))

    def lr_at(epoch):
        if epoch < args.warmup:
            return args.lr * (epoch + 1) / args.warmup
        t = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * t))

    prev_teacher = None
    with open(os.path.join(args.out, "metrics.csv"), "w", newline="") as log:
        w = csv.writer(log)
        w.writerow(COLUMNS)
        for epoch in range(args.epochs):
            for g in opt.param_groups:
                g["lr"] = lr_at(epoch)
            model.train()
            total, nb = 0.0, 0
            for imgs, _ in loader:
                imgs = imgs.to(device, non_blocking=True)
                loss, _, _, _ = model(imgs, pattern=args.pattern, ratio=args.ratio,
                                      norm_pix=not args.no_norm_pix)
                opt.zero_grad()
                loss.backward()
                opt.step()
                model.update_teacher()
                total += loss.item()
                nb += 1

            d, prev_teacher = probe(model, probe_imgs, model.grid, prev_teacher)
            w.writerow([epoch, total / nb] + [d[c] for c in COLUMNS[2:]])
            log.flush()
            if epoch % 5 == 0 or epoch == args.epochs - 1:
                print(f"epoch {epoch:3d}  loss={total/nb:.4f}  erank={d['cls_erank']:.1f}  "
                      f"std={d['cls_std']:.3f}  locality={d['patch_locality']:.3f}  "
                      f"drift={d['target_drift']:.4f}  agree={d['st_agreement']:.3f}  "
                      f"ppl={d['code_perplexity']:.1f}")

    torch.save({"encoder": model.encoder.state_dict(),
                "predictor": model.predictor.state_dict(),
                "args": vars(args)},
               os.path.join(args.out, "encoder.pt"))
    print("saved encoder to", os.path.join(args.out, "encoder.pt"))


if __name__ == "__main__":
    main()
