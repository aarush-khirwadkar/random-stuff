"""Self-supervised pretraining on CIFAR-10, logging diagnostics every epoch.

Examples
--------
python train.py --method simclr  --temperature 0.5 --out runs/simclr
python train.py --method simsiam --out runs/simsiam
python train.py --method simsiam --no_stopgrad --out runs/simsiam_nosg   # expect collapse
python train.py --method byol    --momentum 0.99 --out runs/byol

Note: --no_stopgrad applies to SimSiam only. BYOL's target is an EMA branch with
requires_grad=False, so there is no gradient to remove (see methods.py).
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
def probe(model, loader, device):
    """Compute alignment/uniformity (projection space) and effective rank (feature space)."""
    model.eval()
    (x1, x2), _ = next(iter(loader))
    x1, x2 = x1.to(device), x2.to(device)
    f1 = model.features(x1)
    z1 = model.projector(f1)
    z2 = model.projector(model.features(x2))
    erank, _ = metrics.effective_rank(f1)
    model.train()
    return (metrics.alignment(z1, z2), metrics.uniformity(z1),
            metrics.output_std(z1), erank)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["simclr", "simsiam", "byol"], required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--momentum", type=float, default=0.99)
    p.add_argument("--no_stopgrad", action="store_true",
                   help="ablation (SimSiam only): let gradients flow through the target branch")
    p.add_argument("--no_predictor", action="store_true", help="ablation: identity predictor")
    p.add_argument("--out", default="runs/exp")
    p.add_argument("--data", default="./data")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    ds = data.get_ssl_dataset(args.data, train=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True, pin_memory=True)
    probe_loader = DataLoader(ds, batch_size=512, shuffle=True,
                              num_workers=2, drop_last=True)

    model = build_model(args).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    with open(os.path.join(args.out, "metrics.csv"), "w", newline="") as log:
        writer = csv.writer(log)
        writer.writerow(["epoch", "loss", "alignment", "uniformity",
                         "output_std", "effective_rank"])
        for epoch in range(args.epochs):
            model.train()
            total, nb = 0.0, 0
            for (x1, x2), _ in loader:
                x1, x2 = x1.to(device), x2.to(device)
                loss = model(x1, x2)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if args.method == "byol":
                    model.update_target()
                total += loss.item()
                nb += 1
            sched.step()
            al, un, ostd, er = probe(model, probe_loader, device)
            writer.writerow([epoch, total / nb, al, un, ostd, er])
            log.flush()
            print(f"epoch {epoch:3d}  loss={total/nb:.4f}  align={al:.4f}  "
                  f"unif={un:.4f}  out_std={ostd:.4f}  erank={er:.2f}")

    torch.save({"encoder": model.encoder.state_dict()},
               os.path.join(args.out, "encoder.pt"))
    print("saved encoder to", os.path.join(args.out, "encoder.pt"))


if __name__ == "__main__":
    main()
