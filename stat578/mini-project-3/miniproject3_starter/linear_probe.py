"""Frozen-feature evaluation: train a linear classifier on the encoder's output.

The encoder is never updated. Two knobs matter for this project:

    --pool cls | mean   which token(s) the classifier reads
    --frac f            fraction of the training LABELS used (stratified), e.g. 0.01
                        = 500 labelled images. Low-label probing is where latent
                        targets tend to separate from pixel targets.

Examples
--------
python linear_probe.py --ckpt runs/feat/encoder.pt --pool mean
python linear_probe.py --ckpt runs/feat/encoder.pt --pool mean --frac 0.01
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import data
from vit import Encoder


def load_encoder(ckpt, device):
    ck = torch.load(ckpt, map_location=device)
    a = ck.get("args", {})
    enc = Encoder(dim=a.get("dim", 192), depth=a.get("depth", 6)).to(device)
    enc.load_state_dict(ck["encoder"])
    enc.eval()
    return enc, a


@torch.no_grad()
def extract(encoder, loader, device, pool):
    feats, labels = [], []
    for x, y in loader:
        cls, patches = encoder.features(x.to(device))
        f = cls if pool == "cls" else patches.mean(dim=1)
        feats.append(f.cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def stratified_subset(y, frac, seed=0):
    """Indices of a class-balanced subset holding `frac` of the labels."""
    g = torch.Generator().manual_seed(seed)
    keep = []
    for c in y.unique():
        idx = (y == c).nonzero(as_tuple=True)[0]
        n = max(1, int(round(frac * len(idx))))
        keep.append(idx[torch.randperm(len(idx), generator=g)[:n]])
    return torch.cat(keep)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--pool", choices=["cls", "mean"], default="mean")
    p.add_argument("--frac", type=float, default=1.0, help="fraction of training labels")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default="./data")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, _ = load_encoder(args.ckpt, device)

    train_ds, test_ds = data.get_eval_datasets(args.data)
    tr = DataLoader(train_ds, batch_size=512, shuffle=False, num_workers=4)
    te = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4)

    Xtr, ytr = extract(encoder, tr, device, args.pool)
    Xte, yte = extract(encoder, te, device, args.pool)
    if args.frac < 1.0:
        sel = stratified_subset(ytr, args.frac, args.seed)
        Xtr, ytr = Xtr[sel], ytr[sel]

    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Xtr, ytr, Xte, yte = Xtr.to(device), ytr.to(device), Xte.to(device), yte.to(device)

    clf = nn.Linear(Xtr.shape[1], 10).to(device)
    # more epochs when there are few labels, so every setting sees ~the same #updates
    epochs = args.epochs if args.frac >= 1.0 else int(args.epochs / max(args.frac, 0.05))
    opt = torch.optim.AdamW(clf.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss()

    n = Xtr.shape[0]
    bs = min(1024, n)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = lossf(clf(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    with torch.no_grad():
        acc = (clf(Xte).argmax(1) == yte).float().mean().item()
    print(f"pool={args.pool}  labels={args.frac:.0%} ({n})  "
          f"linear probe test accuracy = {acc*100:.2f}%")


if __name__ == "__main__":
    main()
