"""Frozen-feature evaluation: train a linear classifier on the encoder's output.

The encoder is never updated. Two pooling choices are offered because masked
models often carry more in the patch tokens than in the CLS vector:

    --pool cls    use the CLS vector only
    --pool mean   average the patch tokens (usually stronger for MAE-style models)
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import data
from vit import Encoder


@torch.no_grad()
def extract(encoder, loader, device, pool):
    feats, labels = [], []
    for x, y in loader:
        cls, patches = encoder.features(x.to(device))
        f = cls if pool == "cls" else patches.mean(dim=1)
        feats.append(f.cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--pool", choices=["cls", "mean"], default="mean")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data", default="./data")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    a = ck.get("args", {})
    encoder = Encoder(dim=a.get("dim", 192), depth=a.get("depth", 6)).to(device)
    encoder.load_state_dict(ck["encoder"])
    encoder.eval()

    train_ds, test_ds = data.get_eval_datasets(args.data)
    tr = DataLoader(train_ds, batch_size=512, shuffle=False, num_workers=4)
    te = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4)

    Xtr, ytr = extract(encoder, tr, device, args.pool)
    Xte, yte = extract(encoder, te, device, args.pool)

    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Xtr, ytr, Xte, yte = Xtr.to(device), ytr.to(device), Xte.to(device), yte.to(device)

    clf = nn.Linear(Xtr.shape[1], 10).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss()

    n = Xtr.shape[0]
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 1024):
            idx = perm[i:i + 1024]
            loss = lossf(clf(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    with torch.no_grad():
        acc = (clf(Xte).argmax(1) == yte).float().mean().item()
    print(f"pool={args.pool}  linear probe test accuracy = {acc*100:.2f}%")


if __name__ == "__main__":
    main()
