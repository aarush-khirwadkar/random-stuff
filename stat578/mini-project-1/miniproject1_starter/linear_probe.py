"""Linear probe: freeze a pretrained encoder, train a linear classifier, report test accuracy.

Example
-------
python linear_probe.py --ckpt runs/simclr/encoder.pt
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import data
from models import cifar_encoder


@torch.no_grad()
def extract(encoder, loader, device):
    encoder.eval()
    feats, labels = [], []
    for x, y in loader:
        feats.append(encoder(x.to(device)).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--data", default="./data")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc, feat = cifar_encoder()
    enc.load_state_dict(torch.load(args.ckpt, map_location=device)["encoder"])
    enc.to(device)

    train_ds, test_ds = data.get_eval_datasets(args.data)
    trl = DataLoader(train_ds, batch_size=512, shuffle=False, num_workers=args.workers)
    tel = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=args.workers)

    Xtr, ytr = extract(enc, trl, device)
    Xte, yte = extract(enc, tel, device)

    # standardize features with train statistics
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    clf = nn.Linear(feat, 10).to(device)
    opt = torch.optim.SGD(clf.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0)
    Xtr_d, ytr_d = Xtr.to(device), ytr.to(device)

    for _ in range(args.epochs):
        clf.train()
        perm = torch.randperm(Xtr_d.size(0))
        for i in range(0, Xtr_d.size(0), 512):
            idx = perm[i:i + 512]
            loss = F.cross_entropy(clf(Xtr_d[idx]), ytr_d[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    clf.eval()
    with torch.no_grad():
        pred = clf(Xte.to(device)).argmax(1).cpu()
    acc = (pred == yte).float().mean().item()
    print(f"linear probe test accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
