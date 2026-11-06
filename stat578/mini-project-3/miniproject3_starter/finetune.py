"""End-to-end fine-tuning: update the WHOLE encoder plus a linear head.

This is the other way to use a pretrained encoder, and the one that pixel targets
are known to be good at. Compare with linear_probe.py on the same checkpoint: the gap
between the two numbers is the object of study, not either number alone.

    python finetune.py --ckpt runs/pix/encoder.pt  --epochs 15
    python finetune.py --ckpt runs/feat/encoder.pt --epochs 15
    python finetune.py --scratch                  --epochs 15   # no pretraining, reference

Kept deliberately short (15 epochs, ~10 minutes on a modest GPU) so that it fits the
project budget; a longer schedule would raise every number without changing the story.
"""
import argparse
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

import data
from vit import Encoder


def ft_transform():
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(data.CIFAR_MEAN, data.CIFAR_STD),
    ])


class Classifier(nn.Module):
    def __init__(self, encoder, pool="mean"):
        super().__init__()
        self.encoder, self.pool = encoder, pool
        self.head = nn.Linear(encoder.dim, 10)

    def forward(self, x):
        t = self.encoder(x)
        f = t[:, 0] if self.pool == "cls" else t[:, 1:].mean(dim=1)
        return self.head(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--scratch", action="store_true", help="random init, no pretraining")
    p.add_argument("--pool", choices=["cls", "mean"], default="mean")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default="./data")
    args = p.parse_args()
    assert args.scratch or args.ckpt, "give --ckpt or --scratch"

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.scratch:
        enc = Encoder().to(device)
    else:
        ck = torch.load(args.ckpt, map_location=device)
        a = ck.get("args", {})
        enc = Encoder(dim=a.get("dim", 192), depth=a.get("depth", 6)).to(device)
        enc.load_state_dict(ck["encoder"])
    model = Classifier(enc, args.pool).to(device)

    train_ds = data.get_finetune_dataset(args.data, ft_transform())
    _, test_ds = data.get_eval_datasets(args.data)
    tr = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                    drop_last=True)
    te = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * len(tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / (len(tr) * 1)) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)

    for ep in range(args.epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            loss = lossf(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in te:
                correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
        acc = correct / len(test_ds)
        print(f"epoch {ep:2d}  test accuracy = {acc*100:.2f}%")
    tag = "scratch" if args.scratch else args.ckpt
    print(f"fine-tuned ({tag}, pool={args.pool}) test accuracy = {acc*100:.2f}%")


if __name__ == "__main__":
    main()
