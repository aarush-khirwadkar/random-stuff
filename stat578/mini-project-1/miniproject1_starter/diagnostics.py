"""One-shot diagnostics for a pretrained encoder: alignment, uniformity, effective
rank, and a saved singular-value spectrum plot (the collapse fingerprint).

Example
-------
python diagnostics.py --ckpt runs/simsiam/encoder.pt --out runs/simsiam/spectrum.png
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import data
import metrics
from models import cifar_encoder


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="./data")
    p.add_argument("--out", default="spectrum.png")
    p.add_argument("--batch", type=int, default=1024)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc, feat = cifar_encoder()
    enc.load_state_dict(torch.load(args.ckpt, map_location=device)["encoder"])
    enc.to(device).eval()

    ds = data.get_ssl_dataset(args.data, train=True)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)
    (x1, x2), _ = next(iter(loader))
    x1, x2 = x1.to(device), x2.to(device)
    f1, f2 = enc(x1), enc(x2)

    al = metrics.alignment(f1, f2)
    un = metrics.uniformity(f1)
    ostd = metrics.output_std(f1)
    er, sv = metrics.effective_rank(f1)
    print(f"alignment   = {al:.4f}   (lower = positives closer)")
    print(f"uniformity  = {un:.4f}   (lower = more spread)")
    print(f"output std  = {ostd:.4f}   (~{1.0/feat**0.5:.4f} healthy, ~0 => complete collapse)")
    print(f"eff. rank   = {er:.2f} / {feat}   (small => dimensional collapse)")

    sv = (sv / sv.max()).numpy()
    plt.figure(figsize=(4, 3))
    plt.plot(sv)
    plt.yscale("log")
    plt.xlabel("singular value index")
    plt.ylabel("normalized singular value")
    plt.title(f"feature spectrum (eff. rank = {er:.1f})")
    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print("saved spectrum to", args.out)


if __name__ == "__main__":
    main()
