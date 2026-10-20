"""Build the required Mini-Project II figures and table from runs/*.

    python make_figures.py            # reads runs/, writes figures/

fig1_ratio.png     probe accuracy and final pretraining loss vs masking ratio
fig2_patterns.png  random / block / grid at ratio 0.75: probe accuracy + recon grids
                   (side by side, same 4 test images and seeded masks for every pattern)
fig3_collapse.png  loss, cls_erank, cls_std, cls_patch_sim, patch_locality vs epoch,
                   centered (feat75) vs uncentered (feat_nocenter)
table_probe.md/.csv  linear-probe accuracy over all eight runs

Uses runs/part2_probe_results.csv (written by run_part2.py) for accuracies and each
run's metrics.csv for loss / diagnostics. Probe accuracy uses --pool mean in the
main panels (one pooling choice, as the assignment asks); cls is shown alongside.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "runs"
OUT = "figures"
POOL = "mean"          # pooling used consistently in the headline numbers
C_MEAN, C_CLS = "#1f77b4", "#ff7f0e"
C_CENT, C_UNCENT = "#1f77b4", "#d62728"

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25})


def read_csv(path):
    with open(path, newline="") as f:
        return [{k: (float(v) if k not in ("run", "target", "pattern", "no_center")
                     and v not in ("", None) else v) for k, v in r.items()}
                for r in csv.DictReader(f)]


def metrics(run):
    return read_csv(os.path.join(RUNS, run, "metrics.csv"))


def recon_grid(run, pattern, imgs, ratio=0.75, seed=0):
    """original / masked / reconstruction rows for a pixel-target run, as one image.

    Same logic as visualize.py, but on a fixed set of images and a seeded mask so
    every pattern is shown on the same inputs.
    """
    import torch
    from torchvision.utils import make_grid
    import data
    from methods import MaskedModel
    from vit import patchify, unpatchify

    ck = torch.load(os.path.join(RUNS, run, "encoder.pt"), map_location="cpu")
    a = ck.get("args", {})
    m = MaskedModel(target="pixel", dim=a.get("dim", 192), depth=a.get("depth", 6),
                    dec_dim=a.get("dec_dim", 96), dec_depth=a.get("dec_depth", 2))
    m.encoder.load_state_dict(ck["encoder"])
    m.decoder.load_state_dict(ck["decoder"])
    m.eval()
    norm_pix = not a.get("no_norm_pix", False)
    torch.manual_seed(seed)
    with torch.no_grad():
        _, pred, mask = m(imgs, pattern=pattern, ratio=ratio, norm_pix=norm_pix)
        y = patchify(imgs, m.patch)
        if norm_pix:                                        # undo the per-patch norm
            mu = y.mean(dim=-1, keepdim=True)
            sd = (y.var(dim=-1, keepdim=True) + 1e-6).sqrt()
            pred = pred * sd + mu
        keep = (1 - mask).unsqueeze(-1)
        rows = [imgs, unpatchify(y * keep, m.patch, m.grid),
                unpatchify(y * keep + pred * (1 - keep), m.patch, m.grid)]
    tiles = data.denormalize(torch.cat(rows))
    return make_grid(tiles, nrow=imgs.shape[0], padding=2, pad_value=1.0).permute(1, 2, 0).numpy()


def main():
    os.makedirs(OUT, exist_ok=True)
    acc = {r["run"]: r for r in read_csv(os.path.join(RUNS, "part2_probe_results.csv"))}
    final_loss = {n: metrics(n)[-1]["loss"] for n in acc}

    # ---- Fig 1: ratio sweep -------------------------------------------------
    sweep = [("mae_r0.15", 0.15), ("mae_r0.50", 0.50), ("mae75", 0.75), ("mae_r0.90", 0.90)]
    ratios = [r for _, r in sweep]
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3))
    ax[0].plot(ratios, [acc[n]["acc_mean"] for n, _ in sweep], "o-", c=C_MEAN, label="mean pool")
    ax[0].plot(ratios, [acc[n]["acc_cls"] for n, _ in sweep], "s--", c=C_CLS, label="cls pool")
    ax[0].set(xlabel="masking ratio", ylabel="linear-probe test accuracy (%)",
              title="Probe accuracy vs ratio", xticks=ratios)
    ax[0].legend(frameon=False)
    ax[1].plot(ratios, [final_loss[n] for n, _ in sweep], "o-", c="#555555")
    ax[1].set(xlabel="masking ratio", ylabel="final pretraining loss (epoch 100)",
              title="Final loss vs ratio (not comparable)", xticks=ratios)
    fig.suptitle("Pixel target, random masking, 100 epochs", fontsize=9, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_ratio.png"), dpi=200)
    plt.close(fig)

    # ---- Fig 2: pattern comparison -----------------------------------------
    import torch
    import data
    pats = [("mae75", "random"), ("mae_block", "block"), ("mae_grid", "grid")]
    _, test_ds = data.get_eval_datasets("./data")
    idx = torch.randperm(len(test_ds), generator=torch.Generator().manual_seed(0))[:4]
    imgs = torch.stack([test_ds[int(i)][0] for i in idx])   # same 4 test images for every pattern
    fig = plt.figure(figsize=(11.5, 2.9))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.25, 0.22, 1, 1, 1], wspace=0.06)   # col 1 = gap for row labels
    axb = fig.add_subplot(gs[0, 0])
    w = 0.35
    xs = range(len(pats))
    axb.bar([x - w / 2 for x in xs], [acc[n]["acc_mean"] for n, _ in pats], w,
            color=C_MEAN, label="mean pool")
    axb.bar([x + w / 2 for x in xs], [acc[n]["acc_cls"] for n, _ in pats], w,
            color=C_CLS, label="cls pool")
    for x, (n, _) in zip(xs, pats):
        axb.text(x - w / 2, acc[n]["acc_mean"] + 0.4, f"{acc[n]['acc_mean']:.1f}", ha="center", fontsize=7.5)
        axb.text(x + w / 2, acc[n]["acc_cls"] + 0.4, f"{acc[n]['acc_cls']:.1f}", ha="center", fontsize=7.5)
    lo = min(acc[n]["acc_cls"] for n, _ in pats)
    axb.set(ylim=(lo - 6, max(acc[n]["acc_mean"] for n, _ in pats) + 4),
            xticks=list(xs), xticklabels=[p for _, p in pats],
            ylabel="probe test accuracy (%)",
            title="Probe accuracy, ratio 0.75")
    axb.legend(frameon=False, loc="upper right", ncol=2)
    for i, (n, p) in enumerate(pats):
        a = fig.add_subplot(gs[0, i + 2])
        g = recon_grid(n, p, imgs)
        a.imshow(g)
        a.set_title(f"{p} masking", fontsize=9)
        a.set_xticks([]); a.grid(False)
        for s in a.spines.values():
            s.set_visible(False)
        h = g.shape[0] / 3
        a.set_yticks([h * (k + 0.5) for k in range(3)])
        a.set_yticklabels(["original", "masked", "recon."] if i == 0 else [], fontsize=8)
        a.tick_params(left=False)
    fig.savefig(os.path.join(OUT, "fig2_patterns.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 3: collapse panel ---------------------------------------------
    cent, unc = metrics("feat75"), metrics("feat_nocenter")
    cols = [("loss", "feature loss"), ("cls_erank", "effective rank (cls)"),
            ("cls_std", "output std (cls, L2-normalised)"),
            ("cls_patch_sim", "cls-patch similarity"),
            ("patch_locality", "patch locality")]
    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5))
    for a, (k, title) in zip(axes.flat, cols):
        a.plot([r["epoch"] for r in cent], [r[k] for r in cent], c=C_CENT, label="centered (feat75)")
        a.plot([r["epoch"] for r in unc], [r[k] for r in unc], c=C_UNCENT, label="--no_center")
        a.set(title=title, xlabel="epoch")
    ax_leg = axes.flat[-1]
    ax_leg.axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    ax_leg.legend(h, l, loc="center", frameon=False)
    ax_leg.text(0.5, 0.2,
                f"probe acc ({POOL}):\ncentered {acc['feat75']['acc_mean']:.1f}%   "
                f"uncentered {acc['feat_nocenter']['acc_mean']:.1f}%",
                ha="center", transform=ax_leg.transAxes)
    fig.suptitle("Feature target: centered vs uncentered teacher", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_collapse.png"), dpi=200)
    plt.close(fig)

    # ---- Table --------------------------------------------------------------
    order = ["mae_r0.15", "mae_r0.50", "mae75", "mae_r0.90", "mae_block", "mae_grid",
             "feat75", "feat_nocenter"]
    rows = [[n, acc[n]["target"], acc[n]["pattern"], f"{acc[n]['ratio']:g}",
             "no" if acc[n]["no_center"] in (False, "False") else "yes",
             f"{acc[n]['acc_cls']:.2f}", f"{acc[n]['acc_mean']:.2f}",
             f"{final_loss[n]:.4f}"] for n in order]
    head = ["run", "target", "pattern", "ratio", "no_center", "probe cls (%)",
            "probe mean (%)", "final loss"]
    with open(os.path.join(OUT, "table_probe.csv"), "w", newline="") as f:
        csv.writer(f).writerows([head] + rows)
    with open(os.path.join(OUT, "table_probe.md"), "w") as f:
        f.write("| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n")
        for r in rows:
            f.write("| " + " | ".join(r) + " |\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
