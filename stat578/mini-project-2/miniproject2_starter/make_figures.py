"""Build the required Mini-Project II figures and table from runs/*.

    python make_figures.py            # reads runs/, writes figures/

fig1_ratio.png     probe accuracy and final pretraining loss vs masking ratio
fig2_patterns.png  random / block / grid at ratio 0.75: probe accuracy + recon grids
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
import matplotlib.image as mpimg
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
    pats = [("mae75", "random"), ("mae_block", "block"), ("mae_grid", "grid")]
    fig = plt.figure(figsize=(7.5, 8.6))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.5, 1, 1, 1], hspace=0.4)
    axb = fig.add_subplot(gs[0, 0])
    w = 0.35
    xs = range(len(pats))
    axb.bar([x - w / 2 for x in xs], [acc[n]["acc_mean"] for n, _ in pats], w,
            color=C_MEAN, label="mean pool")
    axb.bar([x + w / 2 for x in xs], [acc[n]["acc_cls"] for n, _ in pats], w,
            color=C_CLS, label="cls pool")
    for x, (n, _) in zip(xs, pats):
        axb.text(x - w / 2, acc[n]["acc_mean"] + 0.4, f"{acc[n]['acc_mean']:.1f}", ha="center")
        axb.text(x + w / 2, acc[n]["acc_cls"] + 0.4, f"{acc[n]['acc_cls']:.1f}", ha="center")
    lo = min(acc[n]["acc_cls"] for n, _ in pats)
    axb.set(ylim=(lo - 6, max(acc[n]["acc_mean"] for n, _ in pats) + 4),
            xticks=list(xs), xticklabels=[p for _, p in pats],
            ylabel="probe test accuracy (%)",
            title="Masking pattern at ratio 0.75 (pixel target, 100 epochs)")
    axb.legend(frameon=False, loc="upper right", ncol=2)
    for i, (n, p) in enumerate(pats):
        a = fig.add_subplot(gs[i + 1, 0])
        a.imshow(mpimg.imread(os.path.join(RUNS, n, "recon.png")))
        a.set_title(f"{p} masking: original / masked / reconstruction", fontsize=8)
        a.axis("off")
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
