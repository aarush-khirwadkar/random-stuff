"""Make every Part 2 report figure from runs/*/metrics.csv and runs/eval/results.csv.

    python make_figures.py              # writes ../report/figures/*.png

Fine-tuning numbers that do not exist yet (Part 1: pix, feat, scratch) are drawn as
"n/a" and appear automatically once runs/eval/<run>_finetune.log is present.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import masking

OUT = os.path.join("..", "report", "figures")
os.makedirs(OUT, exist_ok=True)

# validated categorical slots 1-3 (blue, orange, aqua); identity also carried by
# marker / line style and a legend
C = ["#2a78d6", "#eb6834", "#1baf7a"]
LS = ["-", "--", ":"]
MK = ["o", "s", "^"]
INK, MUTED, GRID = "#222222", "#666666", "#e3e3e3"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.4, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
W = 6.5  # \textwidth in inches


def load(run):
    rows = list(csv.DictReader(open(os.path.join("runs", run, "metrics.csv"))))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def results():
    out = {}
    for r in csv.DictReader(open(os.path.join("runs", "eval", "results.csv"))):
        out[r["run"]] = {k: (float(v) if v not in ("", None) else None)
                         for k, v in r.items() if k != "run"}
    return out


RES = results()
M = {r: load(r) for r in RES}


def curves(ax, runs, labels, key, ylabel, logy=False, markevery=10):
    for i, (r, lab) in enumerate(zip(runs, labels)):
        y = M[r][key]
        ax.plot(M[r]["epoch"], y, color=C[i], ls=LS[i], marker=MK[i], ms=3.5,
                markevery=markevery, label=lab)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")


def bars(ax, runs, labels, cols=("probe_100", "probe_1pct_avg"),
         col_names=("probe\n100% labels", "probe\n1% labels")):
    n, k = len(runs), len(cols)
    w = 0.8 / n
    x = np.arange(k)
    for i, (r, lab) in enumerate(zip(runs, labels)):
        vals = [RES[r][c] for c in cols]
        xs = x - 0.4 + w * (i + 0.5)
        ax.bar(xs, [v if v is not None else 0 for v in vals], w, color=C[i],
               edgecolor="white", linewidth=1.0, label=lab)
        for xi, v in zip(xs, vals):
            ax.text(xi, (v if v is not None else 0) + 1, "n/a" if v is None else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=6, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(col_names)
    ax.set_ylabel("test accuracy (%)")
    ax.grid(axis="x", visible=False)


def fig_legend(fig, ax, ncol):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncol, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.07, 1, 1))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=300)
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


# ---- Figure 1: target space ---------------------------------------------------------
fig, (a, b) = plt.subplots(1, 2, figsize=(W, 1.75), gridspec_kw={"width_ratios": [1.5, 1]})
bars(a, ["pix", "code", "feat"], ["pixel", "code", "feature"],
     cols=("probe_100", "probe_1pct_avg", "finetune"),
     col_names=("probe\n100% labels", "probe\n1% labels", "fine-tune\n100% labels"))
a.set_ylim(0, 95)
a.set_title("(a) evaluation by target space")
cp = M["code"]["code_perplexity"]
b.plot(M["code"]["epoch"], cp, color=C[1], ls="--", marker="s", ms=3.5, markevery=10,
       label="code perplexity")
b.axhline(256, color=MUTED, lw=0.8, ls=":")
b.text(99, 256, "$K=256$", ha="right", va="bottom", fontsize=6.5, color=MUTED)
b.set_ylim(0, 280)
b.set_xlabel("epoch")
b.set_ylabel("perplexity")
b.set_title("(b) code-usage perplexity")
fig_legend(fig, a, 3)
save(fig, "fig_targets.png")

# ---- Figure 2: teacher panel -----------------------------------------------------------
runs, labs = ["feat", "feat_fixed", "feat_frozen"], ["EMA", "fixed", "frozen"]
fig, (a, b, c) = plt.subplots(1, 3, figsize=(W, 1.65))
bars(a, runs, labs)
a.set_ylim(0, 85)
a.set_title("(a) final linear probe")
curves(b, runs, labs, "target_drift", "target drift", logy=True)
b.set_ylim(1e-9, 0.3)
b.set_title("(b) target drift")
b.text(50, 1e-6,"fixed, frozen: 0 up to float precision", ha="center", fontsize=6, color=MUTED)
curves(c, runs, labs, "st_agreement", "student-teacher agreement")
c.set_ylim(0, 1.05)
c.set_title("(c) student-teacher agreement", x=0.45)
c.set_ylabel("agreement (cosine)")
fig_legend(fig, b, 3)
save(fig, "fig_teacher.png")

# ---- Figure 3: what the teacher sees -----------------------------------------------------
runs, labs = ["feat", "feat_targets"], ["teacher sees full image", "teacher sees targets only"]
fig, (a, b, c) = plt.subplots(1, 3, figsize=(W, 1.65))
bars(a, runs, labs)
a.set_ylim(0, 85)
a.set_title("(a) final linear probe")
curves(b, runs, labs, "patch_locality", "patch locality")
b.set_title("(b) patch locality (lower = more local)")
curves(c, runs, labs, "cls_erank", "effective rank (CLS)")
c.set_title("(c) effective rank")
fig_legend(fig, b, 2)
save(fig, "fig_context.png")

# ---- Figure 4: where the prediction is made -------------------------------------------------
runs, labs = ["feat", "feat_inside"], ["outside predictor", "mask tokens inside encoder"]
fig, (a, b, c) = plt.subplots(1, 3, figsize=(W, 1.65))
bars(a, runs, labs)
a.set_ylim(0, 85)
a.set_title("(a) final linear probe")
curves(b, runs, labs, "cls_patch_sim", "CLS-patch similarity")
b.set_ylim(0, 1)
b.set_title("(b) CLS-patch similarity")
curves(c, runs, labs, "patch_locality", "patch locality")
c.set_title("(c) patch locality")
fig_legend(fig, b, 2)
save(fig, "fig_predictor.png")

# ---- Figure 5: masks ---------------------------------------------------------------------
torch.manual_seed(0)
cmap = matplotlib.colors.ListedColormap(["#cde2fb", "#d9d9d9", "#eb6834"])
n = 6
fig, axes = plt.subplots(2, n, figsize=(W, 1.7))
for row, (pat, run) in enumerate([("multiblock", "feat"), ("random", "feat_random")]):
    _, _, mask, loss_mask = masking.make_mask(pat, n, 8, 0.6, "cpu")
    code = (mask + loss_mask).view(n, 8, 8)       # 0 context, 1 dropped, 2 target
    for j in range(n):
        ax = axes[row, j]
        ax.imshow(code[j], cmap=cmap, vmin=0, vmax=2)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(MUTED)
    r = RES[run]
    axes[row, 0].set_ylabel(f"{pat}\n\nprobe accuracy\n{r['probe_100']:.1f}% (100% labels)\n"
                            f"{r['probe_1pct_avg']:.1f}% (1% labels)",
                            fontsize=7, rotation=0, ha="right", va="center", labelpad=6)
handles = [matplotlib.patches.Patch(color=cmap(i), label=l)
           for i, l in enumerate(["context (visible)", "dropped (hidden, not scored)",
                                  "target (hidden, scored)"])]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7,
           bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.05, 1, 1))
save(fig, "fig_masks.png")
