"""Collect the accuracies in runs/eval/*.log into runs/eval/results.csv (and print it)."""
import csv
import os
import re

RUNS = ["pix", "feat", "code", "feat_frozen", "feat_fixed",
        "feat_targets", "feat_inside", "feat_random"]
EVAL = os.path.join("runs", "eval")


def acc(name, pattern):
    path = os.path.join(EVAL, name + ".log")
    if not os.path.exists(path):
        return None
    m = re.findall(pattern, open(path, encoding="utf-8", errors="ignore").read())
    return float(m[-1]) if m else None


rows = []
for r in RUNS:
    p100 = acc(f"{r}_probe100", r"linear probe test accuracy = ([\d.]+)%")
    s0 = acc(f"{r}_probe1_s0", r"linear probe test accuracy = ([\d.]+)%")
    s1 = acc(f"{r}_probe1_s1", r"linear probe test accuracy = ([\d.]+)%")
    avg = round((s0 + s1) / 2, 2) if s0 is not None and s1 is not None else None
    ft = acc(f"{r}_finetune", r"fine-tuned .* test accuracy = ([\d.]+)%")
    rows.append([r, p100, s0, s1, avg, ft])

header = ["run", "probe_100", "probe_1pct_seed0", "probe_1pct_seed1", "probe_1pct_avg", "finetune"]
with open(os.path.join(EVAL, "results.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows([[x if x is not None else "" for x in row] for row in rows])

fmt = lambda x: "  -  " if x is None else f"{x:5.2f}"
print(f"{'run':14s} probe100  1%s0   1%s1   1%avg  finetune")
for r, *v in rows:
    print(f"{r:14s} " + "  ".join(fmt(x) for x in v))
