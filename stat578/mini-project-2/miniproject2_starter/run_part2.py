"""Run every experiment Part 2 needs (six pretraining runs + probes + recon grids).

Part 2a ratio sweep   : mae_r0.15, mae_r0.50, mae_r0.90      (pixel, random)
Part 2a pattern       : mae_block, mae_grid                  (pixel, ratio 0.75)
Part 2b target        : feat_nocenter                        (feature, --no_center)

runs/mae75 and runs/feat75 are the Part 1 reference runs and are NOT trained here.
If they exist (e.g. teammate copied them in), they are probed too so the final
table is complete.

Usage
-----
python run_part2.py                    # full 100-epoch budget
python run_part2.py --epochs 1 --runs_dir runs_smoke   # quick smoke test
python run_part2.py --only mae_block feat_nocenter     # subset

Finished runs (encoder.pt present) are skipped, so it is safe to re-run after
an interruption. Probe results go to <runs_dir>/part2_probe_results.csv.
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time

# name -> train.py flags (everything else is the starter default)
RUNS = {
    "mae_r0.15":     ["--target", "pixel", "--ratio", "0.15", "--pattern", "random"],
    "mae_r0.50":     ["--target", "pixel", "--ratio", "0.50", "--pattern", "random"],
    "mae_r0.90":     ["--target", "pixel", "--ratio", "0.90", "--pattern", "random"],
    "mae_block":     ["--target", "pixel", "--ratio", "0.75", "--pattern", "block"],
    "mae_grid":      ["--target", "pixel", "--ratio", "0.75", "--pattern", "grid"],
    "feat_nocenter": ["--target", "feature", "--ratio", "0.75", "--pattern", "random",
                      "--no_center"],
}
# Part 1 runs, only probed if already present
REFERENCE = {"mae75": ("pixel", "random", "0.75"), "feat75": ("feature", "random", "0.75")}


def run(cmd, log_path=None):
    print(">", " ".join(cmd), flush=True)
    if log_path is None:
        return subprocess.run(cmd, capture_output=True, text=True)
    with open(log_path, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", help="subset of run names to train")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    py = sys.executable
    names = args.only or list(RUNS)
    unknown = [n for n in names if n not in RUNS]
    if unknown:
        sys.exit(f"unknown run(s): {unknown}; choose from {list(RUNS)}")

    # 1. pretrain
    for name in names:
        out = os.path.join(args.runs_dir, name)
        if os.path.exists(os.path.join(out, "encoder.pt")):
            print(f"[skip] {name}: encoder.pt exists", flush=True)
            continue
        os.makedirs(out, exist_ok=True)
        t0 = time.time()
        cmd = [py, "train.py", *RUNS[name], "--epochs", str(args.epochs),
               "--seed", str(args.seed), "--workers", str(args.workers), "--out", out]
        r = run(cmd, os.path.join(out, "train.log"))
        print(f"[{'ok' if r.returncode == 0 else 'FAILED'}] {name} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        if r.returncode != 0:
            sys.exit(f"{name} failed; see {out}/train.log")

    # 2. probe (both poolings) + reconstructions for every available run
    todo = [(n, RUNS[n]) for n in RUNS]
    for n, (tgt, pat, ratio) in REFERENCE.items():
        todo.append((n, ["--target", tgt, "--pattern", pat, "--ratio", ratio]))
    results_path = os.path.join(args.runs_dir, "part2_probe_results.csv")
    rows = []
    for name, flags in todo:
        ckpt = os.path.join(args.runs_dir, name, "encoder.pt")
        if not os.path.exists(ckpt):
            continue
        row = {"run": name, "target": flags[flags.index("--target") + 1],
               "pattern": flags[flags.index("--pattern") + 1],
               "ratio": flags[flags.index("--ratio") + 1],
               "no_center": "--no_center" in flags}
        for pool in ("cls", "mean"):
            r = run([py, "linear_probe.py", "--ckpt", ckpt, "--pool", pool])
            m = re.search(r"accuracy = ([\d.]+)%", r.stdout)
            row[f"acc_{pool}"] = float(m.group(1)) if m else None
            if not m:
                print(r.stdout, r.stderr)
        rows.append(row)
        print(f"[probe] {name}: cls={row['acc_cls']}  mean={row['acc_mean']}", flush=True)

        if row["target"] == "pixel":
            img = os.path.join(args.runs_dir, name, "recon.png")
            r = run([py, "visualize.py", "--ckpt", ckpt, "--pattern", row["pattern"],
                     "--ratio", row["ratio"], "--out", img])
            print(r.stdout.strip() or r.stderr.strip(), flush=True)

    with open(results_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "target", "pattern", "ratio",
                                          "no_center", "acc_cls", "acc_mean"])
        w.writeheader()
        w.writerows(rows)
    print("wrote", results_path)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
