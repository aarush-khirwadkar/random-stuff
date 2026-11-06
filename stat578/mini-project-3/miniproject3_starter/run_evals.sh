#!/bin/bash
# Part 2 evaluations, one at a time on the GPU. Same evaluation for every run:
#   linear probe, mean pooling, 100% labels (seed 0)
#   linear probe, mean pooling, 1% labels, averaged over label subsets seed 0 and 1
#   fine-tune (15 epochs, mean pooling) for the code target only
#   mask pictures for multiblock and random
# Logs go to runs/eval/; summarize_evals.py collects them into runs/eval/results.csv.
cd "$(dirname "$0")"
mkdir -p runs/eval
RUNS="pix feat code feat_frozen feat_fixed feat_targets feat_inside feat_random"

# wait for the last pretraining run
until grep -qE "saved encoder|Traceback" runs/logs/feat_fixed.log 2>/dev/null; do sleep 30; done

python visualize.py masks --pattern multiblock --out runs/eval/masks_multiblock.png > runs/eval/masks.log 2>&1
python visualize.py masks --pattern random     --out runs/eval/masks_random.png    >> runs/eval/masks.log 2>&1

for r in $RUNS; do
  [ -f runs/$r/encoder.pt ] || { echo "$(date +%H:%M) skip $r (no encoder.pt)"; continue; }
  echo "$(date +%H:%M) probes $r"
  python linear_probe.py --ckpt runs/$r/encoder.pt --pool mean                       > runs/eval/${r}_probe100.log 2>&1
  python linear_probe.py --ckpt runs/$r/encoder.pt --pool mean --frac 0.01 --seed 0  > runs/eval/${r}_probe1_s0.log 2>&1
  python linear_probe.py --ckpt runs/$r/encoder.pt --pool mean --frac 0.01 --seed 1  > runs/eval/${r}_probe1_s1.log 2>&1
done

echo "$(date +%H:%M) finetune code"
python finetune.py --ckpt runs/code/encoder.pt --epochs 15 --pool mean > runs/eval/code_finetune.log 2>&1

python summarize_evals.py
echo "$(date +%H:%M) evals done"
