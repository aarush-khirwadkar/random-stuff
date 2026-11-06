#!/bin/bash
# Part 2 runs that don't need the Part 1 checkpoints. All at --epochs 100, seed 0.
cd "$(dirname "$0")"
mkdir -p runs/logs
python -u train.py --target code                           --epochs 100 --seed 0 --out runs/code         > runs/logs/code.log 2>&1 &
python -u train.py --target feature --teacher frozen       --epochs 100 --seed 0 --out runs/feat_frozen  > runs/logs/feat_frozen.log 2>&1 &
python -u train.py --target feature --teacher_sees targets --epochs 100 --seed 0 --out runs/feat_targets > runs/logs/feat_targets.log 2>&1 &
python -u train.py --target feature --mask_in_encoder      --epochs 100 --seed 0 --out runs/feat_inside  > runs/logs/feat_inside.log 2>&1 &
python -u train.py --target feature --pattern random       --epochs 100 --seed 0 --out runs/feat_random  > runs/logs/feat_random.log 2>&1 &
wait
