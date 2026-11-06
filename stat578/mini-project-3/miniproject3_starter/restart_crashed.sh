#!/bin/bash
# Restart code + feat_targets (crashed: out of system RAM) once frozen & random finish.
cd "$(dirname "$0")"
until [ -f runs/feat_frozen/encoder.pt ] && [ -f runs/feat_random/encoder.pt ]; do sleep 30; done
python -u train.py --target code                           --epochs 100 --seed 0 --workers 2 --out runs/code         > runs/logs/code.log 2>&1 &
python -u train.py --target feature --teacher_sees targets --epochs 100 --seed 0 --workers 2 --out runs/feat_targets > runs/logs/feat_targets.log 2>&1 &
wait
