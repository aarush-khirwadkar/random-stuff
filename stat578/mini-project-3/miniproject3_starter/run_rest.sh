#!/bin/bash
# Last two pretraining runs. code / feat_targets were restarted by restart_crashed.sh; feat by run_queue.sh.
# pix starts when feat_inside finishes (keeps 4 runs max on the GPU); feat_fixed needs runs/feat/encoder.pt.
cd "$(dirname "$0")"
ended() { grep -qE "saved encoder|Traceback" "runs/logs/$1.log" 2>/dev/null; }
until ended feat_inside; do sleep 30; done
echo "$(date +%H:%M) start pix"
python -u train.py --target pixel --epochs 100 --seed 0 --workers 2 --out runs/pix > runs/logs/pix.log 2>&1 &
until [ -f runs/feat/encoder.pt ]; do
  grep -q Traceback runs/logs/feat.log && { echo "feat crashed; not starting feat_fixed"; wait; exit 1; }
  sleep 30
done
echo "$(date +%H:%M) start feat_fixed"
python -u train.py --target feature --teacher fixed --teacher_ckpt runs/feat/encoder.pt --epochs 100 --seed 0 --workers 2 --out runs/feat_fixed > runs/logs/feat_fixed.log 2>&1 &
wait
echo "$(date +%H:%M) done"
