#!/bin/bash
# Queue for the remaining pretraining runs. At most 2 train.py runs on the GPU at once
# (counting runs started earlier), --workers 2 to avoid running out of system RAM.
# All at --epochs 100, seed 0.
cd "$(dirname "$0")"
MAX=2
done_or_dead() {  # $1 = run name: finished (saved) or crashed
  grep -qE "saved encoder|Traceback" "runs/logs/$1.log" 2>/dev/null
}
active() {
  local n=0 r
  for r in $ACTIVE; do done_or_dead "$r" || n=$((n+1)); done
  echo $n
}
ACTIVE="feat_frozen feat_random feat_inside"   # started earlier
launch() {  # name, then train.py args
  local name=$1; shift
  while [ "$(active)" -ge $MAX ]; do sleep 30; done
  echo "$(date +%H:%M) start $name"
  python -u train.py "$@" --epochs 100 --seed 0 --workers 2 --out runs/$name > runs/logs/$name.log 2>&1 &
  ACTIVE="$ACTIVE $name"
  sleep 60
}
launch feat         --target feature
launch pix          --target pixel
launch code         --target code
launch feat_targets --target feature --teacher_sees targets
until [ -f runs/feat/encoder.pt ]; do
  grep -q Traceback runs/logs/feat.log && { echo "feat crashed; skipping feat_fixed"; wait; exit 1; }
  sleep 30
done
launch feat_fixed   --target feature --teacher fixed --teacher_ckpt runs/feat/encoder.pt
wait
echo "$(date +%H:%M) queue done"
