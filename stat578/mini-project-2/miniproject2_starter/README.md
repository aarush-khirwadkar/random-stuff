# STAT 578 — Mini-Project II starter code

Masked image modeling on CIFAR-10 with a small Vision Transformer. The two
design objects from Week 7 are exposed directly as command-line flags:

| Design object | Flag | Meaning |
|---|---|---|
| the masking distribution `pi` | `--ratio`, `--pattern` | **where** and **how much** we hide |
| the target `Y` | `--target` | **what** the model must predict |

Everything else is held fixed, so a change in a flag is a controlled experiment.

## Setup
```
pip install -r requirements.txt
```
CIFAR-10 downloads automatically on first run.

## Files
| file | what it is |
|------|------------|
| `masking.py` | the masking distribution: `random`, `block`, `grid`, at any ratio |
| `vit.py` | small ViT encoder (kept) + light decoder (discarded); patchify helpers |
| `methods.py` | `MaskedModel(target="pixel" or "feature")` — one class, two targets |
| `data.py` | CIFAR-10 with crop + flip (masked modeling needs little augmentation) |
| `metrics.py` | effective rank, output std, CLS–patch similarity, patch locality |
| `train.py` | pretraining loop; logs `metrics.csv` per epoch; saves `encoder.pt` |
| `linear_probe.py` | freeze the encoder, train a linear classifier, report test accuracy |
| `visualize.py` | save an original / masked / reconstruction image grid |

## Pretrain

The whole project is **eight runs, all at `--epochs 100`**. Keeping one budget everywhere
is what makes the comparisons fair, and it lets the two Part 1 runs double as the
reference points for both sweeps — nothing is trained twice.

Part 1 — the two headline runs (reused throughout):
```
python train.py --target pixel   --ratio 0.75 --pattern random --epochs 100 --out runs/mae75
python train.py --target feature --ratio 0.75 --pattern random --epochs 100 --out runs/feat75
```

Part 2a — the masking-ratio sweep (0.75 is `runs/mae75`, already done):
```
for R in 0.15 0.50 0.90; do
  python train.py --target pixel --ratio $R --epochs 100 --out runs/mae_r$R
done
```

Part 2a — the pattern comparison at fixed ratio (`random` is `runs/mae75`, already done):
```
python train.py --target pixel --ratio 0.75 --pattern block --epochs 100 --out runs/mae_block
python train.py --target pixel --ratio 0.75 --pattern grid  --epochs 100 --out runs/mae_grid
```

Part 2b — collapse ablation for the feature target:
```
python train.py --target feature --no_center --epochs 100 --out runs/feat_nocenter
```

Two extra ablation flags, available if you want them for Part 3:
`--no_norm_pix` (predict raw pixels instead of per-patch standardized ones) and
`--seed` (fix it, and report it).

## Evaluate
```
python linear_probe.py --ckpt runs/mae75/encoder.pt --pool mean
python linear_probe.py --ckpt runs/mae75/encoder.pt --pool cls
python visualize.py    --ckpt runs/mae75/encoder.pt --ratio 0.75 --out recon75.png
```

## Reading the diagnostics

`metrics.csv` columns: `epoch, loss, cls_erank, cls_std, cls_patch_sim, patch_locality`.

- **`loss`** is *not* comparable across masking ratios or across targets. A higher
  ratio makes each prediction harder, and the pixel and feature losses are
  different quantities entirely. Compare linear-probe accuracy instead.
- **`cls_erank`** — effective rank of the CLS features. Small means dimensional
  collapse. Scale-invariant, so it cannot see a constant representation.
- **`cls_std`** — std of L2-normalized CLS features; near 0 means complete collapse.
- **`cls_patch_sim`** — mean cosine similarity between the CLS vector and the patch
  tokens. This is the quantity DINOv3 reports *rising* over long training. It starts
  high in a small model, so watch the **trend**, not the absolute value.
- **`patch_locality`** — for each patch, the mean grid distance to its most similar
  patches, normalized to [0, 1]. Small means patch features still encode local
  content; drifting upward means patches are becoming interchangeable.

## Notes
- The **encoder never sees mask tokens** — it runs on visible patches only, which is
  why a high masking ratio makes training *faster*, not slower.
- The `--target feature` variant predicts an EMA teacher's representation, so a
  constant representation satisfies it perfectly. Watch `cls_erank` and `cls_std`.
  `--no_center` removes the anti-collapse device on purpose.
- Defaults are a reasonable starting point, not tuned optima. Absolute accuracy on
  CIFAR-10 from a short run will be modest; the trends are what the project is about.
