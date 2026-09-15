# STAT 578 — Mini-Project I starter code

Contrastive (SimCLR) and non-contrastive (BYOL, SimSiam) self-supervised learning
on CIFAR-10, from scratch, with the diagnostics you need for the report:
**alignment, uniformity, effective rank**, plus a **linear probe**.

Everything runs on a single modest GPU. A full run is ~100 epochs and takes roughly
**2–4 hours depending on the GPU** (faster on an A100, slower on a shared RTX 2080/3060);
use fewer epochs while debugging.

## Setup
```
pip install -r requirements.txt
```
CIFAR-10 downloads automatically on first run.

## Files
| file | what it is |
|------|------------|
| `data.py` | CIFAR-10 with a two-view SSL augmentation (crop + color jitter + blur/gray) |
| `models.py` | CIFAR ResNet-18 encoder (512-d), projector `g`, predictor `q` |
| `methods.py` | `SimCLR`, `SimSiam`, `BYOL`, each `forward(x1,x2) -> loss`; ablation flags |
| `metrics.py` | `alignment`, `uniformity`, `effective_rank` |
| `train.py` | pretraining loop; logs `metrics.csv` per epoch; saves `encoder.pt` |
| `linear_probe.py` | freeze encoder, train a linear classifier, report test accuracy |
| `diagnostics.py` | alignment/uniformity/effective rank + singular-value spectrum plot |

## Pretrain
```
python train.py --method simclr  --temperature 0.5 --out runs/simclr
python train.py --method byol    --momentum 0.99  --out runs/byol
python train.py --method simsiam                  --out runs/simsiam
```

## Collapse ablations (Part 2)
```
python train.py --method simsiam --no_stopgrad  --out runs/simsiam_nosg   # remove stop-gradient
python train.py --method simsiam --no_predictor --out runs/simsiam_nopred # remove predictor
```
Watch `output_std` in `metrics.csv` fall toward **0** as the loss drops to its floor —
that is **complete collapse**. (`effective_rank` is scale-invariant and instead flags
**dimensional** collapse; report both.)

**`--no_stopgrad` is SimSiam-only.** SimSiam's target is the online projection with an
explicit `.detach()`, so removing it is a real ablation. BYOL's target is a separate EMA
network with `requires_grad=False`; gradients through it are discarded either way, so
there is nothing to turn off — BYOL rejects `--no_stopgrad` with a helpful message. Use
SimSiam to demonstrate stop-gradient-removal collapse.

## Sweep the contrastive knobs (Part 2)
```
for T in 0.1 0.2 0.5 1.0; do python train.py --method simclr --temperature $T --out runs/simclr_T$T; done
python train.py --method simclr --batch_size 128 --out runs/simclr_b128
python train.py --method simclr --batch_size 512 --out runs/simclr_b512
```

## Evaluate
```
python linear_probe.py --ckpt runs/simclr/encoder.pt
python diagnostics.py  --ckpt runs/simsiam_nosg/encoder.pt --out runs/simsiam_nosg/spectrum.png
```

## Notes
- `metrics.csv` columns: `epoch, loss, alignment, uniformity, output_std, effective_rank`
  — plot these over training for your figures. `output_std` ~ 0 means complete collapse;
  a small `effective_rank` means dimensional collapse.
- Alignment/uniformity are reported on the **projection** space `z` (where the loss lives);
  effective rank is on the **encoder features** `h` (what you keep). Try both if curious.
- Defaults are a reasonable starting point, not tuned optima — part of the project is to explore.
