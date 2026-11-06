# STAT 578 — Mini-Project III starter code

Predictive latent-space modeling on CIFAR-10 with a small Vision Transformer. Every
design axis from Week 10 is a command-line flag, and everything else is held fixed,
so a change in one flag is a controlled experiment:

| Question (Week 10) | Flag | Choices |
|---|---|---|
| **What is hidden?** | `--pattern`, `--ratio` | `multiblock` (I-JEPA), `random`, `block`, `grid` |
| **In what space is the target?** | `--target` | `pixel`, `feature`, `code` |
| **Who constructs the target?** | `--teacher` | `ema`, `frozen`, `fixed --teacher_ckpt …` |
| Does the teacher see the whole image? | `--teacher_sees` | `full`, `targets` |
| Where is the prediction made? | `--pred_depth k` / `--mask_in_encoder` | outside predictor with `k` blocks / mask tokens inside the encoder + linear head |
| **What prevents collapse?** | `--no_center` | remove centering (feature target) |

Defaults: `--target feature --teacher ema --teacher_sees full --pred_depth 2
--pattern multiblock --ratio 0.6`.

## Setup
```
pip install -r requirements.txt
```
CIFAR-10 downloads automatically on first run. To smoke-test the code without a
network connection, set `STAT578_FAKE_DATA=1` (random images; never for results).

## Files
| file | what it is |
|------|------------|
| `masking.py` | the masking distribution; `multiblock` returns a separate `loss_mask` (targets only) |
| `vit.py` | ViT encoder (kept), outside `Predictor` / inside `LinearHead` (discarded) |
| `methods.py` | `PredictiveModel`: three targets, three teachers, two predictor placements; `Codebook` = online tokenizer |
| `data.py` | CIFAR-10 with crop + flip |
| `metrics.py` | effective rank, output std, CLS–patch similarity, patch locality, **target drift**, **student–teacher agreement** |
| `train.py` | pretraining loop; logs `metrics.csv` per epoch; saves `encoder.pt` |
| `linear_probe.py` | frozen encoder + linear classifier; `--pool cls/mean`; `--frac 0.01` for a low-label probe |
| `finetune.py` | end-to-end fine-tuning of the whole encoder (15 epochs); `--scratch` for a no-pretraining reference |
| `visualize.py` | `masks` (context / targets / dropped) and `recon` (pixel target only) |

## Pretrain

The whole project is **eight runs, all at `--epochs 100`**, plus two short fine-tunes.
The two Part 1 runs are the reference points for every comparison in Part 2, so
nothing is trained twice.

Part 1 — same masking, two target spaces:
```
python train.py --target pixel   --out runs/pix
python train.py --target feature --out runs/feat
```

Part 2a — the third target space, an online discrete code:
```
python train.py --target code --out runs/code
```

Part 2b — who constructs the target? (`ema` is `runs/feat`, already done)
```
python train.py --target feature --teacher frozen --out runs/feat_frozen
python train.py --target feature --teacher fixed --teacher_ckpt runs/feat/encoder.pt --out runs/feat_fixed
```

Part 2c — does the teacher see the whole image? (`full` is `runs/feat`)
```
python train.py --target feature --teacher_sees targets --out runs/feat_targets
```

Part 2d — where is the prediction made? (outside, depth 2, is `runs/feat`)
```
python train.py --target feature --mask_in_encoder --out runs/feat_inside
```

Part 2e — what is hidden? (`multiblock` is `runs/feat`)
```
python train.py --target feature --pattern random --out runs/feat_random
```

Optional ablations if you want them for Part 3: `--no_center`, `--pred_depth 1` /
`--pred_depth 4`, `--codebook_size 32`, `--ratio`, `--seed` (fix it, and report it).

## Evaluate
```
python linear_probe.py --ckpt runs/feat/encoder.pt --pool mean
python linear_probe.py --ckpt runs/feat/encoder.pt --pool mean --frac 0.01
python finetune.py     --ckpt runs/feat/encoder.pt --epochs 15
python finetune.py     --scratch --epochs 15
python visualize.py masks --pattern multiblock --out masks.png
python visualize.py recon --ckpt runs/pix/encoder.pt --out recon.png
```

## Reading the diagnostics

`metrics.csv` columns: `epoch, loss, cls_erank, cls_std, cls_patch_sim, patch_locality,
target_drift, st_agreement, code_perplexity`.

- **`loss`** is *not* comparable across targets (a squared error on pixels, a
  spherical distance to a moving teacher, and a cross-entropy over codes are three
  different quantities), nor across teachers (a frozen teacher's targets never move,
  an EMA teacher's do). Compare probe / fine-tune accuracy instead.
- **`cls_erank`**, **`cls_std`**, **`cls_patch_sim`**, **`patch_locality`**: as in
  Mini-Project II (dimensional collapse, complete collapse, CLS–patch drift, locality).
- **`target_drift`** — 1 − cosine between the teacher's patch features on a fixed
  batch now and one epoch ago. Exactly 0 for `frozen` / `fixed` teachers; positive for
  `ema`. *The right-hand side of the loss moves.* Watch it decay as the EMA momentum
  and the learning rate schedule slow the student down.
- **`st_agreement`** — cosine between the student's and the teacher's patch features
  on the same full images. Rises toward 1 for an EMA teacher as the two become
  coupled; stays low for a frozen random teacher. This is the quantity behind the
  phrase "the target is self-generated".
- **`code_perplexity`** (code target only) — exp(entropy) of codebook usage. `K`
  means all codes used equally; a value near 1 means the tokenizer collapsed to one
  code and the prediction task became trivial.

## Notes
- **Multi-block masking scores the loss only on the target blocks.** `mask` says what
  the encoder does not see; `loss_mask` says where a prediction is graded. They
  coincide for `random` / `block` / `grid`. Run `visualize.py masks` first.
- **The teacher is a copy of the encoder that never receives gradients.** With
  `ema` it trails the student; with `frozen` it is the random initialisation for the
  whole run; with `fixed` it is a checkpoint you supply (your own Part 1 run, say).
- **`--teacher_sees targets`** feeds the teacher only the hidden patches, so its
  targets are not contextualised by the visible image. This reproduces I-JEPA's
  "mask the output, not the input" ablation on a small scale.
- **`--mask_in_encoder`** puts a mask token at every hidden position inside the
  encoder and reads predictions off a linear head; the encoder then sees, during
  pretraining, a symbol it never sees at evaluation time.
- Defaults are a reasonable starting point, not tuned optima. Absolute accuracy from
  a 100-epoch run of a tiny ViT will be modest; the *differences between controlled
  runs* are the object of study.
