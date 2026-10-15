| run | target | pattern | ratio | no_center | probe cls (%) | probe mean (%) | final loss |
|---|---|---|---|---|---|---|---|
| mae_r0.15 | pixel | random | 0.15 | no | 50.29 | 57.74 | 0.1281 |
| mae_r0.50 | pixel | random | 0.5 | no | 60.98 | 63.44 | 0.2338 |
| mae75 | pixel | random | 0.75 | no | 60.68 | 68.19 | 0.3866 |
| mae_r0.90 | pixel | random | 0.9 | no | 63.50 | 66.49 | 0.5736 |
| mae_block | pixel | block | 0.75 | no | 65.99 | 68.52 | 0.5565 |
| mae_grid | pixel | grid | 0.75 | no | 64.76 | 65.19 | 0.3394 |
| feat75 | feature | random | 0.75 | no | 55.72 | 57.23 | 0.0651 |
| feat_nocenter | feature | random | 0.75 | yes | 58.04 | 58.65 | 0.0845 |
