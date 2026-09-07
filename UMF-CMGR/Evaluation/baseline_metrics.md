# UMFusion RoadScene baseline metrics

Test images are matched by filename stem.

## Registration (pred IR vs aligned GT `ir_121`)

| Setting | N | RMSE ↓ | NCC ↑ | LNCC ↑ | MSE ↓ |
|---------|---|--------|-------|--------|-------|
| reg_only | 122 | 0.0774 | 0.9325 | 0.4544 | 0.0065 |
| reg_joint | 122 | 0.0774 | 0.9325 | 0.4544 | 0.0065 |

## Fusion (fused vs IR/VI)

| Setting | N | EN ↑ | SD ↑ | SF ↑ | AG ↑ | MI ↑ | SSIM ↑ | CC ↑ | SCD ↑ | Qabf ↑ |
|---------|---|------|------|------|------|------|--------|------|-------|--------|
| fusion_only | 122 | 7.0055 | 35.9832 | 8.8143 | 26.4581 | 3.0266 | 0.7642 | 0.6686 | 1.4464 | 0.4566 |
| fusion_joint | 122 | 7.0362 | 36.8956 | 8.7773 | 26.1619 | 3.0090 | 0.7635 | 0.6666 | 1.4725 | 0.4468 |

Notes:
- `reg_only`: DeformableNet trained 800 epochs, tested on warped IR (`ir_w`).
- `fusion_only`: FusionNet trained 200 epochs on registered IR (`ir_reg`).
- `*_joint`: jointly trained 600 epochs after loading the two pretrained nets.
- `fusion_joint` uses registered IR (`ir_reg`) rather than the warped input IR.
- Higher is better except RMSE/MSE.

