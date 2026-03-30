# Experiment: Muon Weight Decay 0.02 (halved from 0.04)

## Hypothesis
With ~7000 training steps and zero overfitting (train_loss ≈ val_loss), reducing Muon weight decay from 0.04 to 0.02 allows the model to learn more aggressively, improving BPB.

## Change
Single config change: `muon_wd = 0.02` (was 0.04)

## Results
| Metric | Value | vs Best (1.1207 base) |
|--------|-------|-----------------------|
| val_bpb (sliding window) | **1.1186** | **-0.0021 (HUGE WIN)** |
| val_loss | 1.8887 | -0.0036 |
| artifact_size | 17,575,911 bytes | **+2.02MB OVER 16MB** |
| steps | 6977 | +37 |
| ms/step | ~86 | same |

## Analysis
- **BPB: Massive improvement.** -0.0021 BPB is the biggest single-change improvement we've found, matching LeakyReLU(0.5)² in magnitude.
- **Size: Critical failure.** 17.58MB is 1.58MB over the 16MB limit. Lower WD → larger weights → more entropy in int6 values → worse compression ratio.
- The model was not overfitting at WD=0.04 (train/val loss nearly equal), confirming that 0.04 was over-regularized for our step budget.
- With TTT delta of -0.0012, projected BPB would be ~1.1174 — well below SOTA's 1.1194!

## Why artifact is larger
- Lower WD means weights grow larger during training
- EMA 0.998 accumulates these larger weights
- Int6 quantization scales increase → quantized values have more entropy
- Brotli-10 achieves worse compression ratio on higher-entropy data

## Next steps
1. **Try muon_wd=0.03** — compromise between 0.02 (great BPB, too big) and 0.04 (fits, worse BPB)
2. **Try muon_wd=0.02 + EMA 0.997** — tighter EMA might regularize weight magnitudes enough
3. **Try muon_wd=0.025** — fine-grained tuning
4. If 0.03 is still over 16MB, try 0.035
5. The BPB/size tradeoff suggests optimal WD is between 0.02 and 0.04 — need to find the sweet spot that maximizes BPB while staying under 16MB

## Key insight
Weight decay has been a major unexplored axis. The default 0.04 was over-regularizing our model by ~0.002 BPB. Even if we can't use the full WD=0.02 reduction, finding the right WD could be worth -0.001+ BPB.
