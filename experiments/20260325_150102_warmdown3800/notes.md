# Experiment: Warmdown 3500 -> 3800

## Hypothesis
Longer warmdown gives more time for EMA/SWA weight averaging. Going from 3000->3500 improved BPB by 0.0002 in SOTA, so extending to 3800 might help further.

## Changes
- `warmdown_iters`: 3500 -> 3800
- Everything else identical to speed_cleanup_gptq10 (best run)

## Results (8xH100 SXM, seed=1337)
| Metric | Previous Best | This Run | Delta |
|--------|--------------|----------|-------|
| val_bpb | 1.1232 | **1.1234** | **+0.0002 (WORSE)** |
| val_loss | 1.8966 | 1.8967 | +0.0001 |
| steps | 6999 | 6994 | -5 |
| artifact | 15.79MB | **16.28MB** | **OVER 16MB LIMIT** |

## Analysis
**Both metrics regressed:**

1. **BPB got slightly worse** — With 3800 warmdown, ~54% of training is spent with decaying LR vs 50% at 3500. This leaves less time training at full LR, hurting final model quality. The sweet spot is around 3500.

2. **Artifact exceeded 16MB** — Surprising since architecture is identical. The different weight distribution from longer warmdown leads to different GPTQ clip selections, producing less compressible quantized weights.

3. **Fewer steps** — 6994 vs 6999 (-5), minor difference likely from initialization/compile time variance.

## Key Learnings
1. **warmdown_iters=3500 is near-optimal** — extending further hurts. The diminishing returns from 3000->3500 have turned negative at 3800.
2. **Warmdown fraction matters** — 50% warmdown (3500/7000) works, 54% (3800/7000) is too much.
3. **Artifact size is sensitive to weight distribution** — even small training changes can push over the 16MB limit.

## What to Try Next
1. **SWA frequency** (50->25) — more averaging checkpoints without changing warmdown length
2. **EMA decay tuning** (0.997->0.998 or 0.999) — different smoothing factor
3. **Vocab size optimization** — fundamentally different approach, could unlock bigger gains
4. **Speed optimization** — profile the 1.17ms/step gap to SOTA
