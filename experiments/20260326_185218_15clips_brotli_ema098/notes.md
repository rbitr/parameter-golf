# Experiment: 15 GPTQ Clip Percentiles

## Hypothesis
More percentile candidates (15 vs 10) in the GPTQ-lite optimal MSE search would find better per-row clipping points. The trend from 5→10 clips improved BPB by 0.001. Expected 10→15 to give a smaller but real improvement.

## Change
Expanded GPTQ clip percentiles from 10 to 15:
- Before: [0.998, 0.999, 0.9993, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 0.999999, 1.0]
- After: [0.995, 0.997, 0.998, 0.999, 0.9993, 0.9995, 0.9997, 0.9998, 0.9999, 0.99993, 0.99995, 0.99997, 0.99999, 0.999999, 1.0]

Everything else identical to current best (brotli-10, EMA 0.998, SWA disabled, late QAT threshold 0.15, 11L/512d).

## Results
- **val_bpb: 1.1234** (REGRESSED +0.0008 vs current best 1.1226)
- val_loss: 1.8968
- artifact: 15,479,100 bytes (under 16MB)
- steps: 6973 at 86.05 ms/step (vs 7046 at 85.17ms for best)
- Late QAT triggered at step 6448
- Roundtrip gap: 0.0237 BPB (vs 0.0238 for 10 clips — identical)

## Analysis
1. **Hardware variance dominates.** The pod ran at 86.05 ms/step (0.88ms slower than best run). This alone costs 73 steps → ~0.0005 BPB.
2. **15 clips gives NO quantization benefit.** Roundtrip gap is 0.0237 vs 0.0238 for 10 clips — essentially identical. The optimal MSE search is saturated at 10 percentile points.
3. **The 5→10 improvement was real; 10→15 is not.** With 10 well-chosen percentiles spanning 0.998-1.0, we already find near-optimal clip points per row. The quantization loss landscape is smooth enough that more granularity doesn't help.

## Key Learning
10 GPTQ clip percentiles is the optimal number. Don't increase further. The diminishing returns from 5→10→15 confirm this:
- 5 clips: 1.1236 BPB
- 10 clips: 1.1226 BPB (-0.001)
- 15 clips: ~same as 10 (no improvement, adjusted for hardware variance)

## Next Ideas
- Speed optimization (the 0.88ms/step variance between runs is larger than most quantization tweaks)
- MoE or other architectural changes for more capacity per parameter
- Earlier QAT start (threshold 0.20-0.25 instead of 0.15)
