# 5 GPTQ Clip Percentiles + Brotli + EMA 0.998

## Hypothesis
Fewer clip percentiles (5 vs 10) in the GPTQ-lite optimal MSE search would preserve more information (less aggressive clipping) and improve BPB, at cost of slightly larger artifact. With brotli's 533KB headroom, the larger artifact should still fit under 16MB.

## Change
Reduced GPTQ clip percentiles from 10 to 5:
- Before: [0.998, 0.999, 0.9993, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 0.999999, 1.0]
- After: [0.999, 0.9995, 0.9999, 0.99999, 1.0]

Everything else identical to current best (brotli-10, EMA 0.998, SWA disabled, late QAT threshold 0.15).

## Results
- **val_bpb: 1.1236** (REGRESSED +0.001 vs current best 1.1226)
- val_loss: 1.8971
- artifact: 15,465,009 bytes (essentially same as current best 15,467,339)
- steps: 7038 (vs 7046 for current best)
- Late QAT triggered at step 6512

## Analysis
**Hypothesis was WRONG.** The result clearly shows:

1. **More clips = better BPB, not worse.** The 10-point optimal MSE search finds genuinely better per-row clipping points. With 5 points, the search misses intermediate percentiles that happen to give lower reconstruction error for many rows.

2. **Artifact size is unchanged.** The hypothesis that fewer clips → larger artifact was wrong. The brotli compressor doesn't care about the number of clip options searched — it cares about the distribution of quantized values, which are similar regardless.

3. **The GPTQ-lite approach is already well-calibrated at 10 clips.** Adding MORE clips (15? 20?) might marginally help, but diminishing returns and extra quantization time.

## Key Learning
The 10-point GPTQ clip search is a net positive for both BPB AND compression. Don't reduce it. Could explore adding more clip points (12-15) as a very marginal experiment, but probably not worth a RunPod run.

## Next
- Revert to 10 clips (done)
- Try EMA 0.999 or speed optimizations instead
