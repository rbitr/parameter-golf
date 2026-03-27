# Experiment: No QAT (threshold=0) + Brotli + EMA 0.998

## Hypothesis
The trend from QAT 0.20→0.15 (less QAT = better) suggested that removing QAT entirely might further improve BPB. Without fake quantization noise in the final training phase, the model could converge more cleanly.

## Changes from current best (brotli_ema098_noswa, 1.1226)
1. `late_qat_threshold`: 0.15 → 0.0 (QAT completely disabled)

## Results
- **val_bpb: 1.1233** (REGRESSED, +0.0007 BPB vs best 1.1226)
- Steps: 7000 (vs 7046 best, lost 46 steps — hardware variance, 85.72 vs 85.17 ms/step)
- Post-EMA val_bpb: 1.1395 (vs 1.1391 best)
- Quant gap: 0.0076 (vs 0.0073 best)
- Artifact: 15,465,492 bytes (similar to best)

## Analysis

| Metric | No QAT | QAT 0.15 (best) | Delta |
|--------|--------|-----------------|-------|
| post_ema bpb | 1.1395 | 1.1391 | +0.0004 |
| quant gap (roundtrip) | 0.0076 | 0.0073 | +0.0003 |
| sliding window bpb | 1.1233 | 1.1226 | +0.0007 |
| ms/step | 85.72 | 85.17 | +0.55 (hw variance) |
| steps | 7000 | 7046 | -46 |

- QAT at 0.15 helps in TWO ways: (1) slightly better post-EMA model (+0.0004), (2) smaller quantization gap (+0.0003)
- The step count difference (46) is from hardware variance (different pod speed), not QAT overhead
- QAT threshold 0.15 acts as mild regularizer during final warmdown — constraining weights to be quantization-friendly
- The trend from 0.20→0.15 was NOT "less QAT = better". Instead, 0.15 is the sweet spot: enough to narrow quant gap, not so much as to hurt convergence.

## Conclusion
- **QAT at 0.15 is confirmed optimal.** Neither increasing (0.20) nor decreasing (0.0) improves results.
- Don't revisit QAT tuning — it's settled.

## What to try next
- Larger bigram vocab (4096 or 8192) — 10L entry ablation showed -0.002 BPB from bigram=2048→10240
- MoE (Mixture of Experts) — more capacity per parameter
- Speed optimization — profiling to reduce ms/step, gaining more training steps
