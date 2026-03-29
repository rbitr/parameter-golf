# Experiment: Grouped Int6 Quantization (G=128)

## Hypothesis
Finer-grained quantization scales (128 weights per scale instead of entire row) would reduce quantization error, especially for wide MLP matrices where a single per-row scale must span diverse weight magnitudes.

## Changes
- Added `quantize_int6_grouped()` function with group_size=128
- Each group of 128 consecutive elements in a row gets its own fp16 scale
- Optimal MSE search over 5 clip percentiles per group
- Modified `mixed_quantize_int6` and `dequantize_mixed_int6` to handle 2D scales

## Results
| Metric | Grouped (G=128) | Previous Best (per-row) | Delta |
|--------|-----------------|------------------------|-------|
| Sliding window BPB | 1.1203 | 1.1204* | -0.0001 |
| TTT BPB | 1.1200 | 1.1198 | +0.0002 |
| Artifact size | 15,564,643 | 15,561,305 | +3,338 |
| Steps | 6964 | ~6976 | -12 |

*Previous best sliding window from leaky_relu_05_squared run (1.1207), TTT from ttt_2ep_lr0005 (1.1198)

## Analysis
- Grouped quantization gave **negligible improvement** (-0.0001 BPB on sliding window)
- Artifact is 3KB larger due to extra scale parameters
- The GPTQ-lite optimal MSE search (10 percentiles per row) already finds good per-row scales
- Within-row weight variance is low because Muon optimizer produces approximately orthogonal weight matrices
- **Conclusion: Per-row quantization is already near-optimal for our model. Group quantization adds complexity for no benefit.**

## What to try next
- The quantization pipeline is well-optimized. Focus on base model improvements instead.
- Parameter Banking for more training steps remains the highest-impact untried idea.
