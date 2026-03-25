# Experiment: EMA 0.998 + 5 GPTQ Clips

## Hypothesis
Reverting to 5 GPTQ clips (from 10) while keeping EMA 0.998 should produce a smaller artifact that fits under 16MB, while retaining some BPB benefit from the wider EMA window.

## Changes
- `ema_decay`: 0.998 (kept from previous)
- GPTQ clips: 10 → 5 (reverted to SOTA's clip set)
- SWA disabled

## Results (8xH100 SXM, seed=1337)
| Metric | Previous Best | This Run | Delta |
|--------|--------------|----------|-------|
| val_bpb | 1.1232 | 1.12323 | +0.00003 (essentially same) |
| artifact | 15.79MB | 15.92MB | +130KB but under 16MB |
| steps | 6999 | 7000 | +1 (same) |

## Analysis
- BPB is the same as our best (1.1232). EMA 0.998 alone doesn't improve BPB — it was the 10 GPTQ clips providing the improvement.
- Artifact is 15.92MB, under 16MB. The 5 clips produce more compressible weights.
- The 10-clip improvement (~0.0003 BPB) comes from better per-row quantization, not from EMA.

## Key Insight
10 GPTQ clips and EMA 0.998 are both needed for the BPB improvement, but together they exceed 16MB. Need adaptive GPTQ to use 10 clips when possible, 5 when artifact is too large.
