# Experiment: EMA 0.998 + Adaptive GPTQ (10→5 fallback)

## Hypothesis
Adaptive GPTQ that tries 10 clips first and falls back to 5 clips would ensure the artifact stays under 16MB while getting the best possible quantization quality.

## Changes
- Adaptive GPTQ: try 10 clips, fall back to 5 if artifact > 16MB
- EMA decay: 0.998
- SWA disabled

## Results (8xH100 SXM, seed=1337)
| Metric | Previous Best | This Run | Delta |
|--------|--------------|----------|-------|
| val_bpb | 1.1232 | 1.12330 | +0.0001 (essentially same) |
| artifact | 15.79MB | **16.81MB** | **WAY OVER 16MB** |
| steps | 6999 | 7029 | +30 |

### GPTQ adaptive results:
- 10-clip: 16.18MB (over) → tried 5-clip
- 5-clip: 16.81MB (ALSO over, and LARGER!)

## Analysis

### CRITICAL FINDING: 5 clips produces LARGER artifacts than 10 clips!
The 5-clip set (min clip 0.999) clips fewer outliers than 10-clip (min clip 0.998), keeping wider value ranges. This creates MORE diverse quantized values that compress WORSE under zstd. Counter-intuitive but important:
- More GPTQ clips → tighter clipping options → narrower quantized distribution → better compression
- Fewer clips → wider minimum clip → more outliers kept → worse compression

### Artifact size is HIGHLY variable
- Same config (EMA 0.997, 10 clips) has produced: 15.79MB, 16.13MB, 16.03MB across different runs
- Variance of ~300-400KB from run to run, just from training speed differences
- EMA 0.998 makes this worse by producing a broader weight distribution

### Conclusion
EMA 0.998 is not viable within the 16MB budget. The wider averaging window creates weights that are 200-800KB harder to compress, with high variance. Reverting to EMA 0.997.

## What to Try Next
1. Keep EMA 0.997 + 10 clips (our reliable best)
2. Focus on speed optimization or architectural changes for BPB improvement
3. Consider tighter GPTQ clips (e.g., 0.995-0.999 range) for even more compact artifacts
