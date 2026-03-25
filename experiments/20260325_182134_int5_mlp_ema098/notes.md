# Experiment: int5 MLP + EMA 0.998

## Hypothesis
Int5 quantization on MLP weights saves ~1-1.5MB, enabling EMA 0.998 (which showed
1.1229 BPB but was over 16MB with int6-only). If int5 quant error on MLPs < 0.0003 BPB
gain from EMA 0.998, net improvement.

## Changes
- Int5 [-16,15] quantization for MLP weights (clip_range=15)
- Int6 [-32,31] kept for attention weights
- EMA decay changed from 0.997 to 0.998
- Added `int5_cats` parameter to `mixed_quantize_int6()`

## Results
- **val_bpb: 1.1427** — MASSIVE regression (+0.019 vs 1.1232 best)
- artifact: 13.78MB (well under 16MB, saved ~2MB)
- steps: 7017 (similar to baseline)

## Analysis
Int5 quantization is devastating to quality. The 31-level quantization (int5) vs
63-level (int6) adds enormous reconstruction error that dwarfs the EMA 0.998 benefit.

The quantization gap from int5 is roughly +0.019 BPB vs +0.008 BPB from int6.
This 0.011 BPB additional penalty far exceeds the 0.0003 BPB gain from EMA 0.998.

## Key Learning
- Int5 for MLPs is too aggressive at this model size
- Saving space via coarser quantization is not viable — the quality penalty is
  an order of magnitude larger than any training improvement
- Need to find other ways to save space or improve quality
- Alternative: fix the dead-coded QAT to reduce int6 quantization gap instead

## What to try next
- Fix late QAT (currently dead-coded due to torch.compile fullgraph=True)
- Weight-replacement QAT (apply quant outside compiled graph)
- Speed optimizations to get more training steps
