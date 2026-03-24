# Experiment: SOTA Baseline Reproduction (v4) - First Successful Run

## Hypothesis
Reproduce the SOTA leaderboard entry (1.1228 BPB) as our starting baseline.

## Results
- **val_bpb: 1.1282** (sliding window stride=64)
- **val_loss: 1.9049**
- **Artifact: 17.04 MB — OVER 16MB LIMIT**
- Steps: 6038 in 600s (~99.4ms/step)
- Cost: $6.85

## Issues Found

### 1. Artifact Over 16MB (17.04 MB)
The `zstandard` package is not installed on the RunPod template, so compression fell back to `zlib-9` instead of `zstd-22`. The SOTA entry uses zstd-22 and gets ~15.5MB artifacts.
**Fix:** Install zstandard before training.

### 2. Missing FA3 (99ms/step vs SOTA's 85ms/step)
`flash_attn_interface` failed to import, falling back to PyTorch SDPA flash backend. This costs ~15ms/step, resulting in 6038 steps vs SOTA's 7100 steps (~1000 fewer steps = significantly less training).
**Fix:** Ensure flash-attn is available or install it.

### 3. BPB Gap (1.1282 vs 1.1228)
The 0.0054 BPB gap is likely due to:
- Fewer training steps (6038 vs 7100) from missing FA3
- Possibly worse compression (zlib vs zstd) affecting roundtrip quality

## Analysis
With FA3 + zstd fixes, we should be very close to reproducing 1.1228 BPB. These are infrastructure issues, not algorithmic ones.

## Next Steps
1. Fix zstandard + FA3 installation in RunPod setup
2. Re-run baseline with fixes
3. Then explore novel improvements
