# Experiment: rope_dims32

## Hypothesis
Doubling RoPE dims from 16/64 to 32/64 (50% positional, 50% position-independent) gives the model better position awareness, improving next-token prediction.

## Changes
- `rope_dims = 16` → `rope_dims = 32`

## Results
- **CRASHED** (rc=1) on both RunPod runs
- Training itself completed: 6655 steps (run 1), 6662 steps (run 2)  
- Artifact was 16.41MB — **OVER 16MB by ~500KB**
- Pruning code triggered but `torch.quantile()` crashed on the ~25M element importance tensor
- Bug: `RuntimeError: quantile() input tensor is too large` — affects ANY experiment that produces >16MB artifact

## Key Findings
1. **rope_dims=32 increases artifact size by ~900KB** (16.41MB vs 15.54MB baseline). Despite not changing parameter count, different RoPE configuration produces weight distributions that compress worse with brotli. This is a significant finding.
2. **torch.quantile() bug in pruning code** — the magnitude-based pruning system hasn't been triggered since it was written (all recent experiments fit under 16MB). When triggered, torch.quantile fails on tensors >2^24 elements. **Fixed** by switching to `np.percentile()`.

## What to Try Next
- Retry rope_dims=32 with the np.percentile bugfix — pruning should now work, but quality impact of pruning ~500KB is uncertain
- OR revert rope_dims to 16 and try a different experiment (the 900KB compression penalty suggests rope_dims=32 may not be worth it even if training quality improves)
- The bugfix is valuable regardless — any future experiment that goes over 16MB will need working pruning

## Cost
- 2 RunPod runs: ~$15.09 total (both crashed during post-training)
