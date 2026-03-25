# Experiment: STE QAT in Forward Pass

## Hypothesis
Switching from weight-replacement QAT (which bloated artifacts by ~460KB) to the SOTA's STE QAT inside `CastedLinear.forward()` would reduce quantization gap while keeping artifacts under 16MB. The SOTA README reports 15.56MB with this approach.

## Changes
- Added STE QAT inside `CastedLinear.forward()` using `w = w + (w_q - w).detach()` pattern
- Removed weight-replacement QAT code from training loop (clone/restore pattern)
- Enabled `late_qat_threshold=0.15` (was disabled at 0.0)
- Used class-level `CastedLinear._qat_enabled` flag instead of module-level global

## Results
- **val_bpb: 1.1233** (vs 1.1232 current best — essentially identical)
- **artifact: 16,007,044 bytes** (OVER 16MB by 7KB)
- **steps: 7072** (vs 6999 without QAT, vs 7101 SOTA)
- Quant gap: not directly measured but BPB suggests marginal improvement

## Analysis
- STE QAT adds ~217KB to artifact size (15.79MB → 16.0MB), much less than weight-replacement QAT (+460KB), but still enough to exceed 16MB
- The SOTA achieves 15.56MB with identical QAT — the size difference must come from other factors (step count differences, small code differences affecting weight distributions)
- BPB is no better than without QAT — the quantization gap reduction is offset by training with quantization noise
- More steps (7072 vs 6999) suggests QAT doesn't slow down training

## Key Learnings
1. STE QAT is clearly better than weight-replacement for artifact size (half the bloat)
2. But even STE QAT pushes us ~7KB over the 16MB limit
3. The gap to SOTA is likely NOT about QAT — it's about getting more steps (7101 vs 7072) and other subtle code differences
4. Need to either (a) find ~200KB of savings elsewhere to enable QAT, or (b) focus on other improvements

## Next Ideas
- Speed optimization to get more training steps (SOTA gets 7101, we get 7072)
- Disable QAT, focus on compression improvements (better zstd dict, pruning tiny weights)
- Try QAT with threshold=0.05 (later activation = less weight perturbation = better compression)
- MoE architecture for more capacity per parameter
