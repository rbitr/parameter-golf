# Experiment: MTP 1 Head Auxiliary Loss

## Hypothesis
Adding 1 multi-token prediction head (predict 2 tokens ahead) as an auxiliary training loss would improve hidden representations. MTP heads are excluded from the exported artifact, so zero size cost. Inspired by DeepSeek's MTP approach.

## Changes from current best (brotli_ema098_noswa, 1.1226)
1. `mtp_num_heads`: 0 → 1 (one auxiliary head predicting 2 tokens ahead)
2. `mtp_loss_weight`: 0.2 (default, added to main CE loss)

## Results
- **val_bpb: 1.1339** (REGRESSED, +0.0113 BPB vs best 1.1226)
- Steps: 6940 (vs 7046 best, lost 106 steps)
- ms/step: 86.46 (vs 85.17 best, +1.5% overhead)
- Post-EMA val_bpb: 1.1498
- Roundtrip quant gap: 0.0081 BPB (1.1498 → 1.1579)
- Sliding window val_bpb: 1.1339

## Analysis
- MTP is devastating at this model scale (+0.0113 BPB regression)
- Two factors hurt:
  1. **Compute overhead**: 86.46ms/step vs 85.17ms → 106 fewer training steps (-1.5%)
  2. **Gradient interference**: The auxiliary loss at weight 0.2 corrupts the main task gradients. At 512-dim, the model can't learn rich enough representations to predict 2 tokens ahead, so the MTP gradient is mostly noise.
- Even without the compute overhead, the regression is far too large for step count alone to explain
- The model is too small for MTP to be beneficial — every parameter and gradient must be focused on next-token prediction

## Conclusion
- MTP is NOT viable at this model size. Don't try lower MTP weights either — the overhead isn't worth a marginal potential gain.
- Focus on techniques that improve the primary task directly, not auxiliary losses.

## What to try next
- Disable late QAT entirely (threshold=0) — low risk, clean test
- Speed optimization to recover more training steps
- MoE or other architectural changes that add capacity without auxiliary losses
