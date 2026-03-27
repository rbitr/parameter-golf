# Experiment: LeakyReLU(0.5)² Activation

## Hypothesis
Replace ReLU² with LeakyReLU(0.5)² in MLP. Preserves negative gradient flow through the MLP while maintaining the squared activation inductive bias. Inspired by new SOTA leaderboard entry (1.1194) where this was the biggest single contributor at -0.0021 BPB.

## Change
One-line change in MLP.forward():
- Before: `x = torch.relu(self.fc(x))` → `x.square()`
- After: `x = F.leaky_relu(self.fc(x), negative_slope=0.5)` → `x.square()`

## Results
- **val_bpb: 1.1207** — NEW BEST! -0.0019 vs previous best (1.1226)
- Steps: 6940 (vs 7046 — 106 fewer due to 86.5ms/step vs 85.2ms)
- Post-EMA val_bpb: 1.1372 (before quant+sliding window)
- Artifact: 15.55MB (under 16MB, 450KB headroom)
- Quant gap improved slightly

## Analysis
Massive win! LeakyReLU(0.5) preserves gradient information from negative pre-activations. With ReLU, any negative activation is zeroed — half the information is lost. With LeakyReLU(0.5), negative activations are scaled by 0.5 before squaring, keeping gradient flow alive.

The 0.5 slope is key — it still creates an asymmetry (positive activations are amplified more than negative ones) but doesn't kill gradients entirely. The squaring then maps both sides to positive values.

Note: 106 fewer steps than baseline due to slightly more compute per step (leaky_relu slightly more expensive than relu). This could potentially be recovered with speed optimizations.

## Next Steps
- This is now our best model. Build on it.
- Consider: Can we recover the lost ~100 steps? The leaky_relu overhead is ~1.3ms/step.
- Consider: Try LeakyReLU with different slopes (0.3, 0.7) — 0.5 is from the leaderboard but may not be optimal.
- Consider: Combine with TTT for further gains.
