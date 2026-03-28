# Experiment: LeakyReLU(0.7)² activation

## Hypothesis
Slope 0.7 preserves more negative gradient flow than 0.5, which could further reduce BPB.
The squaring amplifies the slope quadratically: 0.5→0.25x², 0.7→0.49x².

## Change
Single constant change: `F.leaky_relu(x, negative_slope=0.5)` → `F.leaky_relu(x, negative_slope=0.7)`

## Results
- **val_bpb: 1.1242** (REGRESSED, +0.0035 vs best 1.1207)
- val_loss: 1.8982
- Steps: 6622 (fewer than 0.5's 6940 — also slower per step)
- Even worse than ReLU² baseline (1.1226)

## Analysis
The slope-BPB curve is NOT monotonic and 0.7 is worse than both 0.0 and 0.5:
- Slope 0.0 (ReLU²): 1.1226 BPB
- Slope 0.5: 1.1207 BPB (BEST)
- Slope 0.7: 1.1242 BPB (worst!)

At slope 0.7, the squared negative contribution is 0.49x² — nearly half the positive x².
This makes the MLP too symmetric, losing the beneficial asymmetry where positive activations
dominate. The MLP needs SOME gating (killing very negative activations) but not total gating (ReLU).

Additionally, 0.7 got fewer steps (6622 vs 6940), likely due to slightly different compute
characteristics of the larger negative contributions.

## Key Insight
The LeakyReLU slope has a clear optimum near 0.5. Higher slopes (more symmetric) hurt MORE
than lower slopes (closer to ReLU). The effective negative weight is slope², so:
- 0.3 → 0.09x² (mild negative flow)
- 0.5 → 0.25x² (optimal balance)
- 0.7 → 0.49x² (too symmetric)

## Next Steps
- Try slope 0.3 to confirm the optimum is near 0.5 (not between 0.3-0.5)
- The leaderboard authors likely already tuned this — 0.5 may be the global optimum
- Consider other approaches: MoE, vocab size, speed optimization
