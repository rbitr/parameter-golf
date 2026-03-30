# matrix_lr=0.02 Experiment

## Hypothesis
Lower matrix_lr (0.02 vs 0.025) might improve convergence quality, similar to how lower weight decay improved BPB. Several strong leaderboard entries use matrix_lr=0.02.

## Changes
- matrix_lr: 0.025 → 0.020
- scalar_lr: 0.025 → 0.020
- tied_embed_lr: 0.035 → 0.030

## Results
- val_bpb: 1.1230 (base, no TTT) — **+0.0023 worse** than best base (1.1207)
- Artifact: 15.27MB (under 16MB, 732KB headroom)
- Steps: 6929 (same as baseline)

## Analysis
Lower LR causes underfitting at our ~7000 step budget. The entries using matrix_lr=0.02 either had more steps (via speed optimizations) or different architectures. Our LeakyReLU(0.5)² + 11L/512d architecture is well-matched to matrix_lr=0.025.

The smaller artifact (15.27MB vs 15.55MB) makes sense — lower LR = smaller weight magnitudes = better compression. But the BPB tradeoff is terrible.

## What to try next
- matrix_lr=0.030 (slightly higher than current 0.025) — the beneficial WD=0.03 experiment showed the model benefits from larger weights. Higher LR might push weights larger AND improve convergence.
- Speed optimizations (batched NS in Muon) to get more steps, which could make lower LR viable.
