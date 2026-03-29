# TTT Adam Optimizer (lr=0.001)

## Hypothesis
Adam's per-parameter adaptive learning rate would reduce catastrophic forgetting during TTT compared to SGD, allowing higher effective learning rates without destroying generalized knowledge. Expected improvement in TTT delta from -0.0012 to potentially -0.0018+.

## Changes
- Replaced `torch.optim.SGD` with `torch.optim.Adam` in TTT eval
- TTT lr=0.001 (higher than SGD's 0.0005 since Adam is more conservative)
- 1 epoch, no anchor regularization

## Results
- **Base sliding window BPB: 1.1204** (on par with best, model training is fine)
- **TTT BPB: 1.2620** (CATASTROPHIC regression of +0.1416)
- TTT delta: +0.1416 (should be negative; SGD gets -0.0012)
- Artifact size: 15,551,437 bytes (under 16MB)
- Steps: 6964, ~86ms/step

## Analysis
Adam is completely wrong for TTT. The problem is:
1. Adam accumulates variance estimates (second moment) from very few gradient steps per chunk
2. With only a few updates per chunk, the variance estimates are noisy and poorly calibrated
3. The denominator (sqrt(v) + eps) gives unpredictable per-parameter scaling
4. The result is that some parameters get massive updates while others barely move
5. This destroys the model's generalized knowledge far worse than SGD

SGD with momentum provides stable, uniform updates that are predictable and easy to control with LR. Adam's adaptivity is a liability when you only have a handful of gradient steps.

## What to try next
- TTT is exhausted for our architecture. Focus on base model improvements.
- Parameter Banking for more training steps (most promising remaining optimization)
- Or try a fundamentally different approach: cross-layer KV sharing, MoE, etc.
