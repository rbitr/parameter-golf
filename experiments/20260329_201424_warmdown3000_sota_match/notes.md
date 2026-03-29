# warmdown_iters=3000 (SOTA match attempt)

## Hypothesis
SOTA uses warmdown_iters=3000 (42% of training) vs our 3500 (50%). Reducing warmdown gives 500 more peak-LR steps, potentially improving training quality. SOTA achieves 1.1194 with this setting.

## Result: REGRESSED
- Sliding window (no TTT): **1.1211** (vs best 1.1207, +0.0004 worse)
- TTT: **1.1205** (vs best 1.1198, +0.0007 worse)
- TTT delta: **-0.0006** (vs best -0.0012, halved effectiveness)
- Steps: 6967 (slightly fewer than average ~7000)
- Artifact: 15.72MB (larger than our best 15.55MB)

## Analysis
Shorter warmdown hurts our model. The key difference from SOTA:
- We use EMA 0.998 (broader averaging), SOTA uses EMA 0.997
- Broader EMA needs more warmdown time for weights to settle into a good average
- The extra 500 peak-LR steps don't compensate for the worse convergence at end of training
- Artifact size also increased (weights less smooth → worse compression)

## Key Learning
- warmdown_iters=3500 is optimal FOR OUR CONFIG (EMA 0.998, brotli)
- SOTA's warmdown=3000 works for them because of EMA 0.997 (tighter averaging)
- Don't blindly copy SOTA hyperparameters — they interact with other choices
- The warmdown range is now fully characterized: 3000 (worse), 3500 (optimal), 3800 (worse)

## What to try next
- Speed optimizations (batched NS or parameter banking) are the main remaining path
- Cross-layer KV sharing as a novel architecture idea
- Or accept that we're within noise of our ceiling and focus on reducing variance
