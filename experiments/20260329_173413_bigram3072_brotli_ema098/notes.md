# BigramHash 3072 Buckets

## Hypothesis
SOTA ablation shows BigramHash expansion 2048→3072 gives -0.0009 BPB. Our 4096 experiment regressed (+0.0018) because extra params were undertrained. 3072 adds only half the params of 4096 (131K vs 262K), so should train adequately at ~7000 steps.

## Change
Single hyperparameter: `bigram_vocab_size` from 2048 to 3072.

## Results
- Sliding window (base): **1.1205** (vs previous best 1.1207 = **-0.0002**)
- TTT: **1.1198** (tied with previous best 1.1198)
- TTT delta: -0.0007 (vs -0.0012 with 2048 buckets at 1ep)
- Artifact: 15.68MB (under 16MB, 318KB headroom)
- Steps: 6986 (similar to baseline)
- ms/step: ~85.9 (no speed penalty)

## Analysis
- Base improved by -0.0002, not the -0.0009 SOTA claims. This suggests either:
  - SOTA's bigger improvement comes from their extra steps (~7200 vs our 6986)
  - The -0.0009 in SOTA ablation interacts with other techniques (parallel muon, EMA 0.997+SWA)
  - 3072 buckets still somewhat undertrained at our step count
- TTT delta dropped from -0.0012 to -0.0007 — bigger bigram table captures more bigram statistics in the base model, leaving less for TTT to improve
- Net effect: tied with previous best on TTT. Small base improvement.

## Key Learning
- BigramHash 3072 is directionally positive but the improvement is within noise (-0.0002)
- More bigram buckets trade off against TTT delta (base improves, TTT delta shrinks)
- The sweet spot for our step count is probably near 2048-3072 buckets
- SOTA's -0.0009 from BigramHash expansion likely requires their speed optimization (more steps)

## Next Ideas
- **TTT 3ep + 2 frozen blocks at lr=0.0005**: SOTA uses this exact config. We tried 6 frozen (too many) and 0 frozen multi-epoch (causes forgetting). 2 frozen might be the sweet spot.
- **Revert to 2048 bigram + focus on TTT improvement**: Since TTT delta is our main gap
