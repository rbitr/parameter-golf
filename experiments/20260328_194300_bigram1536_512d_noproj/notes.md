# Experiment: BigramHash 1536@512d (no projection)

## Hypothesis
Using full-dimensional (512d) bigram embeddings with 1536 buckets, matching SOTA config, would:
1. Improve base BPB by providing more expressive bigram features without projection bottleneck
2. Improve TTT effectiveness (strong bigram priors act as regularizer during test-time adaptation)

## Changes
- `bigram_vocab_size`: 2048 → 1536
- `bigram_dim`: 128 → 512
- This eliminates the 128→512 projection layer (BigramHashEmbedding sets proj=None when bigram_dim==model_dim)

## Results
| Metric | Previous Best | This Run |
|--------|--------------|----------|
| Base sliding window | 1.1207 | 1.1217 (+0.0010) |
| TTT | 1.1195 | 1.1213 (+0.0018) |
| TTT delta | -0.0012 | -0.0004 (WORSE) |
| Artifact size | 15.56MB | 15.89MB |
| Steps | 6943 | 6953 |

## Analysis
**REGRESSED** on all metrics:
1. Base model worse by 0.001 BPB — bigram embedding has 786K params (vs 327K), 2.4x more params to train in same steps. Undertrained, consistent with 4096@128d experiment (+0.0018).
2. TTT delta much worse (0.0004 vs 0.0012) — bigger embeddings do NOT help TTT resilience. Opposite hypothesis: the larger, undertrained bigram embeddings provide a weaker signal, making the model more fragile to TTT perturbation.
3. Artifact size very tight: 15.89MB (105KB headroom).

## Key Learning
- SOTA's BigramHash@512d works in their context because Parameter Banking gives them ~250 more training steps AND their overall architecture is different
- Blindly copying SOTA's bigram config without their speed optimizations doesn't work
- More bigram capacity requires more training steps to be effective
- The bigram undertrain issue is consistent across two experiments (4096@128d and 1536@512d)

## What to Try Next
- Revert to 2048@128d (proven best)
- Focus on speed optimizations (Parameter Banking) before trying bigger bigrams
- Or try grad_clip_norm=1.0 (SOTA uses this, simple one-line change)
