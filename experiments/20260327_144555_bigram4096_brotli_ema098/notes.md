# Experiment: BigramHash 4096 buckets

## Hypothesis
Increasing BigramHash buckets from 2048 to 4096 would reduce hash collisions, giving the model better bigram statistics and lower BPB. Multiple leaderboard entries use 4096+ buckets. We have 533KB artifact headroom.

## Change
Single parameter change: `bigram_vocab_size: 2048 → 4096`

## Results
- **val_bpb: 1.1244** (REGRESSED from 1.1226, +0.0018)
- val_loss: 1.8985
- steps: 7013 (vs 7046 for best — 33 fewer steps due to larger embedding)
- cost: $7.09

## Analysis
The extra bigram buckets hurt rather than helped:
1. **More parameters to train**: 2048 extra buckets × 128 dim = 262K additional parameters. With only 7013 steps, these don't have time to converge.
2. **Fewer training steps**: The larger bigram embedding adds per-step compute, costing 33 steps.
3. **Hash collision reduction irrelevant**: With vocab_size=1024, there are only 1024² = 1M possible bigrams. Even 2048 buckets may be enough for the patterns that matter.
4. **Leaderboard context**: Entries using 4096+ buckets also had different architectures. The bucket count may have been tuned for their specific configurations.

## Lessons
- Don't blindly copy hyperparameters from other entries without understanding the interaction effects.
- At this model size, every parameter needs enough training signal. Adding capacity without adding training time is a net negative.
- BigramHash 2048 is confirmed optimal for our configuration.
- Future bigram improvements should focus on the hash function or embedding dimension, not bucket count.

## Next Ideas
- Speed optimization to get more training steps (the real bottleneck)
- MoE for more capacity without proportional parameter increase
- Cross-layer KV sharing to get more information flow without more parameters
