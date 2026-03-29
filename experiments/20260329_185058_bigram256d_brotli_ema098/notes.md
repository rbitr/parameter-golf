# Experiment: BigramHash 256d (doubled from 128d)

## Hypothesis
The 128d bigram embedding is a bottleneck. Doubling to 256d captures richer character transition patterns. We have 439KB size headroom to accommodate ~240KB of extra parameters.

## Changes
- `bigram_dim`: 128 → 256 (default in hyperparameters)
- Everything else unchanged from best config

## Results
- **Sliding window (base)**: 1.1209 BPB (REGRESSED +0.0002 vs best 1.1207)
- **TTT**: 1.1201 BPB (REGRESSED +0.0003 vs best 1.1198)
- **TTT delta**: -0.0008 (worse than -0.0012 with 128d)
- **Artifact**: 15.76MB (under 16MB, 243KB headroom)
- **Steps**: 6985

## Analysis
Same pattern as BigramHash 3072 buckets: more bigram parameters trade base improvement for TTT effectiveness, resulting in net regression. The extra 328K parameters (embedding: +262K, projection: +66K) are undertrained at ~7000 steps.

The BigramHash dimension appears to have diminishing returns at 128d already. The 128→256 dim increase adds more params per hash bucket (doubling representation capacity) but at our step count, the extra capacity isn't utilized. This contrasts with the bucket count experiment (3072 vs 2048) which at least showed marginal base improvement.

## Key Insight
BigramHash is optimized at 2048 buckets × 128d for our ~7000 training steps. Any expansion (more buckets OR wider dim) hurts net performance because:
1. Extra params are undertrained
2. Larger bigram subsumes some patterns TTT would otherwise capture, reducing TTT delta
3. The net effect is always neutral or slightly negative

## What to try next
Focus on non-bigram improvements. Consider:
- Cross-layer KV sharing (novel, untried)
- Muon optimizer tuning (momentum, warmup)
- Curriculum learning (shorter seqs early)
- Speed optimizations to get more steps (which could then support bigger bigrams)
