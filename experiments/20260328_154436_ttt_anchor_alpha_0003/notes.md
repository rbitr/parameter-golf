# TTT Anchor Regularization (alpha=0.0003)

## Hypothesis
After each TTT SGD step, blend weights back toward original checkpoint values with `p.data.lerp_(orig, 0.0003)`. This prevents unbounded drift from the original trained weights, addressing the degradation seen after chunk 50.

## Results
- **Sliding window BPB: 1.12076** (base model quality)
- **TTT BPB: 1.12004** (**NEW BEST ABSOLUTE**, -0.00073 from TTT)
- TTT time: 291s on 8xH100
- Artifact: 15,547,111 bytes (under 16MB)
- Steps: 6976 (more than prev 6915, hardware variance)

## Comparison with Previous TTT (no anchor)
| Metric | No anchor | Anchor 0.0003 |
|--------|-----------|---------------|
| Base BPB | 1.1215 | 1.12076 |
| TTT BPB | 1.1203 | 1.12004 |
| TTT delta | -0.0012 | -0.00073 |
| Steps | 6915 | 6976 |

## Analysis
1. **Anchor reduces TTT benefit** — from -0.0012 to -0.00073. The anchor pulls weights back after each step, limiting both adaptation AND drift. At alpha=0.0003, it's too conservative.
2. **Absolute BPB is better** — 1.1200 vs 1.1203, but this is mainly from the base model getting 61 more training steps (6976 vs 6915).
3. **Trajectory still degrades after chunk 50** — the anchor doesn't change the fundamental pattern, just damps everything equally.
4. The anchor approach treats all parameters equally. The forgetting problem might require a more targeted solution.

## Trajectory
- Chunk 1: 1.157 (cold start)
- Chunk 51: 1.110 (peak, similar to no-anchor)
- Chunk 101: 1.119 (degradation starts, slightly better than no-anchor's 1.120)
- Chunk 191: 1.137 (worse, but no-anchor had 1.136 — essentially same)
- Chunk 1893: 1.122 (final, same as no-anchor)

## What This Tells Us
- Anchor regularization is not the right approach. It reduces both good adaptation and bad drift equally.
- The degradation is not from individual weights drifting far — it's from the cumulative direction of updates.
- Need a fundamentally different approach to improve TTT:
  - Try **higher lr** (0.001) to get more early benefit, accept more late degradation
  - Try **lr=0.0005 with weight decay** (different from anchor: penalizes weight magnitude, not distance from original)
  - Try **64K chunks** (fewer total SGD steps = less total drift)
  - Try **resetting model every N chunks** (fresh starts prevent cumulative drift)
  - Accept that our architecture gets -0.0007 to -0.0012 from TTT and focus on improving the base model instead

## Next Priority
The base model improvement matters more than TTT tuning. Focus on architectural changes that improve base BPB (currently ~1.1207). MoE, cross-layer KV sharing, or larger vocab could yield bigger gains than squeezing another 0.0005 from TTT.
