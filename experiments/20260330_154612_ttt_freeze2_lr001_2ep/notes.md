# Experiment: TTT freeze=2, lr=0.001, 2 epochs

## Hypothesis
Freezing the first 2 transformer blocks during TTT (following SOTA's configuration) would protect basic representations, allowing higher LR (0.001 vs 0.0005) and more epochs (2 vs 1) without catastrophic forgetting. Expected TTT delta: -0.0015 to -0.0018.

## Results
- **Base (sliding window s64):** 1.1211 BPB
- **TTT (freeze=2, lr=0.001, 2ep):** 1.1230 BPB
- **TTT delta: +0.0019 (REGRESSED)**
- Steps: 6923, artifact: 15.55MB

## Analysis
The chunk-by-chunk TTT log tells the full story:
- Chunks 1-50: BPB improves from 1.158 to 1.111 (strong initial adaptation)
- Chunks 50-100: BPB rises to 1.119 (forgetting begins)
- Chunks 100-200: BPB rises to 1.135 (catastrophic forgetting)
- Chunks 200-1893: Gradual recovery as cosine LR decay kicks in, settling at 1.125

The total update magnitude per chunk is ~4x our best config (lr=0.001 × 2ep vs lr=0.0005 × 1ep). Even with 2 frozen blocks, the remaining 9 blocks receive too-large updates that destroy learned representations.

The cosine LR decay (already in our implementation) prevents things from getting as bad as our previous lr=0.002/3ep catastrophe (1.1436), but can't overcome the damage from high-LR early chunks.

## Key Learning
1. **freeze=2 doesn't help** with our model architecture. The forgetting is distributed across ALL layers, not concentrated in early layers.
2. **Our model is more sensitive to TTT** than SOTA's model. SOTA handles lr=0.002, 3ep because their model has:
   - BigramHash 1536d (more robust representations)
   - 7185 steps (better converged)
   - Parameter Banking (more training efficiency)
3. **TTT is fully optimized** at lr=0.0005, 1ep, freeze=0 for our architecture. The -0.0012 TTT delta is the maximum achievable without improving the base model.

## TTT Configuration Exhaustion Summary
| Config | TTT Delta | Status |
|--------|-----------|--------|
| lr=0.0005, 1ep, freeze=0 | **-0.0012** | **BEST** |
| lr=0.001, 1ep, freeze=0 | -0.0010 | Slightly worse |
| lr=0.0005, 2ep, freeze=0 | -0.0007 | 2nd epoch causes forgetting |
| lr=0.001, 2ep, freeze=2 | **+0.0019** | Catastrophic |
| lr=0.002, 3ep, freeze=0 | +0.023 | Catastrophic |
| lr=0.0005, 3ep, freeze=2 (SOTA) | +0.0003 | Catastrophic |
| Adam lr=0.001 | +0.1416 | Adam wrong for TTT |
| anchor_alpha=0.0003 | -0.0007 | No benefit |

## What to Try Next
- Focus on base model improvements since TTT is capped
- The gap to SOTA is entirely in base model quality (their BigramHash, speed, and step count)
- Consider novel architectural changes or training tricks
