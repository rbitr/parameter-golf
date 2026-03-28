# TTT Conservative: lr=0.0005, 1 epoch

## Hypothesis
Lower lr (0.0005 vs 0.002) and fewer epochs (1 vs 3) will prevent the catastrophic forgetting seen in the first TTT attempt.

## Results
- **Sliding window BPB: 1.1215** (base model quality, slight hardware variance from best 1.1207)
- **TTT BPB: 1.1203** (-0.0012 improvement from TTT!)
- TTT time: 318s on 8xH100
- Artifact: 15,545,342 bytes (under 16MB)

## TTT Trajectory
- Chunk 1 (no training): 1.160
- Chunk 51: 1.110 (rapid improvement — model adapts well)
- Chunk 101: 1.120 (starting to degrade)
- Chunk 1893: 1.122 (mild degradation, much better than v1's 1.146)

## Comparison with First Attempt
| Parameter | v1 (SOTA defaults) | v2 (conservative) |
|-----------|--------------------|--------------------|
| lr | 0.002 | 0.0005 |
| epochs | 3 | 1 |
| Total SGD steps | ~5676 | ~1892 |
| Final TTT BPB | 1.1436 | 1.1203 |
| Improvement | -0.023 (REGRESSED) | -0.0012 (improved!) |

## Analysis
1. TTT works but is very sensitive to hyperparameters
2. 12x less total training (lower lr × fewer epochs) turned regression into improvement
3. The trajectory still shows degradation after chunk 50 — the model adapts quickly to nearby text but slowly loses general knowledge
4. The SOTA gets -0.0021 from TTT; we get -0.0012. Gap likely from their different architecture (parameter banking) being more stable for SGD

## What This Tells Us
- With 1.1207 base + 0.0012 TTT improvement, we'd get ~1.1195 — essentially matching SOTA (1.1194)
- Further TTT tuning could close the remaining 0.0009 gap
- Key dimensions to explore: even lower lr (0.0002), larger chunks (64K), or freeze early blocks

## Next Steps
1. Try lr=0.0002 to further reduce degradation
2. Try 64K or 128K chunks (fewer total steps)
3. Try freezing first 4 blocks (only adapt deeper layers)
4. These could potentially push TTT improvement to -0.002+
