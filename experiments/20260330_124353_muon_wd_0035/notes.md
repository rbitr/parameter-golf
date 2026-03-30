# Experiment: muon_wd_0035

## Hypothesis
WD=0.035 interpolates between WD=0.03 (1.1187 BPB, 16.52MB) and WD=0.04 (1.1207, 15.55MB).
Expected ~1.1197 BPB, ~16.0MB. If it fits under 16MB, could beat SOTA.

## Changes
- Single hyperparameter: `muon_wd` from 0.04 to 0.035

## Results
- val_bpb (TTT): 1.1196
- val_bpb (SW base): 1.1204
- artifact: 16,009,329 bytes (OVER by 9,329 bytes)
- steps: 6989 (85.85ms/step)

## Analysis
**The BPB improvement from lower WD is NOT linear — it's a cliff between 0.03 and 0.035.**

| WD | Base BPB | Model Size | Delta vs 0.04 |
|----|----------|-----------|---------------|
| 0.04 | 1.1207 | 15.48MB | — |
| 0.035 | 1.1204 | 15.93MB | -0.0003 |
| 0.03 | 1.1187 | 16.44MB | -0.0020 |
| 0.02 | 1.1186 | 17.50MB | -0.0021 |

WD=0.035 barely improves BPB (-0.0003 base) but grows model by 450KB.
The massive -0.002 BPB improvement only kicks in at WD ≤ 0.03.

TTT delta was -0.0008 (vs our usual -0.0012), within noise.

**Conclusion**: WD tuning is a dead end for our 16MB budget. The beneficial WD range (≤0.03) produces models too large. WD=0.035-0.04 produces similar BPB. Only viable if we find a way to save ~500KB from compression or model architecture.

## What to try next
- Speed optimizations to get more training steps (batched NS in Muon)
- Novel architectural ideas (cross-layer KV, MoE)
- The gap to SOTA (0.0004) is within run variance — focus on systematic improvements
