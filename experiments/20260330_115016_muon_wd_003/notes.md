# Experiment: Muon Weight Decay 0.03 (from 0.04 default)

## Hypothesis
WD=0.02 gave best-ever BPB (1.1186) but 1.58MB over 16MB. WD=0.03 should be a compromise — better BPB than 0.04 while closer to 16MB size limit.

## Change
Single config change: `muon_wd = 0.03` (was 0.04, previously tried 0.02)

## Results
| Metric | Value | vs Best (1.1207 base) | vs WD=0.02 |
|--------|-------|-----------------------|------------|
| val_bpb (sliding window) | **1.1187** | **-0.0020** | -0.0001 (same) |
| val_loss | 1.8888 | -0.0030 | +0.0001 |
| artifact_size | 16,516,776 bytes | **+0.97MB** | -1.06MB |
| steps | 6977 | same | same |

## Analysis
- **BPB: Essentially identical to WD=0.02.** The BPB improvement plateaus somewhere between 0.03-0.04. Most of the -0.002 gain is captured by going from 0.04→0.03.
- **Size: Still over 16MB by 517KB.** 0.03→0.04 reduces artifact by ~1.0MB (16.52→15.55). Need ~0.52MB more reduction.
- Linear interpolation suggests WD≈0.035 would put artifact at ~16.0MB
- BPB at 0.035 would interpolate to ~1.1197 — still a -0.0010 improvement over current best!

## Weight Decay vs Size/BPB Relationship
| WD | BPB | Artifact MB | Delta BPB |
|----|-----|-------------|-----------|
| 0.02 | 1.1186 | 17.58 | -0.0021 |
| 0.03 | 1.1187 | 16.52 | -0.0020 |
| 0.04 | 1.1207 | 15.55 | baseline |

Key insight: BPB is flat from 0.02-0.03 but jumps +0.002 from 0.03-0.04. The optimal WD is somewhere around 0.035 where we get most of the BPB benefit while fitting under 16MB.

## Next steps
1. **WD=0.035** — should give ~16.0MB artifact and ~1.1197 BPB (estimated -0.0010 improvement)
2. If 0.035 is over, try 0.037
3. If 0.035 fits with room, try 0.033
4. With TTT delta of -0.0012, projected best with WD=0.035: ~1.1185 BPB
