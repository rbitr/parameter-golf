# TTT with Frozen Early Blocks (freeze_blocks=6)

## Hypothesis
Freezing blocks 0-5 (keeping 6-10 trainable) will reduce catastrophic forgetting during TTT by preserving general knowledge in early layers while allowing deeper layers to adapt to local text.

## Results
- **val_bpb: 1.1208** (REGRESSED from 1.1203 with freeze=0)
- val_loss: 1.8924
- artifact: 15,549,178 bytes (under 16MB)
- steps: 6926
- TTT time: 293.7s (faster than freeze=0's 318s, fewer params to update)

## Trajectory Comparison (freeze=6 vs freeze=0)
| Chunk | freeze=6 | freeze=0 |
|-------|----------|----------|
| 1     | 1.1563   | 1.1596   |
| 51    | 1.1103   | 1.1100   |
| 101   | 1.1205   | 1.1196   |
| 191   | 1.1375   | 1.1364   |
| 1893  | 1.1230   | 1.1223   |

## Analysis
1. **Freezing hurts, not helps.** The frozen early blocks can't adapt, reducing total adaptation capacity without meaningfully reducing forgetting.
2. The degradation pattern (peak at chunk 50, then slow recovery) is identical in both cases — forgetting is NOT concentrated in early layers.
3. The 0.0006 BPB gap is consistent throughout the trajectory, suggesting a capacity loss rather than a forgetting reduction.
4. Fewer trainable params (12.8M vs 27M) means each SGD step has less effect but also less ability to adapt.

## What This Tells Us
- The forgetting problem in TTT is distributed across all layers, not just early ones.
- Freezing layers is not the path to better TTT — we need either:
  - Lower learning rate (try lr=0.0002 or 0.0003)
  - Larger chunks (64K tokens → fewer total SGD steps)
  - Weight decay in SGD to prevent drift
  - Or accept that our architecture is less TTT-friendly than SOTA's

## Next Steps
- Try lr=0.0003 (between 0.0005 and 0.0002) with freeze=0
- Try 64K chunks (halve the number of SGD steps)
- Consider adding weight decay to TTT SGD
