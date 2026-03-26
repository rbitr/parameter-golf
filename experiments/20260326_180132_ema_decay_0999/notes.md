# Experiment: EMA decay 0.999

## Hypothesis
EMA 0.997→0.998 gave -0.0006 BPB. Pushing to 0.999 may continue the trend.
With brotli compression, wider weight distributions should still fit under 16MB.

## Change
Single line: `ema_decay = 0.998` → `ema_decay = 0.999`

## Results
- **val_bpb: 1.1293** (REGRESSED +0.0067 vs best 1.1226)
- val_loss: 1.9068
- steps: 6951 (fewer than 7046 with 0.998)
- Training time: 600s

## Analysis
EMA 0.999 is far too aggressive. The effective averaging window is ~1000 steps,
which at 7000 total steps means we're averaging the last ~14% of training.
This includes too many early-warmdown weights that are significantly different
from the final weights.

The sweet spot is EMA 0.998 (effective window ~500 steps = last 7% of training).
This covers the late warmdown phase where weights are close to optimal.

## Key Learning
- EMA decay has a clear optimum around 0.998 for ~7000-step training
- 0.997: too narrow (1.1232), 0.998: optimal (1.1226), 0.999: too broad (1.1293)
- The relationship is NOT monotonic — there's diminishing then negative returns
- Don't pursue 0.9985 or other intermediate values; the improvement from 0.997→0.998
  was only 0.0006 and we're already past the optimum direction

## Next Ideas
- Speed optimization: still 55 steps behind SOTA (7046 vs 7101)
- MoE architecture: more capacity within 16MB
- Wider/deeper architecture tradeoffs with brotli headroom
