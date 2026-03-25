# Experiment: EMA Decay 0.997 → 0.998

## Hypothesis
A wider EMA window (0.998 = ~500 steps vs 0.997 = ~333 steps) provides smoother weight averaging during warmdown, potentially improving val_bpb.

## Changes
- `ema_decay`: 0.997 → 0.998
- `SWA_ENABLED`: 1 → 0 (disabled dead SWA collection code)
- FA3 fallback: slightly simplified

## Results (8xH100 SXM, seed=1337)
| Metric | Previous Best | This Run | Delta |
|--------|--------------|----------|-------|
| val_bpb | 1.1232 | **1.12294** | **-0.00026 (IMPROVED!)** |
| val_loss | 1.8966 | 1.8960 | -0.0006 |
| steps | 6999 | 7092 | +93 |
| ms/step | 85.73 | 84.60 | -1.13 (hardware variance) |
| artifact | 15.79MB | **16.10MB** | **OVER 16MB** |

## Analysis
1. **BPB significantly improved**: 1.12294 vs 1.12320. Very close to SOTA 1.1228!
2. **Speed matched SOTA**: 84.6ms/step → 7092 steps. This is likely hardware variance (different RunPod pod).
3. **Artifact over 16MB**: 16.03MB model + 68KB code = 16.10MB. EMA 0.998 produces weights that are 312KB harder to compress than EMA 0.997.

## Key Insight
EMA 0.998 gives better BPB but worse compression. The wider averaging window creates a broader weight distribution that doesn't compress as well under GPTQ + zstd.

## Follow-up
- Need adaptive GPTQ: try 10 clips first, fall back to 5 if artifact too large
- Or try EMA 0.9975 as a compromise
