# Experiment: 12L/GQA-4/MLP-2.5x

## Hypothesis
Adding a 12th layer while reducing MLP from 3x→2.5x (1536→1280 hidden) to fit under 16MB. Keep GQA-4 (proven quality). Depth is the most critical factor, so the extra layer should compensate for narrower MLP.

## Changes
1. `num_layers`: 11 → 12
2. `mlp_mult`: 3.0 → 2.5
3. `xsa_last_n`: 11 → 12
4. `ve_layers`: "9,10" → "10,11"

## Results
| Metric | Best (11L/GQA-4/MLP-3x) | 12L/GQA-2/MLP-3x | This Run (12L/GQA-4/MLP-2.5x) |
|--------|--------------------------|-------------------|-------------------------------|
| val_bpb | **1.1168** | 1.1164 (OVER 16MB) | **1.1214 (REGRESSED)** |
| artifact | 15,537,700 | 17,723,594 | 15,240,382 |
| steps | 6676 | 6446 | 6391 |

## Analysis
**MLP width matters more than an extra layer.** Key findings:

1. **MLP 2.5x is devastating**: -16.7% hidden dim per layer = +0.0046 BPP worse, far more than the 12th layer helps
2. **Total MLP capacity**: 12×1280 = 15,360 vs 11×1536 = 16,896 (-9.1%). Even the aggregate total is less.
3. **Steps lost**: 6391 vs 6676 = 285 fewer steps (-4.3%). 12 layers is ~4% slower per step
4. **Depth vs width**: At this model size, MLP width per layer matters almost as much as depth. The model needs wide enough layers to learn useful features.

Comparing the two 12L experiments:
- 12L/GQA-2/MLP-3x: 1.1164 (better BPB, but 17.7MB - OVER)
- 12L/GQA-4/MLP-2.5x: 1.1214 (worse BPP, but 15.2MB - fits)

This tells us GQA-2 with wider MLP is better than GQA-4 with narrow MLP. But neither beats 11L/GQA-4/MLP-3x at 1.1168.

## Conclusion
**11L/512d/GQA-4/MLP-3x is the architecture sweet spot for 16MB.** You cannot fit a 12th layer without sacrificing too much width. The architectural exploration of deeper/narrower and shallower/wider is now complete:
- 10L: +0.0073 worse
- 11L: baseline (best)
- 12L/narrow: +0.0046 worse
- 12L/GQA-2: -0.0004 better BUT doesn't fit

Future improvements must come from training recipe, quantization, or fundamentally different approaches — not architecture tuning.
