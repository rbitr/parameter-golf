# Experiment: 12L/GQA-2 Depth Experiment

## Hypothesis
Adding a 12th layer by reducing KV heads from 4→2 (GQA-4 → GQA-2) to save params. Depth was shown to be the most critical factor (-0.007 BPB per lost layer). GQA-2 (4:1 Q/KV ratio) is widely used in production models and should maintain attention quality.

## Changes
1. `num_layers`: 11 → 12
2. `num_kv_heads`: 4 → 2
3. `xsa_last_n`: 11 → 12
4. `ve_layers`: "9,10" → "10,11"

## Results
| Metric | Previous Best (11L/GQA-4) | This Run (12L/GQA-2) | Delta |
|--------|--------------------------|----------------------|-------|
| val_bpb | **1.1168** | **1.1164** | **-0.0004** |
| val_loss | 1.8856 | 1.8849 | -0.0007 |
| artifact_size | 15,537,700 | **17,723,594** | **+2.19MB (OVER 16MB!)** |
| steps | 6676 | 6446 | -230 (-3.4%) |

## Analysis
**BPB improved slightly but artifact OVER 16MB.** This is informative:

1. **12L depth helps** — val_bpb improved from 1.1168 → 1.1164 despite GQA-2 degradation and fewer steps
2. **GQA-2 likely hurts quality** — only -0.0004 improvement despite depth gain. If GQA-2 were neutral, depth alone should give more (10→11 layer was +0.007)
3. **Artifact blowup** — 17.72MB vs 15.54MB = +2.19MB. The 12th layer + changed GQA parameters compress poorly
4. **Steps lost** — 6446 vs 6676 = 230 fewer steps (3.4%) from the extra layer + XSA-12

Key insight: depth gain is real but GQA-2 partially cancels it. The artifact overrun needs addressing.

## Next Steps
- Try 12L/GQA-4/MLP-2.5x: keep proven GQA-4, reduce MLP width to fit. Has fewer total params than 11L model, should fit under 16MB.
- Alternatively, 12L/GQA-2/MLP-2.5x: saves even more params but keeps GQA-2 quality risk.
