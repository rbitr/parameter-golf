# Experiment: 10L/512d + WD=0.03

## Hypothesis
Reducing from 11 to 10 layers saves ~2.4M params (~1.4MB artifact space) and gives ~9% more training steps. This headroom allows using weight decay 0.03 (which previously gave -0.0021 BPB but artifact was 517KB over 16MB). The combination of lower WD and more steps should compensate for losing 1 layer.

## Changes
1. `num_layers`: 11 → 10
2. `muon_wd` / `adam_wd`: 0.04 → 0.03
3. `xsa_last_n`: 11 → 10
4. `ve_layers`: "9,10" → "8,9"

## Results
| Metric | Previous Best (11L/WD=0.04) | This Run (10L/WD=0.03) | Delta |
|--------|---------------------------|----------------------|-------|
| val_bpb | **1.1168** | 1.1241 | **+0.0073 (REGRESSED)** |
| val_loss | 1.8856 | 1.8980 | +0.0124 |
| artifact_size | 15,537,700 | 15,112,870 | -425KB |
| steps | 6676 | 7221 | +545 (+8.2%) |

## Analysis
**Massive regression.** The 11th layer is worth ~0.007+ BPB at this model scale. The extra 545 training steps (+8.2%) and lower weight decay combined could not compensate for losing 1 layer of depth.

Key learnings:
1. **Depth is critical at this scale.** 11 layers is a hard floor — going to 10 loses far more quality than can be recovered through other means.
2. **WD=0.03 + 10L artifact: 15.1MB** — well under 16MB. The saved headroom is real (~888KB) but useless without the 11th layer.
3. **Steps vs depth tradeoff is heavily depth-favored.** 8.2% more steps (~545) vs 1 fewer layer: depth wins overwhelmingly.
4. **Do NOT try fewer layers.** This rules out all "fewer layers + wider/lower WD" strategies.

## Implication for Future Experiments
- 11 layers is the minimum (and probably the optimum given 16MB constraint)
- Going to 12+ layers is likely too slow (14L/448d was 23% slower locally)
- Width/depth at this model size: depth >>> width per parameter
- The 11L/512d architecture may be close to optimal for this constraint set
- Future experiments should focus on: training recipe, better compression, or techniques that don't change architecture

## What To Try Next
- 11L/WD=0.035 with Full Hessian GPTQ + brotli (might fit, was only 9KB over before)
- Novel approaches: structured sparsity for better compression, curriculum learning, or fundamentally different model families
