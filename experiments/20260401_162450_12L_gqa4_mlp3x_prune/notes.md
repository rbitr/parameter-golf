# Experiment: 12L/GQA-4/MLP-3x with Hessian magnitude pruning

## Hypothesis
12L with full GQA-4 and MLP-3x (same per-layer config as 11L) would get the depth benefit seen in 12L/GQA-2 (1.1164) without degrading attention or MLP. The existing magnitude pruning code would trim ~5-6% of weights to fit under 16MB, with modest BPB cost outweighed by the depth benefit.

## Changes
1. `num_layers`: 11 → 12
2. `xsa_last_n`: 11 → 12
3. `ve_layers`: "9,10" → "10,11"
4. Increased pruning search range: 20% → 30%
5. Keep GQA-4, MLP-3x, 512d (unchanged)

## Results
| Metric | Best (11L/GQA-4/MLP-3x) | This Run (12L/GQA-4/MLP-3x) | Delta |
|--------|--------------------------|-------------------------------|-------|
| val_bpb | **1.1168** | **1.1571** | **+0.0403 CATASTROPHIC** |
| base BPB (post-EMA) | ~1.12xx | 1.1341 | ~+0.01 |
| artifact | 15,537,700 | 15,251,813 | -286KB |
| pre-prune artifact | ~15.5MB | **17,778,040** | +2.2MB |
| steps | 6676 (90ms/step) | 6178 (97ms/step) | -498 (7.5%) |
| pruning | 0% | **27%** (up to 43% in proj layers) | — |
| quant gap (base→final) | ~0.004 | **0.023** | 6x worse |

## Analysis
**CATASTROPHIC failure. Two compounding problems:**

### Problem 1: Undertrained (fewer steps)
- 12L is 8% slower per step (97ms vs 90ms) → only 6178 steps vs 6676
- 12L has 9% more parameters to train with 7.5% fewer gradient updates
- Base BPB 1.1341 is much worse than 11L's ~1.12xx — the model needs MORE steps for depth to help, not fewer

### Problem 2: Devastating pruning
- Pre-prune artifact: 17.78MB (1.87MB over 16MB limit)
- Pruning code zeroed 27% of least-important weights (up to 43% in MLP proj layers!)
- Quantization gap ballooned from ~0.004 to 0.023 (6x worse)
- The pruning is magnitude-weighted but NOT Hessian-error-aware — it doesn't compensate remaining weights

### Why 12L/GQA-2 looked better (1.1164)
- That experiment didn't HAVE the pruning code, so it reported pre-prune size (17.72MB OVER)
- If it had been pruned to fit, it would have been similarly catastrophic
- The 1.1164 was the UNPRUNED score — misleading for 16MB target

### Key insight
**At this model size, the 16MB budget is the binding constraint.** Every parameter must pull its weight. Adding a layer means:
1. 8% slower training → fewer steps → worse base model
2. 9% more params → pruning required → destroyed model quality
3. Net: much worse than 11L with full parameters utilized

## Conclusions
1. **12L is definitively dead at 16MB** — all three variants tried:
   - 12L/GQA-2/MLP-3x: 1.1164 but 17.7MB OVER (would need ~10% pruning)
   - 12L/GQA-4/MLP-2.5x: 1.1214 (narrow MLP -0.0046)
   - 12L/GQA-4/MLP-3x+prune: **1.1571** (27% pruning catastrophic)
2. **11L/512d is the FINAL architecture** — depth exploration is complete
3. **Magnitude pruning >10% is catastrophic** — need Hessian-error-aware pruning if ever revisited
4. **Future improvements must come from training recipe, not architecture**

## What to Try Next
- Focus on fundamentally different approaches: SSM hybrid, novel tokenizer, structured sparsity WITH Hessian compensation
- Or accept 11L architecture and find training recipe improvements
- Consider Muon NS improvements (optimized polynomial coefficients, per-iteration tuning)
