# Experiment: Smaller Batch Size (524K tokens)

## Hypothesis
Reducing batch from 786K to 524K tokens gives ~43% more gradient updates (~10K vs ~7K steps).
The model gets nearly double the steps at full LR before warmdown kicks in. More optimizer
updates should improve convergence even with slightly noisier gradients.

## Change
- `train_batch_tokens`: 786,432 → 524,288

## Results
- **val_bpb: 1.1256 (+0.0030 REGRESSED)**
- val_loss: 1.9004
- steps: 10,034 (vs 7,046 baseline — 42% more, as predicted)
- step_avg: ~59.8ms (vs ~85ms — 30% faster per step)

## Analysis
The experiment confirmed the throughput prediction: 42% more steps with 30% faster per-step time.
However, the smaller batch size **significantly hurt convergence** (+0.003 BPB).

The Muon optimizer uses Newton-Schulz orthogonalization of the gradient matrix. With noisier
gradients (smaller batch), the orthogonalization estimate is less accurate. This appears to
matter more than the benefit of extra gradient updates.

The model processed roughly the same total tokens (10,034 * 524K = 5.26B vs 7,046 * 786K = 5.54B)
but with lower quality per update. The extra steps at full LR (6,534 vs 3,546) didn't compensate.

## Key Takeaway
- **Batch size 786K is well-optimized for this model + Muon optimizer combo**
- Don't try smaller batches — gradient quality matters more than update frequency
- Larger batches (1M+) unlikely to help either since we'd lose too many steps
- The 786K batch hits the sweet spot of clean gradients + enough steps

## What to try next
- Focus on techniques that don't change the training dynamics
- Speed optimizations (fewer ms/step at same batch) would be more productive
- Consider novel architectural changes rather than hyperparameter tuning
