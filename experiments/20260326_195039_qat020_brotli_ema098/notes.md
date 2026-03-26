# Experiment: Earlier QAT (threshold 0.20)

## Hypothesis
Increasing late QAT threshold from 0.15 to 0.20 gives ~700 QAT steps instead of ~524. More quantization-aware training should reduce the quantization gap, improving final BPB. Now feasible with brotli compression providing 533KB headroom.

## Changes
- `late_qat_threshold`: 0.15 → 0.20

## Results
- **val_bpb: 1.1234** (REGRESSED, +0.0008 vs best 1.1226)
- **steps: 6981** (65 fewer than best's 7046, slower pod at 85.95ms/step)
- Post-EMA BPB: 1.1397 (vs 1.1391 in best — worse, QAT noise hurts training)
- Quantization gap: 0.0076 (vs 0.0073 in best — MORE QAT didn't even reduce the gap!)
- QAT enabled at step 6279 (702 QAT steps vs 526 in best)

## Analysis
More QAT steps hurt rather than helped. Two effects:
1. **Training quality degraded:** post-EMA BPB 0.0006 worse. The quantization noise in the forward pass interferes with learning, especially during warmdown when the model is fine-tuning.
2. **Quantization gap not reduced:** Despite 33% more QAT steps, the gap was slightly *larger* (0.0076 vs 0.0073). The model couldn't adapt to quantization noise any better with more steps.

The current threshold of 0.15 (~524 QAT steps) appears to be near-optimal. Going higher wastes valuable training capacity on noise tolerance instead of real learning.

## What to try next
- Do NOT try even earlier QAT (0.25+) — clearly worse
- Could try 0.10 (fewer QAT steps) to see if less QAT is better, but diminishing returns
- Focus on speed optimization or architectural changes instead
- MoE (Mixture of Experts) could add more capacity per parameter
- Cross-layer KV sharing could improve info flow without more params
