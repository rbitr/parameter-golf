# Experiment: Temperature Scaling Sweep

## Hypothesis
Post-hoc temperature scaling (dividing logits by T before softmax) could improve BPB if our model's output distribution is poorly calibrated. The ternary entry used T=0.90 for +0.025 BPB improvement.

## Changes
- Added `temperature` parameter to `eval_val_sliding`
- Sweep temperatures [0.90, 0.95, 1.00, 1.05, 1.10] after normal sliding window eval
- Training unchanged from current best (LeakyReLU²+EMA0.998+brotli+TTT)

## Results

### Training (identical to current best config)
- val_bpb: 1.1203 (with TTT), base: 1.1212 (sliding window T=1.0)
- Steps: 6943, 86.42 ms/step
- Artifact: 15,554,533 bytes

### Temperature Sweep (on quantized model, sliding window)
| T    | val_loss | val_bpb | delta vs T=1.0 |
|------|----------|---------|----------------|
| 0.90 | 1.9100   | 1.1312  | +0.0100 |
| 0.95 | 1.8984   | 1.1244  | +0.0032 |
| 1.00 | 1.8930   | 1.1212  | baseline |
| 1.05 | 1.8940   | 1.1217  | +0.0005 |
| 1.10 | 1.8992   | 1.1248  | +0.0036 |

## Analysis
- **T=1.0 is already optimal.** All other temperatures are worse.
- T<1 (sharpening) hurts badly — the model is already appropriately confident.
- T>1 (smoothing) also hurts — the model isn't overconfident.
- The ternary entry's T=0.90 benefit was specific to ternary quantization's under-confidence. Our int6 model is well-calibrated.
- Run-to-run variance: this run gave 1.1203 TTT BPB vs previous best 1.1198 — within normal noise (~0.0005).

## What This Tells Us
- Our model's calibration is good — logit_softcap=30 + cross-entropy training produces well-calibrated outputs
- Post-hoc evaluation tricks are unlikely to help. Need genuine training/architecture improvements.
- The gap to SOTA (0.0004 BPB) is primarily from more training steps (Parameter Banking)

## SwiGLU Ablation (local only)
Also tested SwiGLU MLP activation locally (matched params: hidden=1024 vs 1536):
- SwiGLU at step 100 (1 GPU): val_bpb 3.2492
- LeakyReLU² at step 100 (1 GPU): val_bpb 3.2423
- SwiGLU is consistently ~0.007 BPB worse. The 2/3 width reduction hurts more than gating helps at this model size.

## Next Steps
1. **Parameter Banking** — the only remaining path to more training steps (~200 more = ~0.0003 BPB)
2. Accept that we're at parity with SOTA (within noise) and focus on novel techniques
