# Experiment: Working QAT Fix (Weight-Replacement STE)

## Hypothesis
Late QAT was dead-coded: `torch.compile(fullgraph=True)` constant-folded the
`CastedLinear._qat_enabled` class variable. Fix: replace weight values with
quantized versions BEFORE the forward pass (outside compiled graph), then
restore AFTER backward. This is a weight-replacement STE that doesn't change
the computation graph.

## Changes
1. Removed QAT from CastedLinear.forward (no graph change needed)
2. Added weight-replacement QAT in training loop:
   - Before forward: save W, replace with Q(W)
   - After backward: restore W
   - Optimizer updates W using gradients from Q(W) forward (STE)
3. Moved `_qat_enabled` to module-level global
4. Restored `fullgraph=True` (no graph change needed)

## Results
- **val_bpb: 1.1236** (slightly worse than 1.1232 best)
- **artifact: 16.25MB (10-clip) / 16.53MB (5-clip)** — BOTH OVER 16MB!
- **steps: 6918** (vs 6999 best, due to slower RunPod hardware, 86.1 vs 85.7 ms/step)
- **QAT activated at step 6436** (scale=0.15, last ~7% of training)

## Key Metrics
| Metric | This Run | Best (no QAT) | Delta |
|--------|----------|---------------|-------|
| Pre-quant BPB (EMA) | 1.1419 | 1.1388 | +0.0031 (worse, fewer steps) |
| Post-quant BPB (roundtrip) | 1.1474 | 1.1471 | +0.0003 |
| **Quant gap** | **0.0055** | **0.0083** | **-0.0028 (34% reduction!)** |
| Sliding window BPB | 1.1236 | 1.1232 | +0.0004 |
| Artifact size | 16.25MB | 15.79MB | +0.46MB |

## Analysis
**QAT works!** The quantization gap was reduced from 0.0083 to 0.0055 BPB (-34%).
However, two issues prevent this from being an improvement:

1. **Artifact size increased ~0.46MB** — QAT changes weight distributions to be
   more robust to quantization noise. This paradoxically makes them LESS compressible
   because the weights develop patterns that resist the int6 clipping used in GPTQ.

2. **Slower hardware** — 86.1 ms/step vs 85.7, losing 81 steps. This hurt pre-quant
   quality. If we'd had the same hardware speed, BPB might have been 1.1230-1.1232.

## What to try next
1. **QAT with lower threshold (0.05)** — Fewer QAT steps = less artifact size impact
2. **QAT + tighter GPTQ clips** — Compensate for wider distributions
3. **QAT + structured weight regularization** — Encourage compressible distributions
4. **Full GPTQ** — Use calibration data for layer-wise quantization optimization
5. **Abandon QAT, focus on speed** — Get more steps through code optimization
