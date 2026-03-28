# Experiment: grad_clip_norm=1.0

## Hypothesis
Less aggressive gradient clipping (1.0 vs 0.3) gives Muon optimizer more freedom for its orthogonal updates, potentially improving convergence. SOTA reportedly uses grad_clip=1.0 (confirmed for TTT, uncertain for main training).

## Change
Single line: `grad_clip_norm` default from 0.3 to 1.0.

## Local Results
**Showed clear improvement** on 1xA100:
- step 50 train_loss: 6.20 (was 6.37)
- step 60 val_bpb: 3.6447 (was 3.7445)
- ~2.7% better val_bpb locally

## RunPod Results (8xH100)
- val_bpb: **1.1214** (REGRESSED from 1.1207 by +0.0007)
- val_loss: 1.8934
- steps: 6898 (45 fewer than usual 6943)
- artifact: 15.53MB

## Analysis
**Local vs multi-GPU divergence.** The looser clipping helped on 1 GPU but hurt on 8 GPUs. Possible explanations:
1. With DDP gradient averaging across 8 GPUs, individual gradient norms are already reduced. The 0.3 clip may rarely activate at scale, while 1.0 allows occasional large gradients that cause instability.
2. The 45 fewer steps suggest the training was slightly slower (perhaps unstable early steps?), compounding the issue.
3. Muon's Newton-Schulz orthogonalization may interact differently with gradient norms at different scales.

## Key Learning
- **Local tests can mislead for hyperparameter changes that interact with DDP/multi-GPU.** Gradient clipping is scale-dependent.
- **grad_clip_norm=0.3 is optimal** for our 8xH100 setup. Don't revisit.
- The SOTA (2026-03-22 record) also uses 0.3. The 2026-03-23 entry uses 1.0 only for TTT grad clip, which we already match.

## What to try next
- Parameter Banking for speed optimization (~200 more steps)
- MoE (sparse MLP)
- Cross-layer KV sharing
- TTT with Adam optimizer (could improve TTT delta)
