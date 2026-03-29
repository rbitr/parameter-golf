# Parameter Banking + Parallel Muon

## Hypothesis
Consolidate all nn.Linear weights into 4 contiguous 3D nn.Parameter banks (qo, kv, mlp_up, mlp_down) to enable batched Newton-Schulz orthogonalization in Muon optimizer. Expected to reduce ms/step from ~86 to ~83, gaining ~200 more training steps.

## Result: MASSIVE REGRESSION
- val_bpb: 1.1633 (vs 1.1198 best) — +0.0435 BPB worse
- steps: 3307 (vs ~7000) — only 47% of normal steps
- ms/step: ~182 (vs ~86) — 2.1x SLOWER
- artifact_size: 14.57MB (under budget but irrelevant)

## Root Cause Analysis
Removing DDP (DistributedDataParallel) to implement parameter banking eliminated the critical overlap of gradient communication with backward computation:

1. **DDP overlap is irreplaceable**: DDP overlaps gradient all-reduce with backward pass computation via bucketed coalescing. Without DDP, gradient sync happens sequentially AFTER backward, adding ~90ms/step.
2. **55 separate dist.all_reduce() calls**: Small replicated parameters (biases, norms, embeddings) each needed individual all-reduce, vs DDP's efficient bucketing.
3. **torch.compile regression**: Bank indexing (`self.qo_bank[i]` in a loop) may have prevented optimal fusion.
4. **QAT recompilation spike**: Global variable change at QAT threshold caused a compile cache miss at step 2925.

Interesting note: Steps 1-10 showed ~80ms (faster than baseline!), confirming the forward pass IS faster with banks. The regression comes entirely from sequential gradient sync.

## Key Learning
- DDP's communication-computation overlap is the single most important performance feature for multi-GPU training
- Any architectural change that removes DDP must replicate this overlap (e.g., manual reduce-scatter/all-gather interleaved with backward)
- The SOTA implementation uses a custom three-phase approach: launch_reduce_scatters → Adam steps → Muon step, which manually recreates the overlap

## Next Idea
Simpler approach: Keep DDP + existing model unchanged, but batch NS calls in Muon by grouping same-shape parameters (e.g., stack all 11 Q weights of shape 512×512, do one batched NS call). Reduces kernel launches from ~44 to ~4 without touching DDP.
