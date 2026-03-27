# Experiment: VE 3 layers (8,9,10)

## Hypothesis
Extending Value Embedding from layers 9,10 to 8,9,10 would improve BPB by giving
the model more opportunities to reinject token identity into deep layers. Near-zero
parameter cost (shared embedding table, +1 scale parameter).

## Result
**val_bpb = 1.1241 — REGRESSED (+0.0015 BPB vs best 1.1226)**

## Details
- Steps: 7008 (vs 7046 best — slower pod at 85.62 ms/step)
- Post-EMA BPB: 1.1404 (vs 1.1389 best)
- Quantization gap: identical (~0.016 BPB)
- Artifact: 15.47MB (fits under 16MB)

## Prior attempt: VE 4 layers (7,8,9,10)
- FAILED on RunPod with `RuntimeError: tensor does not have a device` during torch.compile
- Worked locally on 1 GPU but broke with DDP on 8 GPUs
- Likely torch.compile + DDP interaction with ParameterList of 4 elements

## Analysis
- Adding VE to layer 8 hurts model quality by 0.0015 BPB
- Layer 8 is the 4th decoder layer (out of 6) — still in active representation refinement
- Token identity injection at this stage may interfere with learning
- VE layers 9,10 (last 2 decoder layers) is optimal — model needs identity only at the very end
- **Don't extend VE to earlier layers.**

## What to try next
- Speed optimization: getting more training steps (we only got 7008 here vs 7046 for best)
- Completely different approaches: MoE, alternative optimizer settings, etc.
