# Experiment: VE 4 layers (7,8,9,10) → FAILED on RunPod

## Hypothesis
Extending Value Embedding from 2 layers (9,10) to 4 layers (7,8,9,10) would improve BPB
by giving the model more opportunities to reinject token identity into deep layers.
Near-zero parameter cost (shared embedding table + 1 scale per layer).

## Result
**FAILED** on RunPod with `RuntimeError: tensor does not have a device` during torch.compile.
The error occurs in AOT autograd when torch.compile traces the model with DDP on 8 GPUs.
Works fine locally on 1 GPU without DDP.

## Analysis
- The issue is likely a torch.compile + DDP interaction with the expanded ParameterList
- Reverted to VE_LAYERS="8,9,10" (3 layers) for retry
- 3-layer variant passes local test and shows small improvement over baseline

## Local Results (60 steps, not meaningful for absolute BPB)
| Config | Train loss @50 | Post-EMA BPB | Int6 Sliding BPB |
|--------|---------------|-------------|-----------------|
| VE 9,10 (baseline) | 6.4531 | 3.7425 | 3.7604 |
| VE 7,8,9,10 | 6.4043 | 3.7301 | 3.7496 |
| VE 8,9,10 | 6.4033 | 3.7302 | 3.7496 |

## Next
- Try VE 8,9,10 on RunPod when 8xH100 pods become available
- If 3 layers also fails, the issue may be with ANY change to ParameterList length
