# Experiment: 12L KV Sharing (first attempt, timed out)

## What happened
First attempt at 12L with cross-layer KV sharing. SSH timed out at 1800s because:
1. TTT was still enabled (adds ~7 min eval time)
2. Shared K/V weights were stored twice in state_dict (no dedup), making artifact 17.7MB
3. GPTQ + pruning + eval took longer due to larger model

## Results (partial)
- val_bpb: 1.1181 (sliding window, pre-dedup — includes duplicate weights in artifact)
- Artifact: 17,676,571 bytes (17.7MB, OVER 16MB)
- Steps: 6145

## Fix applied
- Added state_dict deduplication before quantization
- Disabled TTT (dead with Full Hessian GPTQ)
- Increased SSH timeout from 1800s to 2100s
- Retry succeeded: see 20260401_130439_12L_kvshare_dedup
