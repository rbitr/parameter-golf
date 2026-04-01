# Experiment: 12L with Cross-Layer KV Sharing

## Hypothesis
By sharing K and V projections between pairs of adjacent layers (stride=2), a 12-layer model can fit under 16MB while keeping full 3x MLP width. The 12L/GQA-2 model (1.1164 BPB, 17.7MB) showed depth helps; KV sharing saves ~1.57M params to make it fit.

## Changes
- num_layers: 11 → 12
- kv_share_stride=2: layers (0,1), (2,3), (4,5), (6,7), (8,9), (10,11) share K/V projections
- XSA on all 12 layers
- VE on layers 10,11
- TTT disabled (dead with Full Hessian GPTQ)
- Added state_dict deduplication to avoid storing shared weights twice

## Results
- **val_bpb: 1.1264 (+0.0096 WORSE than 1.1168 best)**
- Artifact: 15,844,420 bytes (15.8MB, fits under 16MB)
- Steps: 6173 (vs 6676 for 11L — 8% fewer from extra layer overhead)
- Step avg: 97.2ms (vs ~90ms for 11L)
- Required 13.8% magnitude pruning to fit (pre-prune was 16.8MB)

## First attempt (without dedup)
- Artifact was 17.7MB — shared K/V weights were stored twice in state_dict
- Fixed by deduplicating by GPU tensor data_ptr before CPU copy
- Removed 12 shared aliases (6 K + 6 V shared pairs)

## Analysis
1. **KV sharing degrades quality significantly**: +0.0096 BPB is devastating. Shared K/V projections force adjacent layers to attend to the same aspects of the input, limiting attention diversity.
2. **Pruning compounds the damage**: Even with dedup, the 12L model was 16.8MB and needed 13.8% pruning, adding ~0.003 BPB penalty.
3. **Consistent with depth recurrence findings**: At this model size, every unique parameter matters. Weight sharing (whether full blocks or just K/V) always hurts more than the saved params help.
4. **12L needs unique K/V to work**: The 12L/GQA-2 (no sharing) got 1.1164 BPB because each layer had unique K/V. The quality comes from diverse attention patterns, not just depth.

## Key Learning
Cross-layer KV sharing is a dead end for this model size. The attention diversity provided by independent K/V projections per layer is critical. Any parameter-sharing approach that reduces unique weights is unlikely to help.

## What to try next
- Focus on techniques that DON'T reduce unique parameters
- Look at improving training efficiency (more steps within 10 min)
- Consider structured sparsity (2:4) which preserves unique params but improves compression
- Investigate Muon optimizer improvements (fewer NS steps, better coefficients)
