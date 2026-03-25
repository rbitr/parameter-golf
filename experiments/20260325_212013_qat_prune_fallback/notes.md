# Experiment: QAT + Post-Quantization Pruning Fallback

## Hypothesis
If artifact exceeds 16MB after QAT, prune small quantized values (±1) to zero for better compression. More zeros = better zstd compression.

## Changes
- QAT enabled (threshold=0.15)
- 10 GPTQ clips
- Added post-quantization pruning: if artifact > 16MB, zero all ±1 quantized values in int6 tensors

## Results
- **val_bpb: 1.1427** (TERRIBLE - +0.02 BPP regression)
- **artifact: 14,650,944 bytes** (well under 16MB)
- **steps: 7077**
- Pre-prune artifact: 16,470,691 bytes (470KB over)
- Post-prune artifact: 14,650,944 bytes (saved 1.82MB)

## Analysis
- Pruning ALL ±1 values was catastrophically aggressive
- Saved 1.82MB but destroyed 0.02 BPB — unacceptable tradeoff
- ±1 values represent a huge fraction of int6 weights (range [-31,31], ~10-15% of values are ±1)
- Need much more targeted pruning — only prune smallest reconstructed values (q * scale)

## Key Learnings
1. Global ±1 pruning is far too aggressive — DO NOT repeat
2. Need magnitude-aware pruning (consider q * scale, not just q)
3. The compression savings from pruning are huge (1.82MB) — even 5% of this would be enough
4. A binary search approach pruning by reconstructed magnitude could work but adds significant code complexity
