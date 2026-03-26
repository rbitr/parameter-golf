# Experiment: Brotli + EMA 0.998 + No SWA

## Hypothesis
Switching from zstd-22 to brotli-10 compression saves ~645KB, allowing EMA 0.998 (previously 100KB over 16MB) to fit. Disabling SWA recovers ~1ms/step, giving more training steps.

## Changes from current best (speed_cleanup_gptq10, 1.1232)
1. Compression: zstd-22 → brotli-10 (saves ~645KB)
2. EMA decay: 0.997 → 0.998 (better weight averaging)
3. SWA: enabled → disabled (saves ~1ms/step overhead, no BPB benefit)

## Results
- **val_bpb: 1.1226** (NEW BEST, beats SOTA 1.1228!)
- **Artifact: 15,467,339 bytes** (533KB under 16MB)
- **Steps: 7046** at 85.17 ms/step
- val_loss: 1.8955
- Roundtrip quantization gap: 0.0238 BPB (1.1226 → 1.1464 without sliding window)

## Analysis
- Brotli-10 is dramatically better than zstd-22 for quantized weight compression (~4% smaller)
- This was the key unlock: EMA 0.998 always produced better BPB but couldn't fit under 16MB with zstd
- SWA disabled + no SWA overhead = more steps (7046 vs 6940 with SWA)
- The 533KB headroom opens up further experiments (lower clips for better BPB, etc.)

## Key compression findings
- zstd-22: non-monotonic (levels 19-20 sometimes better, sometimes worse)
- brotli-10: consistently ~380-645KB better than zstd-22 across all weight distributions
- brotli-10 is especially good with EMA 0.998 weights (wider distributions compress poorly with zstd)
- Brotli compression takes ~40s on trained weights, decompression <1s

## What to try next
1. Combine brotli + EMA 0.998 + QAT + 7 clips (expected ~1.1224)
2. EMA 0.999 or higher (with brotli, even broader distributions might fit)
3. 5 GPTQ clips with brotli (more BPB headroom from less clipping)
4. Use the 533KB headroom for larger model or more parameters

## Run details
- Seed: 1337
- Pod: cy0u6537v2isiy (8xH100 SXM)
- Cost: $6.28
- Total budget spent: $192.52
