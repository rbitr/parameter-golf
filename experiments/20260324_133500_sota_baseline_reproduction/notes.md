# Experiment: SOTA Baseline Reproduction

## Hypothesis
Reproducing the best known leaderboard entry (1.1233 BPB from PR signalrush 2026-03-22) as our starting point. This establishes a strong baseline to improve upon.

## Changes from repo baseline
Adopted the complete SOTA script with all techniques:
- **Architecture:** 11L, 512-dim, 8 heads (4 KV), 3x MLP, relu²
- **U-Net skip connections** (5 encoder, 6 decoder)
- **SmearGate + BigramHash** (2048 buckets, dim=128)
- **Exclusive Self Attention (XSA)** on last 4 layers
- **Partial RoPE** (16 of 64 dims) + NTK-aware scaling
- **LN Scale** per layer: 1/sqrt(layer_idx+1)
- **Value Embedding** (shared, dim=128, layers 9,10)
- **EMA** (decay=0.997, every step)
- **SWA** (every 50 steps when scale<0.2)
- **Late QAT** (int6 STE when LR scale<0.15)
- **GPTQ-lite** quantization (per-row optimal clip percentile search)
- **Mixed int6/int8** quantization (int6 for MLP+attn, int8 for embeddings)
- **zstd-22** compression
- **Sliding window eval** (stride=64)
- **Training:** Muon(lr=0.025, momentum=0.99, WD=0.04) + AdamW(WD=0.04)
- **Warmdown:** 3500 iters (wallclock-based)
- **Batch:** 786K tokens, seq_len=2048
- **OrthoInit** + muP-scaled output projections

Only modification: Added FA3 fallback to PyTorch SDPA for local A100 testing.

## Risk Level
Low — this is a known-good configuration.

## Local Test Results
- Ran 61 steps in 90s on 1xA100
- Loss decreasing normally: 6.93 → 6.28
- EMA improved val_bpb: 3.72 → 3.61
- Artifact size: 4.76MB (well under 16MB)
- Quantization roundtrip: val_bpb=3.6268

## RunPod Results
(Pending - waiting for 8xH100 run)

## Analysis
(To be filled after RunPod results)
