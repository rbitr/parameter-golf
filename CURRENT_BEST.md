# Current Best

## Best Result

| Metric | Value |
|--------|-------|
| val_bpb | 1.1168 (base = TTT, TTT delta is 0) |
| val_loss | 1.8856 |
| artifact_size | 15,537,700 bytes (under 16MB, 462KB headroom) |
| experiment | 20260331_115138_hessian_gptq_xsa11 |
| seed | 1337 |
| steps | 6676 |
| ms/step | ~90 |

## Key Configuration
- 11 layers, 512-dim, 8 heads (4 KV), 3x MLP, 1024 vocab
- **LeakyReLU(0.5)² activation**
- FA3 (FlashAttention 3), **XSA on ALL 11 layers**
- Partial RoPE (16/64 dims), LN Scale
- EMA (decay=0.998), SWA DISABLED
- **Full Hessian GPTQ** (Cholesky + column reorder, 5 clip percentiles, block_size=128)
- **AR self-generated calibration** (64 x 2048 seqs, temp=0.8, ~202s on 8xH100)
- Late QAT (threshold=0.15)
- SmearGate + BigramHash (2048 buckets, dim=128)
- Value Embedding (dim=128, layers 9,10)
- Int6 + **brotli-10** compression, legacy torch.save format
- Warmdown 3500 iters, Muon optimizer
- TTT: DISABLED (delta=0.0000 — Full Hessian GPTQ eliminates TTT benefit)

## Key Innovation: Full Hessian GPTQ
- Replaced GPTQ-lite (diagonal Hessian, per-row percentile search) with full GPTQ
- Collects H = X^T X from model's own AR-generated calibration data
- Cholesky decomposition of H^{-1}, column reordering by importance
- Block-wise quantization with cross-block error propagation
- **-0.0036 BPB improvement** — biggest single improvement in the project

## Previous Innovation: LeakyReLU(0.5)²
- `torch.relu(x)` → `F.leaky_relu(x, 0.5)` in MLP
- -0.0019 BPB improvement over ReLU² baseline

## Previous Innovation: Brotli Compression
- brotli quality=10 saves ~645KB vs zstd-22
- Key unlock: EMA 0.998 always gave better BPB but couldn't fit under 16MB with zstd

## TTT Status
- TTT is **DEAD** for this configuration (delta=0.0000)
- Full Hessian GPTQ reduced quantization gap so much that TTT has nothing to recover
- Can be disabled to save ~5min eval time on 8xH100

## Leaderboard SOTA (for reference)
- **1.1147 BPB** — AR Self-Gen GPTQ + all-layer XSA by abaybektursun (2026-03-25)
- Our gap: **+0.0021 BPB** (down from +0.0051)

## Compression Comparison (this run)
- brotli-10: 15,450,738 bytes — FITS
- LZMA preset=9: 16,092,640 bytes — OVER 16MB
- Brotli is definitively better for our model

## Dead Ends (confirmed no improvement)
- LZMA preset=9: 645KB worse than brotli-10. Don't switch.
- TTT: delta=0.0000 with Full Hessian GPTQ. Not worth the eval time.
- Grouped int6 quantization (G=128): -0.0001 BPB (noise). Per-row already optimal.
- eval_seq_len=4096: CATASTROPHIC (1.5502 BPB). RoPE 4x extrapolation fails.

## Infrastructure Notes
- RunPod template has FA3 pre-installed (no pip install needed)
- brotli needs `pip install --break-system-packages brotli`
- Legacy torch.save format (`_use_new_zipfile_serialization=False`) compresses better with brotli
- SSH timeout increased to 1800s to accommodate eval time
- AR generation takes ~202s on 8xH100 (single GPU, no KV cache)
