# Current Best

## Best Result

| Metric | Value |
|--------|-------|
| val_bpb | 1.1200 (with TTT) / 1.1208 (base, no TTT) |
| val_loss | 1.8911 (TTT) / 1.8924 (base) |
| artifact_size | 15,547,111 bytes (under 16MB, 453KB headroom) |
| experiment | 20260328_154436_ttt_anchor_alpha_0003 |
| base_experiment | 20260327_212743_leaky_relu_05_squared |
| seed | 1337 |
| steps | 6976 |
| ms/step | ~86 |

## Key Configuration
- 11 layers, 512-dim, 8 heads (4 KV), 3x MLP, 1024 vocab
- **LeakyReLU(0.5)² activation**
- FA3 (FlashAttention 3), XSA on last 4 layers
- Partial RoPE (16/64 dims), LN Scale
- EMA (decay=0.998), SWA DISABLED
- GPTQ-lite (10 clip percentiles per row, optimal MSE search)
- Late QAT (threshold=0.15)
- SmearGate + BigramHash (2048 buckets, dim=128)
- Value Embedding (dim=128, layers 9,10)
- Int6 + **brotli-10** compression, legacy torch.save format
- Warmdown 3500 iters, Muon optimizer
- **TTT: SGD lr=0.0005, 1 epoch, 32K chunks, freeze_blocks=0, anchor_alpha=0.0003**

## TTT Status
- Anchor regularization (alpha=0.0003) reduces TTT delta from -0.0012 to -0.0007
- But base model got more steps this run (6976 vs 6915), so absolute BPB is better
- The anchor approach is not clearly beneficial — it damps good adaptation and bad drift equally
- TTT improvement range: -0.0007 to -0.0012 depending on run
- Trajectory still degrades after chunk 50 regardless of anchor

## Previous Innovation: LeakyReLU(0.5)²
- `torch.relu(x)` → `F.leaky_relu(x, 0.5)` in MLP
- -0.0019 BPB improvement over ReLU² baseline

## Previous Innovation: Brotli Compression
- brotli quality=10 saves ~645KB vs zstd-22
- Key unlock: EMA 0.998 always gave better BPB but couldn't fit under 16MB with zstd

## Leaderboard SOTA (for reference)
- **1.1194 BPB** — LeakyReLU² + Legal TTT + Parallel Muon (2026-03-23)
- Our gap: +0.0006 BPB (down from +0.0009)
- SOTA gets -0.0021 from TTT; we get -0.0007 to -0.0012
- SOTA uses Parameter Banking + BigramHash 3072 — architectural differences may explain TTT resilience

## Infrastructure Notes
- RunPod template has FA3 pre-installed (no pip install needed)
- brotli needs `pip install --break-system-packages brotli`
- Legacy torch.save format (`_use_new_zipfile_serialization=False`) compresses better with brotli
- SSH timeout increased to 1800s to accommodate TTT eval time
