# Current Best

## Best Result

| Metric | Value |
|--------|-------|
| val_bpb | 1.1207 |
| val_loss | 1.8923 |
| artifact_size | 15,548,444 bytes (under 16MB, 452KB headroom) |
| experiment | 20260327_212743_leaky_relu_05_squared |
| seed | 1337 |
| steps | 6940 |
| ms/step | ~86.46 |

## Key Configuration
- 11 layers, 512-dim, 8 heads (4 KV), 3x MLP, 1024 vocab
- **LeakyReLU(0.5)² activation** (key change from previous best)
- FA3 (FlashAttention 3), XSA on last 4 layers
- Partial RoPE (16/64 dims), LN Scale
- EMA (decay=0.998), SWA DISABLED
- GPTQ-lite (10 clip percentiles per row, optimal MSE search)
- Late QAT (threshold=0.15)
- SmearGate + BigramHash (2048 buckets, dim=128)
- Value Embedding (dim=128, layers 9,10)
- Int6 + **brotli-10** compression, legacy torch.save format
- Warmdown 3500 iters, Muon optimizer

## Key Innovation: LeakyReLU(0.5)²
- One-line change: `torch.relu(x)` → `F.leaky_relu(x, 0.5)`
- Preserves negative gradient flow through MLP (ReLU kills 50% of activations)
- -0.0019 BPB improvement over ReLU² baseline
- Slightly slower per step (86.5ms vs 85.2ms) → 106 fewer steps, but BPB improvement dominates

## Previous Innovation: Brotli Compression
- Switched from zstd-22 to brotli quality=10
- Saves ~645KB for EMA 0.998 weights (broader distributions)
- Key unlock: EMA 0.998 always gave better BPB but couldn't fit under 16MB with zstd

## Leaderboard SOTA (for reference)
- **1.1194 BPB** — LeakyReLU² + Legal TTT + Parallel Muon (2026-03-23)
- Our gap: +0.0013 BPB (TTT accounts for most of this difference)
- Without TTT, SOTA base model is ~1.1215 — we're at 1.1207 which is competitive

## Infrastructure Notes
- RunPod template has FA3 pre-installed (no pip install needed)
- brotli needs `pip install --break-system-packages brotli`
- Legacy torch.save format (`_use_new_zipfile_serialization=False`) compresses better with brotli
- 452KB headroom allows further experiments
