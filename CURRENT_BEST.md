# Current Best

## Best Result

| Metric | Value |
|--------|-------|
| val_bpb | 1.1226 |
| val_loss | 1.8955 |
| artifact_size | 15,467,339 bytes (under 16MB, 533KB headroom) |
| experiment | 20260326_161051_brotli_ema098_noswa |
| seed | 1337 |
| steps | 7046 |
| ms/step | ~85.17 |

## Key Configuration
- 11 layers, 512-dim, 8 heads (4 KV), 3x MLP, 1024 vocab
- FA3 (FlashAttention 3), XSA on last 4 layers
- Partial RoPE (16/64 dims), LN Scale
- EMA (decay=0.998), SWA DISABLED
- GPTQ-lite (10 clip percentiles per row, optimal MSE search)
- Late QAT (threshold=0.15)
- SmearGate + BigramHash (2048 buckets, dim=128)
- Value Embedding (dim=128, layers 9,10)
- Int6 + **brotli-10** compression, legacy torch.save format
- Warmdown 3500 iters, Muon optimizer

## Key Innovation: Brotli Compression
- Switched from zstd-22 to brotli quality=10
- Saves ~645KB for EMA 0.998 weights (broader distributions)
- This was the key unlock: EMA 0.998 always gave better BPB but couldn't fit under 16MB with zstd
- Combined with SWA disabled (saves ~1ms/step overhead → more training steps)

## Leaderboard SOTA (for reference)
- **1.1228 BPB** — same architecture, 7101 steps (signalrush, 2026-03-22)
- **WE NOW BEAT SOTA: 1.1226 vs 1.1228 (-0.0002 BPB)**
- We get 7046 steps vs 7101 (55 fewer, but better BPB)

## Infrastructure Notes
- RunPod template has FA3 pre-installed (no pip install needed)
- brotli needs `pip install --break-system-packages brotli`
- Legacy torch.save format (`_use_new_zipfile_serialization=False`) compresses better with brotli
- 533KB headroom allows further experiments (QAT + fewer clips, EMA tuning, etc.)
