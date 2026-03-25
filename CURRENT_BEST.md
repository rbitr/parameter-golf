# Current Best

## Best Result

| Metric | Value |
|--------|-------|
| val_bpb | 1.1232 |
| val_loss | 1.8966 |
| artifact_size | 15,789,703 bytes (under 16MB) |
| experiment | 20260325_135701_speed_cleanup_gptq10 |
| seed | 1337 |
| steps | 6999 |
| ms/step | ~85.7 |

## Key Configuration
- 11 layers, 512-dim, 8 heads (4 KV), 3x MLP, 1024 vocab
- FA3 (FlashAttention 3), XSA on last 4 layers
- Partial RoPE (16/64 dims), LN Scale
- EMA (decay=0.997), Tight SWA (every 50 steps, scale<0.2)
- GPTQ-lite (10 clip percentiles per row, optimal MSE search)
- Late QAT (threshold=0.15)
- SmearGate + BigramHash (2048 buckets, dim=128)
- Value Embedding (dim=128, layers 9,10)
- Int6 + zstd-22, legacy torch.save format
- Warmdown 3500 iters, Muon optimizer

## Leaderboard SOTA (for reference)
- **1.1228 BPB** — same architecture, 7101 steps (signalrush, 2026-03-22)
- Gap: 0.0004 BPB (down from 0.0009)
- We get 6999 steps vs 7101 (102 fewer, 85.7 vs 84.6 ms/step)

## Infrastructure Notes
- RunPod template has FA3 pre-installed (no pip install needed)
- zstandard needs `pip install --break-system-packages zstandard`
- Legacy torch.save format (`_use_new_zipfile_serialization=False`) saves ~170KB
