# Current Best

## Best Result

| Metric | Value |
|--------|-------|
| val_bpb | 1.1237 |
| val_loss | 1.8974 |
| artifact_size | 15,957,740 bytes (under 16MB) |
| experiment | 20260324_200231_legacy_format |
| seed | 1337 |
| steps | 6898 |
| ms/step | ~87 |

## Key Configuration
- 11 layers, 512-dim, 8 heads (4 KV), 3x MLP, 1024 vocab
- FA3 (FlashAttention 3), XSA on last 4 layers
- Partial RoPE (16/64 dims), LN Scale
- EMA (decay=0.997), Tight SWA (every 50 steps, scale<0.2)
- GPTQ-lite (per-row optimal clip percentile)
- Late QAT (threshold=0.15)
- SmearGate + BigramHash (2048 buckets, dim=128)
- Value Embedding (dim=128, layers 9,10)
- Int6 + zstd-22, legacy torch.save format
- Warmdown 3500 iters, Muon optimizer

## Leaderboard SOTA (for reference)
- **1.1228 BPB** — same architecture, 7101 steps (signalrush, 2026-03-22)
- Gap: 0.0009 BPB, likely from fewer steps (6898 vs 7101)

## Infrastructure Notes
- RunPod template has FA3 pre-installed (no pip install needed)
- zstandard needs `pip install --break-system-packages zstandard`
- Legacy torch.save format (`_use_new_zipfile_serialization=False`) saves ~645KB
