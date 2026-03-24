# Current Best

No experiments run yet. Starting from baseline.

## Baseline Reference

- **Script:** `train_gpt.py` (repo default)
- **Expected val_bpb:** ~1.2244 (from leaderboard "Naive Baseline")
- **Architecture:** 9 layers, 512-dim, 8 heads (4 KV), 2x MLP, 1024 vocab, tied embeddings
- **Quantization:** int8 + zlib

## Best Result

| Metric | Value |
|--------|-------|
| val_bpb | — |
| val_loss | — |
| artifact_size | — |
| experiment | — |
| seed | — |

## Leaderboard SOTA (for reference)

- **1.1228 BPB** — 11L EMA + GPTQ-lite + warmdown3500 (signalrush, 2026-03-22)
