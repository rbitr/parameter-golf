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
| val_bpb | 1.1282 |
| val_loss | 1.9049 |
| artifact_size | 17,040,647 bytes (OVER LIMIT - zlib; ~15.9MB with lzma fallback) |
| experiment | 20260324_143151_sota_baseline_v4 |
| seed | 1337 |

**Note:** Artifact is over 16MB due to zlib compression (zstandard not on RunPod).
lzma fallback added — should bring it to ~15.9MB. Needs validation on RunPod.
Also running ~99ms/step (vs SOTA 85ms/step) due to missing FA3 — ~1000 fewer steps.

## Leaderboard SOTA (for reference)

- **1.1228 BPB** — 11L EMA + GPTQ-lite + warmdown3500 (signalrush, 2026-03-22)
