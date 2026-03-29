# Current Best

## Best Result

| Metric | Value |
|--------|-------|
| val_bpb | 1.1198 (with TTT) / 1.1204 (base, no TTT) |
| val_loss | 1.8907 (TTT) / 1.8918 (base) |
| artifact_size | 15,561,305 bytes (under 16MB, 439KB headroom) |
| experiment | 20260328_180704_ttt_2ep_lr0005 |
| base_experiment | 20260327_212743_leaky_relu_05_squared |
| seed | 1337 |
| steps | 6943 |
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
- **TTT: SGD lr=0.0005, 2 epochs, 32K chunks, freeze_blocks=0, anchor_alpha=0.0**

## TTT Status
- TTT is saturated at -0.0007 to -0.0012 for our architecture
- 2 epochs at lr=0.0005: delta=-0.0007 (same as 1ep+anchor, worse than 1ep no-anchor)
- 1 epoch at lr=0.0005: delta=-0.0012 (best delta)
- Anchor alpha=0.0003: delta=-0.0007 (no benefit)
- **CORRECTED: SOTA TTT delta is only -0.0004** (from ablation table). The -0.0021 was LeakyReLU ablation, not TTT.
- Our TTT (-0.0012) is actually BETTER than SOTA's (-0.0004)
- **Focus on base model improvements, TTT is capped**

## Previous Innovation: LeakyReLU(0.5)²
- `torch.relu(x)` → `F.leaky_relu(x, 0.5)` in MLP
- -0.0019 BPB improvement over ReLU² baseline

## Previous Innovation: Brotli Compression
- brotli quality=10 saves ~645KB vs zstd-22
- Key unlock: EMA 0.998 always gave better BPB but couldn't fit under 16MB with zstd

## Leaderboard SOTA (for reference)
- **1.1194 BPB** — LeakyReLU² + Legal TTT + Parallel Muon (2026-03-23)
- Our gap: +0.0004 BPB (down from +0.0006)
- **CORRECTED SOTA ablation**: LeakyReLU=-0.0021, BigramHash=-0.0009, TTT=-0.0004
- SOTA base (sliding window): ~1.1198, our base: ~1.1207. Gap is -0.0009 = BigramHash expansion
- SOTA TTT delta: -0.0004, our TTT delta: -0.0012 (we're better!)
- SOTA key differences: Parameter Banking (83.3ms/step, +200 steps), BigramHash 1536d (full dim, -0.0009), EMA 0.997+SWA, GPTQ+lzma

## Dead Ends (confirmed no improvement)
- Grouped int6 quantization (G=128): -0.0001 BPB (noise). Per-row already optimal.
- eval_seq_len=4096: CATASTROPHIC (1.5502 BPB). RoPE 4x extrapolation fails.
- eval_seq_len > 2048: Don't try without training at that length.

## Infrastructure Notes
- RunPod template has FA3 pre-installed (no pip install needed)
- brotli needs `pip install --break-system-packages brotli`
- Legacy torch.save format (`_use_new_zipfile_serialization=False`) compresses better with brotli
- SSH timeout increased to 1800s to accommodate TTT eval time
