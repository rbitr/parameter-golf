# Experiment: Legacy Format Fix (Infrastructure Validation)

## Hypothesis
Fix artifact size issue by switching from torch.save's ZIP format to legacy pickle format, which compresses ~645KB better with zstd-22. Also validates full infrastructure fixes (FA3 + zstd on RunPod).

## Changes
1. `torch.save(..., _use_new_zipfile_serialization=False)` — legacy format compresses better
2. RunPod setup: `--break-system-packages` for pip install, per-command timeouts
3. TRAINING_TIMEOUT increased from 900s to 1200s to handle training + eval

## Results
- **val_bpb: 1.1237** (sliding window stride=64) — close to SOTA 1.1228
- **val_loss: 1.8974**
- **artifact: 15,957,740 bytes — UNDER 16MB** (42KB headroom)
- Steps: 6898 in 600s (~87ms/step)
- FA3 working (vs 99ms/step without)
- zstd-22 working
- Cost: $6.47

## Analysis
- The legacy torch.save format produces data that zstd compresses ~645KB better than the ZIP format. Same logical content, different byte layout.
- 6898 steps (vs SOTA's 7101) is ~3% fewer, likely explaining the 0.0009 BPB gap.
- Step variance likely comes from torch.compile overhead or pod hardware differences.
- All SOTA techniques working: GPTQ-lite, EMA, late QAT, XSA, partial RoPE, etc.

## Key Learnings
- PyTorch's new zipfile serialization format is WORSE for external compression (zstd). The ZIP headers break zstd's compression patterns.
- The RunPod parameter-golf template has FA3 pre-installed but NOT zstandard. Need --break-system-packages for pip.
- flash-attn is available as pre-built on the template, no compilation needed.

## Next Steps
- Try to get more training steps (reduce per-step overhead or extend training)
- Explore novel improvements beyond the SOTA baseline
- Consider depth recurrence, larger vocab, or MoE for better BPB
