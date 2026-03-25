# Experiment: Match SOTA Config (QAT + Code Cleanup)

## Hypothesis
Enabling late QAT (threshold=0.15) + cleaning up code (direct FA3 import pattern, remove 10-clip GPTQ fallback, strip comments) would match SOTA's 1.1228 BPB. QAT makes weights more quantization-friendly, code reduction + legacy format keeps artifact under 16MB.

## Changes
- `late_qat_threshold=0.15` (was 0.0)
- Removed GPTQ_CLIPS_10/GPTQ_CLIPS_5 constants and fallback loop
- Used 5 GPTQ clips inline (matching SOTA)
- Cleaned up comments, added `_HAS_FA3` constant for torch.compile
- Kept legacy `torch.save` format

## Results
- **val_bpb: 1.1230** (NEW BEST BPB but artifact over)
- **artifact: 16,080,252 bytes** (80KB OVER 16MB limit)
- **steps: 7048** (vs 6999 best, vs 7101 SOTA)
- Pre-quant BPB: 1.1230 sliding window, 1.1469 roundtrip

## Analysis
- BPB improved from 1.1232 to 1.1230 with QAT — the quantization gap reduction is real
- But artifact is 80KB over 16MB — QAT produces weights that compress worse than SOTA's
- SOTA gets 15.55MB with same config — 530KB smaller. The gap is NOT explained by code size (only ~45 bytes different)
- 5-clip GPTQ produced 16.08MB; previous 10-clip QAT run was 16.01MB (7KB over)
- The weight compression gap (~500KB vs SOTA) remains unexplained

## Key Learnings
1. QAT improves BPB by ~0.0002 (1.1230 vs 1.1232)
2. Our QAT weights compress ~500KB worse than SOTA's — this is the core blocker
3. 5 GPTQ clips + QAT: 16.08MB; 10 clips + QAT: varies 16.01-16.14MB
4. Step count: 7048 (85.1ms/step) vs SOTA 7101 (84.5ms/step) — we're still 0.6ms/step slower
