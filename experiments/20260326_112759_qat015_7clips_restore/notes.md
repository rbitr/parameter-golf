# Experiment: QAT + 7 GPTQ clips (restore best config)

## Hypothesis
QAT + 5 clips gave best BPB (1.1230) but was 80KB over 16MB. QAT + 10 clips fit (15.93MB) but BPB was 1.1234. 7 clips should be the sweet spot — better BPB than 10 clips while fitting under 16MB.

## Changes
- Restored SWA=1, QAT threshold=0.15, eval_stride=64 (working copy had drifted to SWA=0, QAT=0.0, stride=32)
- Changed GPTQ clips from 10 to 7: [0.999, 0.9995, 0.9998, 0.9999, 0.99995, 0.99999, 1.0]

## Results
- val_bpb: **1.1231** (best BPP ever, -0.0001 from current best)
- val_loss: 1.8964
- steps: 7003 (85.69 ms/step)
- model size: 15,964,932 bytes
- code size: 67,608 bytes
- **total artifact: 16,032,540 bytes (32KB OVER 16MB limit) — INVALID**

## Analysis
- 7 clips gives slightly better BPB than 10 clips (1.1231 vs 1.1232) but LARGER model
- This confirms the counterintuitive finding: more GPTQ clips = better compression
  - 10 clips: 15.79MB model, 1.1232 BPB
  - 7 clips: 15.96MB model, 1.1231 BPB
  - 5 clips: ~16.01MB model, 1.1230 BPB
- More clips search more aggressively, finding tighter clips that produce narrower distributions → better compression
- Fewer clips pick wider clips → more outliers preserved → worse compression

## Key finding: clip count trades BPB vs artifact size
- Each clip value removed costs ~0.0001 BPP but adds ~80-100KB to artifact
- At 10 clips: fits comfortably (15.79MB) but BPP ceiling
- At 5 clips: best BPB (1.1230) but 80KB+ over

## What to try next
1. **Label smoothing with 10 clips** — stay with proven-to-fit config, improve BPB via training
2. **Code size reduction** — trim ~500+ bytes of dead code to give more room
3. **More GPTQ clips (15-20)** — even better compression, sacrificing a tiny bit of BPB
4. **Speed optimization** — SOTA gets 7101 steps vs our 7003. More steps = better BPB.
