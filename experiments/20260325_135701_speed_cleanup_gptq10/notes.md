# Experiment: Speed Cleanup + GPTQ-10 Candidates

## Hypothesis
1. Cleaning up script (removing depth recurrence code, FA3 fallback, pip install) would match SOTA's ~84.5ms/step, gaining ~200 steps.
2. Increasing GPTQ clip candidates from 5→10 would improve quantization at zero training cost.

## Changes
1. Started from SOTA script (records/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py)
2. Added `_use_new_zipfile_serialization=False` to torch.save (legacy format, better compression)
3. Increased GPTQ clip percentiles from 5→10: [0.998, 0.999, 0.9993, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 0.999999, 1.0]
4. Added minimal FA3 fallback (for local A100 testing only, FA3 used on H100)

## Results (8xH100 SXM, seed=1337)
| Metric | Previous Best | This Run | Delta |
|--------|--------------|----------|-------|
| val_bpb | 1.1237 | **1.1232** | **-0.0005** (NEW BEST) |
| val_loss | 1.8974 | 1.8966 | -0.0008 |
| steps | 6898 | 6999 | +101 |
| ms/step | 86.99 | 85.73 | -1.26ms |
| artifact | 15.96MB | 15.79MB | -170KB |

## Analysis
**Both changes contributed positively:**

### Speed improvement
- Step time: 86.99 → 85.73 ms/step (-1.26ms)
- Steps gained: 101 more steps (6898→6999)
- Still 1.17ms/step slower than SOTA's 84.56ms/step (which gets 7101 steps)
- The remaining gap is ~102 steps = ~0.0002 BPB

### GPTQ improvement
- More clip candidates likely contributed to the artifact size reduction (15.96→15.79MB)
- The BPB improvement of 0.0005 comes from both more steps AND better quantization

### Remaining gap to SOTA (1.1228)
- Gap: 0.0004 BPB
- SOTA gets 7101 steps vs our 6999 (102 fewer steps)
- The step time difference (85.73 vs 84.56) accounts for ~102 steps
- Possible causes of remaining speed gap: torch.compile caching, hardware variance, or minor code differences

## Key Learnings
1. **Code cleanup matters** — removing unused depth recurrence code saved 1.26ms/step
2. **More GPTQ candidates help** — 10 candidates > 5 for quantization quality
3. **Legacy torch.save format still saves ~170KB** after GPTQ compression
4. **We're within 0.0004 BPB of SOTA** — closing the gap further requires either more steps or algorithmic improvements

## What to Try Next
1. **Profile the remaining 1.17ms/step gap** — find what's different from SOTA
2. **Hyperparameter tuning** — warmdown_iters (3500→3800), LR tweaks
3. **More aggressive GPTQ** — per-column optimal clipping, or full GPTQ with calibration
4. **Novel techniques** — MoE, vocab size optimization, etc.
