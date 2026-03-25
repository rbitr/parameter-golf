# Experiment: Depth Recurrence (Weight Sharing)

## Hypothesis
Sharing heavy weights (attention Q/K/V/O + MLP fc/proj) between decoder layers 7-10 and mid layers 3-6 would reduce unique parameter count, yielding:
1. Faster optimizer steps (fewer unique params in Muon Newton-Schulz)
2. More training steps in 600s budget
3. Smaller artifact (3MB vs 16MB) → freed budget for future model expansion

## Changes
- Added `depth_recurrence` flag and `share_layers` mapping (default: "7:3,8:4,9:5,10:6")
- After model init, shared weight Parameter references between specified layer pairs
- Added state_dict deduplication in quantization export/import
- Per-layer adapter params (attn_scale, mlp_scale, resid_mix, q_gain, norms, XSA flags) remain unique per virtual layer

## Results (8xH100 SXM, seed=1337)
| Metric | Baseline | Depth Recurrence | Delta |
|--------|----------|-----------------|-------|
| val_bpb | 1.1237 | **1.1591** | **+0.035** (WORSE) |
| val_loss | 1.8974 | 1.9570 | +0.060 |
| steps | 6898 | 7146 | +248 (3.6% more) |
| ms/step | 87 | 84 | -3.4% (faster) |
| artifact | 15.96MB | 10.69MB | -5.27MB |
| unique params | 27.0M | ~17.5M | -35% |

## Analysis
**Depth recurrence HURTS significantly at this model scale.**

The 35% reduction in unique parameters causes a 0.035 BPB regression that far exceeds the benefit of 3.6% more training steps. The model is too small for weight sharing to work — every unique parameter matters.

### Why the local test was misleading
- Local test (90s, 1 GPU): depth recurrence got 62 steps vs baseline's 39, with lower loss at each step
- This was misleading because the short training favored the faster model. At convergence on 600s/8GPU, the capacity reduction dominates.

### Per-step speedup analysis
On 8xH100, the Muon optimizer speedup from fewer unique params is modest: ~3ms/step (87→84ms). This translates to only 248 extra steps (~0.5% more tokens). Nowhere near enough to compensate for 35% fewer unique parameters.

### Why it might work at larger scales
At larger model scales (e.g., 1B+ params), weight sharing between layers has been shown to work well (ALBERT, Universal Transformer). The intuition is that layers learn similar functions, and sharing helps regularization. At our scale (~27M params), the model isn't over-parameterized enough for sharing to help.

## Key Learnings
1. **Weight sharing doesn't work at 27M param scale** — the model needs all its unique parameters
2. **Step time improvement is real but modest on 8xH100** (3.4% faster)
3. **The freed artifact budget is large (5MB)** — if a future approach can maintain quality while reducing params, int8 quantization could help
4. **Local short-run tests can be misleading for convergence behavior** — need to be cautious about extrapolating

## What to Try Next
1. **Abandon depth recurrence** at this model scale
2. **Focus on step time optimization** without reducing params (code optimization, compiler flags)
3. **Hyperparameter tuning** (warmdown_iters, LR schedule, SWA frequency)
4. **Better post-training quantization** (more GPTQ-lite candidates, GPTQ-full)
5. **Novel approaches** that don't reduce effective param count
