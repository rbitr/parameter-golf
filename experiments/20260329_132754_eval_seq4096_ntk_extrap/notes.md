# Experiment: Eval with 4096 Context (NTK RoPE Extrapolation)

## Hypothesis
With partial RoPE (16/64 dims = 25% positional, 75% content-based attention), the model might benefit from longer eval context (4096 tokens) via NTK-aware RoPE frequency scaling. The Rotary train_seq_len is 1024, so the model already extrapolates 2x during training (1024→2048). Going to 4x (1024→4096) might be feasible.

## Changes
- Changed eval_seq_len default from 2048 to 4096
- Training completely unchanged (still trains with 2048)
- RoPE NTK scaling: new_base = 10000 * (4096/1024)^(16/14) = ~46,600

## Results
| Metric | eval_seq=4096 | eval_seq=2048 (prev) | Delta |
|--------|--------------|---------------------|-------|
| DIAGNOSTIC (2048) | 1.1373 | ~1.1372 | ~same |
| Roundtrip (full-seq) | 1.1564 | ~1.1440 | +0.012 |
| **Sliding window** | **1.5502** | **1.1203** | **+0.430** |
| TTT (2048) | 1.1201 | 1.1198 | +0.0003 |
| Steps | 6933 | ~6964 | -31 |

## Analysis
- **Sliding window at 4096 is CATASTROPHIC**: 1.5502 BPB (random-level for positions >2048)
- RoPE NTK extrapolation from 1024→4096 (4x) fails completely
- Full-seq roundtrip at 4096 is also degraded (+0.012 BPB) but not as extreme
- TTT result (1.1201) is unaffected because TTT uses train_seq_len=2048
- The slight TTT regression (1.1201 vs 1.1198) is from fewer training steps (6933 vs 6964)
- **Conclusion: 4x RoPE extrapolation with NTK scaling does not work for this model. Even partial RoPE (25% of dims) is not enough to enable 4x extrapolation.**

## Key Learning
- RoPE NTK scaling from 1024→2048 (2x, during training) works fine
- RoPE NTK scaling from 1024→4096 (4x, at eval) fails catastrophically
- Position-independent attention dims (75%) are NOT enough to compensate for broken positional encoding
- Don't try eval_seq_len > 2048 without training at that length
