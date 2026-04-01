# Experiment: attn_gate_vrl — Gated Attention + Value Residual

## Hypothesis
The SOTA (1.1147) uses per-head sigmoid attention gates and value residual (VRL) connections that we don't have. These small architectural additions give the model more expressive control over attention behavior. Expected improvement: 0.001-0.003 BPB.

## Changes
1. Added `attn_gate = nn.Parameter(zeros(num_heads))` — per-head sigmoid gate on attention output
2. Added `vr_lambda = nn.Parameter(zeros(1))` — sigmoid-gated value residual connection
3. After attention: `y = y * sigmoid(attn_gate)` then `y = y + sigmoid(vr_lambda) * v_expanded`
4. Added both to CONTROL_TENSOR_NAME_PATTERNS for proper quantization handling

## Results
- **val_bpb: 1.1258** (vs 1.1168 baseline = **+0.0090 REGRESSED**)
- artifact: 15,523,700 bytes (fits under 16MB)
- steps: 6518 (vs 6676 baseline, 2.4% fewer due to VRL compute overhead)
- cost: $10.47

## Analysis
**The regression is due to bad initialization, not a bad technique.**

- `attn_gate` initialized at 0 → sigmoid(0) = 0.5 → attention output HALVED at initialization
- `vr_lambda` initialized at 0 → sigmoid(0) = 0.5 → 50% raw value residual added immediately
- These initial values severely damage early training dynamics:
  - The halved attention means the model's residual stream receives much weaker attention contributions
  - The 50% VRL adds unweighted value projections, interfering with the attention mechanism
  - The model needs many steps to learn appropriate gate values, wasting training budget

**Correct initialization would be:**
- `attn_gate = Parameter(full(num_heads, 4.0))` → sigmoid(4)=0.982 ≈ no gating initially
- `vr_lambda = Parameter(full(1, -4.0))` → sigmoid(-4)=0.018 ≈ no VRL initially

This way the model starts from identical behavior to the baseline and can learn to use gates if beneficial.

## Key Learnings
1. Sigmoid-gated architectural additions MUST be initialized at "no effect" values
2. sigmoid(0)=0.5 is a TERRIBLE default for multiplicative gates (halves the signal)
3. The SOTA likely initializes these gates carefully (or their training regime compensates)
4. The 158 fewer steps (6518 vs 6676) suggest ~2.4% compute overhead from VRL

## What to Try Next
- **attn_gate_vrl_v2**: Same architecture but with proper initialization (attn_gate=4.0, vr_lambda=-4.0)
- This should start from baseline behavior and only deviate if the gates learn useful patterns
- The compute overhead is minimal (99 extra params, ~2% slower)
