# Experiment: attn_gate_vrl_v2 — Gated Attention + Value Residual (proper init)

## Hypothesis
The previous attn_gate_vrl experiment regressed by +0.009 due to bad initialization (sigmoid(0)=0.5 halved attention). With proper initialization (attn_gate=4.0→sigmoid≈0.98, vr_lambda=-4.0→sigmoid≈0.02), the model starts from near-baseline behavior and can learn to use gates if beneficial. SOTA uses both techniques.

## Changes
1. `attn_gate = Parameter(full(num_heads, 4.0))` — per-head sigmoid gate, starts near 1.0
2. `vr_lambda = Parameter(full(1, -4.0))` — value residual gate, starts near 0.0
3. After attention: `y = y * sigmoid(attn_gate)` then `y = y + sigmoid(vr_lambda) * v_expanded`
4. Added both to CONTROL_TENSOR_NAME_PATTERNS

## Results
- **val_bpb: 1.1266** (vs 1.1168 baseline = **+0.0098 REGRESSED**)
- val_loss: 1.9022 (sliding window)
- artifact: 15,517,945 bytes (fits under 16MB, needed pruning)
- steps: 6376 (vs 6676 baseline, **4.5% fewer** due to VRL compute overhead)
- cost: $10.81
- base (pre-quant EMA) val_bpb: 1.1387

## Analysis
**The regression is NOT due to initialization — the technique itself hurts our model.**

Key evidence:
- Bad init (v1, sigmoid(0)=0.5): +0.0090 BPB worse
- Proper init (v2, near identity): +0.0098 BPB worse — essentially the SAME regression
- This means the model learned similar gate values regardless of initialization
- The 4.5% fewer steps (6376 vs 6676) from VRL compute overhead also hurts

Why this doesn't work for us but works for SOTA:
1. **Step budget**: Our model gets 6376 steps with gates (vs SOTA's ~7185). The extra parameters (99 per layer × 11 layers = 1089 params) need training that we can't afford
2. **Compute overhead**: The v_expand + gating adds ~4.5% per-step cost, costing 300 training steps
3. **Architecture mismatch**: SOTA's BigramHash at 1536×512d provides much stronger token conditioning. Their model may benefit from attention modulation that ours doesn't need
4. **XSA interaction**: Our XSA (cross-head suppression) already modulates attention. Adding sigmoid gates on top may conflict

## Key Learnings
1. Gated attention + VRL is a **dead end** for our architecture (tested twice, both +0.009-0.010 worse)
2. The initialization didn't matter — the technique fundamentally doesn't help at our scale/step-budget
3. SOTA techniques don't automatically transfer — architecture-specific interactions dominate
4. Any per-step compute overhead >2% costs hundreds of training steps, which matters at our scale

## What to Try Next
- **Structured sparsity (2:4)** to unlock lower WD (0.03 gave best BPB but was 517KB over)
- **Muon NS per-component orthogonalization** (free quality improvement if Q/K/V are concatenated)
- **SSM/Mamba hybrid** — fundamentally different from attention
