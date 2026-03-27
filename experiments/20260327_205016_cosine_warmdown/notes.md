# Experiment: Cosine Warmdown Schedule

## Hypothesis
Cosine warmdown (maintaining higher LR longer, then dropping sharply at the end) would improve convergence compared to linear warmdown. Well-established technique in many training settings.

## Change
Replaced linear warmdown `scale = remaining/total` with cosine `scale = 0.5*(1+cos(pi*progress))`.

## Results
- **val_bpb: 1.1236** (+0.001 vs 1.1226 best) — REGRESSED
- Steps: 7034 (vs 7046 linear)
- QAT triggered at step 6146 (later than linear ~5900 due to cosine staying higher longer)
- Artifact: 15.69MB (under 16MB)
- Post-EMA val_bpb: 1.1400

## Analysis
Cosine warmdown hurt performance. The linear schedule works better for Muon optimizer at this model size.

Possible reasons:
1. Muon optimizer benefits from gradual, steady LR decay — the prolonged high LR from cosine may cause instability
2. Later QAT trigger (6146 vs ~5900) means ~250 fewer QAT training steps, potentially larger quant gap
3. The sharp drop at the end of cosine may not give EMA enough time to smooth

## Key Learning
- Linear warmdown is optimal for this Muon+EMA combination
- Don't try other warmdown shapes (they're all likely to hurt)
- The LR schedule is already well-tuned

## Reverted
Working script reverted to linear warmdown.
