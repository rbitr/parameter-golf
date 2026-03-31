# Experiment: Untied Embeddings (head_lr=0.04)

## Hypothesis
With tied embeddings, a single 1024×512 matrix serves dual roles — input embedding and output logit projection. These have different optimal representations. Untying lets each specialize. The output head is the most gradient-rich parameter, so it should train well in ~6700 steps.

## Changes
- `tie_embeddings = 0` (separate lm_head)
- `head_lr = 0.04` (increased from 0.008, closer to tied_embed_lr=0.035)
- `embed_lr = 0.6` (default for untied)

## Results
| Metric | Current Best | This Run | Delta |
|--------|-------------|----------|-------|
| val_bpb | 1.1168 | **1.1376** | **+0.0208** |
| val_loss | 1.8856 | 1.9209 | +0.0353 |
| artifact_size | 15.54MB | 15.91MB | +0.37MB |
| steps | 6676 | 6675 | -1 |

## Analysis
**Catastrophic regression.** Untied embeddings cost +0.0208 BPB — one of the worst results in the project.

Key findings:
1. **Tied embedding constraint is beneficial, not harmful.** The shared gradient from both input embedding lookups and output logit computation provides richer training signal. With only ~6700 training steps, 524K extra untrained parameters are severely undertrained.
2. **Artifact size increased by ~370KB** as expected (separate int8 lm_head matrix).
3. **Step count unchanged** — the extra parameters add negligible compute overhead.
4. **The inductive bias of "input ≈ output" helps small models.** At this scale (18M params, 6700 steps), the constraint acts as useful regularization. The model can't afford to learn two independent representations.

## What This Tells Us
- Don't add parameters unless they get sufficient gradient. BigramHash expansion, VE 3-layer, and now untied embeddings all regressed for the same reason: undertrained params.
- The tied embedding is optimal for this training budget. Don't revisit.
- Focus should be on techniques that DON'T add parameters: better training dynamics, architecture changes that rearrange existing params, better quantization.

## Next Steps
- Try techniques that improve quality without adding params
- Cross-layer KV sharing (rearranges params, doesn't add)
- Structured sparsity (reduces effective params, enables other changes)
- Or completely different architectures (SSM/Mamba)

## Cost
$11.90 (RunPod)
