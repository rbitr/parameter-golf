# Optimization Ideas

Evolving list of ideas to explore. Mark with status as you go:
- [ ] = untried
- [~] = in progress
- [x] = tried (note result)
- [-] = abandoned (note reason)

## Architecture

- [x] **Depth recurrence / weight sharing** — TRIED: shared attn+MLP weights between layers 7-10 and 3-6. RESULT: **+0.035 BPB worse** (1.159 vs 1.124). Model too small for weight sharing; every unique param matters. Gives 3.6% more steps but 35% fewer unique params is devastating.
- [ ] **Mixture of Experts (MoE)** — Sparse MLP with 2-4 experts, top-1 routing. More capacity per parameter. Need to ensure it quantizes well.
- [ ] **Linear attention variants** — Replace softmax attention in some layers with linear attention for efficiency. Could allow more layers or longer context.
- [ ] **Cross-layer attention / dense connections** — Reuse KV from earlier layers. More information flow without more parameters.
- [ ] **Wider vs deeper tradeoffs** — Systematic sweep: more layers with smaller dim vs fewer layers with larger dim. The SOTA uses 11L/512d but is that optimal?
- [ ] **Alternative activation functions** — SwiGLU, GELU variants. relu^2 is good but maybe not optimal at this scale.
- [ ] **Factored embeddings** — Low-rank embedding matrix to save parameters, especially if increasing vocab size.
- [ ] **Mixture of depths** — Skip some layers for some tokens via a learned router.

## Training

- [ ] **Curriculum learning** — Start with shorter sequences, ramp up to 2048. May help early training efficiency.
- [ ] **Alternative LR schedules** — WSD (warmup-stable-decay), cyclic, etc. Warmdown is standard but is it optimal?
- [ ] **Larger batch size** — If training is compute-bound, larger batches could help with 8 GPUs.
- [ ] **Gradient accumulation tweaks** — Trade off batch size vs sequence length.
- [ ] **Different optimizers** — SOAP, Lion, Adalayer. Muon works well but alternatives exist.
- [ ] **Data ordering** — Smart curriculum over FineWeb shards. Some data is harder/more useful than others.
- [ ] **Label smoothing** — Small amount may improve generalization.

## Quantization & Compression

- [ ] **BitNet / ternary weights** — Train with ternary or binary weights from scratch. Zero quantization gap.
- [ ] **Mixed precision per-layer** — Different bit widths for different layer types. Attention may need more precision than MLP.
- [ ] **QAT from the start** — Instead of late QAT, train with fake quantization from step 0.
- [ ] **Int4/Int5 for some tensors** — Push below int6 where possible.
- [ ] **Better compression** — zstd-22 vs brotli vs custom schemes. The compressor matters.
- [ ] **Structured sparsity** — 2:4 or 4:8 structured sparsity for better compression ratios.
- [ ] **Vocabulary size optimization** — Larger vocab = fewer tokens = potentially better BPB, but more embedding params. Find the sweet spot.
- [ ] **Knowledge distillation into quantized model** — Train large, distill into the 16MB target.

## Evaluation

- [ ] **Sliding window stride optimization** — Current SOTA uses stride=64. Is that optimal?
- [ ] **Test-time training (TTT)** — LoRA-based TTT is already on the leaderboard at 1.19. Could be combined with better base models.
- [ ] **Longer eval context** — Evaluate with context > training length via position extrapolation.
- [ ] **Ensembling within 16MB** — Multiple tiny models that vote? Probably not enough budget.

## Meta

- [ ] **Systematic leaderboard analysis** — Read every entry, build a table of techniques vs scores.
- [ ] **Ablation studies** — For known techniques, which ones actually contribute most?
- [ ] **Search the literature** — Recent papers on efficient LM training, parameter-efficient methods.
- [ ] **Analyze the baseline** — Profile where parameters are spent. What's the bottleneck?

## Priority Queue (next experiments)

1. **Reduce per-step time** — Currently 87ms/step vs SOTA's 85ms. Remove redundant zero_grad, optimize compilation. Even 2ms/step = ~150 more steps.
2. **Hyperparameter tuning** — Warmdown iters (try 3800), SWA frequency (every 25 vs 50), LR schedule tweaks.
3. **Better GPTQ quantization** — More clip percentile candidates (10 instead of 5), or full column-wise GPTQ with calibration data.
4. **Vocabulary size optimization** — Larger vocab (2048, 4096) = fewer tokens = better BPB, but more embedding params.
5. **MoE (Mixture of Experts)** — 2-4 experts with top-1 routing. More capacity per parameter. But increases artifact size.

## Results Log

| Date | Experiment | val_bpb | Artifact | Notes |
|------|-----------|---------|----------|-------|
| 2026-03-24 | sota_baseline_v4 | 1.1282 | 17.04MB (OVER) | First working run, no zstd |
| 2026-03-24 | infra_fix_v2 | 1.1233 | 16.34MB (OVER) | FA3+zstd working, ZIP format too big |
| 2026-03-24 | legacy_format | 1.1237 | 15.96MB | First valid submission! Legacy torch.save |
| 2026-03-25 | depth_recurrence_512d | 1.1591 | 10.69MB | **REGRESSED**: weight sharing -0.035 BPB. Model too small for sharing. |
