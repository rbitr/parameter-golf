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
- [x] **Warmdown iters tuning** — TRIED: 3500→3800. RESULT: **+0.0002 BPB worse**, artifact OVER 16MB. 3500 is near-optimal; 54% warmdown is too much.
- [x] **EMA decay tuning** — TRIED: 0.997→0.998. RESULT: **-0.0003 BPB better** but artifact OVER 16MB (16.0-16.7MB). 0.997 is optimal for 16MB budget.
- [x] **EMA+SWA blend** — TRIED: blend averaging at various alphas. RESULT: Pure EMA is best at scale (alpha=1.0). SWA adds no value.
- [ ] **Alternative LR schedules** — WSD (warmup-stable-decay), cyclic, etc. Warmdown is standard but is it optimal?
- [ ] **Larger batch size** — If training is compute-bound, larger batches could help with 8 GPUs.
- [ ] **Gradient accumulation tweaks** — Trade off batch size vs sequence length.
- [ ] **Different optimizers** — SOAP, Lion, Adalayer. Muon works well but alternatives exist.
- [ ] **Data ordering** — Smart curriculum over FineWeb shards. Some data is harder/more useful than others.
- [ ] **Label smoothing** — Small amount may improve generalization.

## Quantization & Compression

- [ ] **BitNet / ternary weights** — Train with ternary or binary weights from scratch. Zero quantization gap.
- [x] **Mixed precision per-layer (int5 MLP)** — TRIED: int5 for MLPs, int6 for attention. RESULT: **+0.019 BPB worse**. Int5 quantization gap is devastating. Not viable at this model size.
- [x] **Fix dead-coded QAT** — TRIED: weight-replacement QAT (outside compiled graph). RESULT: Quant gap reduced 34% (0.0083→0.0055 BPB) but artifact grew +0.46MB (16.25MB, OVER). QAT weights compress poorly. Threshold=0.15 is too aggressive; try 0.05 or combine with tighter GPTQ.
- [-] **QAT from the start** — Abandoned: too expensive (model can't learn with full quantization noise from step 0).
- [-] **Int4/Int5 for some tensors** — Abandoned: int5 MLP experiment showed quantization error is too large.
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

1. **Fix dead-coded QAT** — IN PROGRESS. Weight-replacement QAT avoids graph recompilation. Quant gap is 0.0083 BPB; reducing by 20-30% gives 0.0017-0.0025 BPB — much bigger than 0.0004 gap to SOTA.
2. **Speed optimization** — SOTA gets 7101 steps vs our 6999. Each additional step matters. Profile where time is spent.
3. **MoE (Mixture of Experts)** — 2-4 experts with top-1 routing. More capacity per parameter. Medium risk.
4. **Vocabulary size optimization** — Larger vocab (2048, 4096) = fewer tokens = better BPB, but more embedding params.
5. **Sliding window stride tuning** — Currently stride=64. Is that optimal?

## Key Findings

### EMA decay tradeoff
- EMA 0.997: better compression (15.7-15.8MB model), BPB 1.1232
- EMA 0.998: worse compression (16.0-16.7MB model), BPB 1.1229
- The wider window produces broader weight distributions that compress poorly
- EMA 0.997 is optimal for 16MB budget

### GPTQ clips tradeoff
- More clips (10) = better MSE AND often better compression (tighter clips produce narrower distributions)
- Fewer clips (5) = can actually be LARGER (wider minimum clip keeps more outliers)
- Artifact size has ~400KB variance between runs regardless of clips

### Artifact size variance
- Same configuration can produce 15.7-16.2MB artifacts across runs
- Varies with step count (hardware speed) and training randomness
- Need ~200KB headroom for reliable submissions

## Results Log

| Date | Experiment | val_bpb | Artifact | Notes |
|------|-----------|---------|----------|-------|
| 2026-03-24 | sota_baseline_v4 | 1.1282 | 17.04MB (OVER) | First working run, no zstd |
| 2026-03-24 | infra_fix_v2 | 1.1233 | 16.34MB (OVER) | FA3+zstd working, ZIP format too big |
| 2026-03-24 | legacy_format | 1.1237 | 15.96MB | First valid submission! Legacy torch.save |
| 2026-03-25 | depth_recurrence_512d | 1.1591 | 10.69MB | **REGRESSED**: weight sharing -0.035 BPB. Model too small for sharing. |
| 2026-03-25 | speed_cleanup_gptq10 | 1.1232 | 15.79MB | **NEW BEST**: +101 steps from cleanup, 10 GPTQ clips. -0.0005 BPB. |
| 2026-03-25 | warmdown3800 | 1.1234 | 16.28MB | **REGRESSED**: +0.0002 BPB worse, artifact OVER 16MB. 3500 is optimal. |
| 2026-03-25 | ema_swa_blend | 1.1232 | 16.20MB | **FAILED**: Artifact over 16MB. EMA+SWA blend doesn't help at scale (pure EMA best). |
| 2026-03-25 | ema_decay_0998 | 1.1229 | 16.10MB | **FAILED**: Best BPB yet but artifact over 16MB. EMA 0.998 compresses poorly. |
| 2026-03-25 | ema098_gptq5 | 1.1232 | 15.92MB | Under 16MB but BPB same as EMA 0.997 + 10 clips. 5 clips alone doesn't improve. |
| 2026-03-25 | ema098_adaptive_gptq | 1.1233 | 16.81MB | **FAILED**: Both 10-clip and 5-clip over 16MB. 5-clip LARGER than 10-clip! |
| 2026-03-25 | int5_mlp_ema098 | 1.1427 | 13.78MB | **REGRESSED**: Int5 MLP quant gap devastating (+0.019 BPB). |
| 2026-03-25 | working_qat_fix | 1.1236 | 16.25MB | **FAILED**: QAT reduced quant gap 34% but artifact +0.46MB over 16MB. |
