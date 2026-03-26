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
- [x] **EMA decay tuning** — TRIED: 0.997→0.998. RESULT: **-0.0006 BPB better** (1.1226 vs 1.1232). Previously blocked by artifact size (zstd), unblocked by brotli compression. **NEW BEST**.
- [x] **EMA+SWA blend** — TRIED: blend averaging at various alphas. RESULT: Pure EMA is best at scale (alpha=1.0). SWA adds no value AND costs ~1ms/step. Disabling SWA + EMA 0.998 = optimal.
- [x] **Brotli compression** — TRIED: brotli quality 10 vs zstd-22. RESULT: **Saves 380-645KB** across all models. Key unlock for EMA 0.998. Decompression <1s.
- [ ] **Alternative LR schedules** — WSD (warmup-stable-decay), cyclic, etc. Warmdown is standard but is it optimal?
- [ ] **Larger batch size** — If training is compute-bound, larger batches could help with 8 GPUs.
- [ ] **Gradient accumulation tweaks** — Trade off batch size vs sequence length.
- [ ] **Different optimizers** — SOAP, Lion, Adalayer. Muon works well but alternatives exist.
- [ ] **Data ordering** — Smart curriculum over FineWeb shards. Some data is harder/more useful than others.
- [x] **Label smoothing** — TRIED: epsilon=0.02. RESULT: **+0.0212 BPB worse** (1.1444 vs 1.1232). Devastating at this model size — model too small to waste capacity softening targets.

## Quantization & Compression

- [ ] **BitNet / ternary weights** — Train with ternary or binary weights from scratch. Zero quantization gap.
- [x] **Mixed precision per-layer (int5 MLP)** — TRIED: int5 for MLPs, int6 for attention. RESULT: **+0.019 BPB worse**. Int5 quantization gap is devastating. Not viable at this model size.
- [x] **Fix dead-coded QAT** — TRIED 5 times: QAT + 5 clips gives best BPB (1.1230) but artifact 80KB over 16MB. QAT + 10 clips fits under 16MB (15.93MB) but BPB is 1.1234 (worse than no-QAT best 1.1232). The GPTQ clip count interacts with QAT. Need to try 7-8 clips or find other ways to save ~100KB.
- [-] **QAT from the start** — Abandoned: too expensive (model can't learn with full quantization noise from step 0).
- [-] **Int4/Int5 for some tensors** — Abandoned: int5 MLP experiment showed quantization error is too large.
- [-] **Aggressive post-quant pruning** — Abandoned: zeroing all ±1 quantized values saves 1.82MB but destroys BPB (+0.02). Need magnitude-aware pruning but code complexity is high.
- [ ] **Better compression** — zstd-22 vs brotli vs custom schemes. The compressor matters.
- [ ] **Structured sparsity** — 2:4 or 4:8 structured sparsity for better compression ratios.
- [ ] **Magnitude-aware pruning** — Prune by |q * scale| instead of |q|. Could save 100KB with minimal BPB impact.
- [ ] **Vocabulary size optimization** — Larger vocab = fewer tokens = potentially better BPB, but more embedding params. Find the sweet spot.
- [ ] **Knowledge distillation into quantized model** — Train large, distill into the 16MB target.

## Evaluation

- [x] **Sliding window stride optimization** — TRIED stride=32 vs stride=64. RESULT: Only 0.00003 BPB difference. Not worth the 2x eval time. Stride=64 is optimal.
- [ ] **Test-time training (TTT)** — LoRA-based TTT is already on the leaderboard at 1.19. Could be combined with better base models.
- [ ] **Longer eval context** — Evaluate with context > training length via position extrapolation.
- [ ] **Ensembling within 16MB** — Multiple tiny models that vote? Probably not enough budget.

## Meta

- [ ] **Systematic leaderboard analysis** — Read every entry, build a table of techniques vs scores.
- [ ] **Ablation studies** — For known techniques, which ones actually contribute most?
- [ ] **Search the literature** — Recent papers on efficient LM training, parameter-efficient methods.
- [ ] **Analyze the baseline** — Profile where parameters are spent. What's the bottleneck?

## Priority Queue (next experiments)

1. **QAT + 5-7 clips with brotli** — QAT+7clips gave 1.1231 before (zstd, EMA 0.997). With brotli + EMA 0.998 + no SWA, expect ~1.1224. 533KB headroom means 5 clips should easily fit.
2. **EMA 0.999** — Even broader averaging. With brotli, the wider distributions fit. Risk: may need more training steps.
3. **Speed optimization** — Still 55 steps behind SOTA (7046 vs 7101). Profile ms/step gap.
4. **MoE (Mixture of Experts)** — 2-4 experts with top-1 routing. More capacity per parameter. Medium risk.
5. **Vocabulary size optimization** — Larger vocab (2048, 4096) = fewer tokens = better BPB, but more embedding params.

## Key Findings

### EMA decay tradeoff (UPDATED)
- EMA 0.997 + zstd: BPB 1.1232, 15.7-15.8MB (old best)
- EMA 0.998 + zstd: BPB 1.1229, 16.0-16.7MB (over 16MB)
- **EMA 0.998 + brotli: BPB 1.1226, 15.4-15.5MB (NEW BEST, under 16MB!)**
- Brotli compression eliminates the EMA 0.998 size penalty
- SWA disabled saves ~1ms/step → ~50 more training steps

### Compression: brotli vs zstd (NEW)
- brotli quality=10 beats zstd-22 by 380-645KB across all models
- Especially effective on EMA 0.998 weights (broader distributions)
- Compression: ~40s on trained model, decompression: <1s
- `pip install brotli` required on eval machine

### GPTQ clips tradeoff (clear trend)
- 5 clips: ~16.01MB model, 1.1230 BPB (best BPB, worst compression)
- 7 clips: ~15.96MB model, 1.1231 BPB
- 10 clips: ~15.79MB model, 1.1232 BPB (worst BPB, best compression)
- More clips = tighter clips = narrower distributions = better compression but slightly worse BPB
- Artifact size has ~400KB variance between runs regardless of clips

### QAT + GPTQ clips interaction
- QAT + 5 clips: 1.1230 BPB, 16.08MB (best BPB, over 16MB)
- QAT + 10 clips: 1.1234 BPB, 15.93MB (fits, but worse BPB)
- No QAT + 10 clips: 1.1232 BPB, 15.79MB (current best valid)
- 10 clips may pick suboptimal clip points for QAT-trained weights
- Try 7-8 clips as middle ground

### Sliding window stride
- stride=32 vs stride=64: only 0.00003 BPB difference
- Not worth the 2x eval time. stride=64 is optimal.

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
| 2026-03-25 | ste_qat_in_forward | 1.1233 | 16.01MB | **FAILED**: STE QAT in forward (SOTA approach). Only +0.22MB over vs +0.46MB, but still 7KB over 16MB. |
| 2026-03-25 | match_sota_qat_cleanup | 1.1230 | 16.08MB | **BEST BPB** but artifact 80KB over. QAT + 5 clips + code cleanup. |
| 2026-03-25 | qat_10clip_codeclean | 1.1236 | 16.14MB | **FAILED**: 10 clips + QAT worse than 5 clips + QAT. |
| 2026-03-25 | qat_prune_fallback | 1.1427 | 14.65MB | **REGRESSED**: Pruning all ±1 quant values saves 1.82MB but costs 0.02 BPB. Way too aggressive. |
| 2026-03-25 | qat015_10clips_legacy | 1.1234 | 15.93MB | QAT + 10 clips fits under 16MB but BPB worse than best (1.1232). Stride=32 vs 64: negligible. |
| 2026-03-26 | qat015_7clips_restore | 1.1231 | 16.03MB | **BEST BPB** but 32KB over 16MB. 7 clips: better BPB, worse compression than 10 clips. |
| 2026-03-26 | ttt_lora_eval | 1.2669 | 16.01MB | **REGRESSED**: TTT LoRA eval much worse. eval_seq=1024 < train_seq=2048 loses context. |
| 2026-03-26 | label_smoothing_002 | 1.1444 | 15.83MB | **REGRESSED**: Label smoothing epsilon=0.02 devastating (+0.021 BPB). Model too small. |
| 2026-03-26 | brotli_ema098 | 1.1246 | 15.47MB | **REGRESSED**: Brotli works but SWA enabled → 6940 steps (too few). |
| 2026-03-26 | brotli_ema098_noswa | **1.1226** | 15.47MB | **NEW BEST! BEATS SOTA!** Brotli-10 + EMA 0.998 + SWA disabled = 7046 steps. |
