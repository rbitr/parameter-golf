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
- [x] **Alternative activation functions** — TRIED: LeakyReLU(0.5)²: **-0.0019 BPB** (1.1207). LeakyReLU(0.7)²: **+0.0035 BPB** (1.1242, worse than ReLU!). Optimum near 0.5. Slope² determines negative weight: 0.5→0.25 is the sweet spot. Could try 0.3 but unlikely to beat 0.5.
- [ ] **Factored embeddings** — Low-rank embedding matrix to save parameters, especially if increasing vocab size.
- [ ] **Mixture of depths** — Skip some layers for some tokens via a learned router.
- [x] **SwiGLU MLP activation** — TRIED: SwiGLU (gate+up+down at matched params, hidden=1024 vs 1536). RESULT: **+0.007 BPB worse** locally at 100 steps. The 2/3 width reduction hurts more than gating helps at this model size. LeakyReLU(0.5)² is optimal.

## Training

- [ ] **Curriculum learning** — Start with shorter sequences, ramp up to 2048. May help early training efficiency.
- [x] **Warmdown iters tuning** — TRIED: 3500→3800: **+0.0002 BPB worse**, artifact OVER 16MB. 3500→3000 (SOTA config): **+0.0004 BPB worse** (1.1211 base), TTT delta halved (-0.0006 vs -0.0012). EMA 0.998 needs longer warmdown. **3500 is optimal. Fully characterized: 3000 < 3500 > 3800.**
- [x] **EMA decay tuning** — TRIED: 0.997→0.998: **-0.0006 BPB better** (1.1226). 0.999: **+0.0067 BPB worse** (1.1293). Optimum is 0.998. Not monotonic — 0.999 averages too much outdated history.
- [x] **EMA+SWA blend** — TRIED: blend averaging at various alphas. RESULT: Pure EMA is best at scale (alpha=1.0). SWA adds no value AND costs ~1ms/step. Disabling SWA + EMA 0.998 = optimal.
- [x] **Brotli compression** — TRIED: brotli quality 10 vs zstd-22. RESULT: **Saves 380-645KB** across all models. Key unlock for EMA 0.998. Decompression <1s.
- [x] **Alternative LR schedules** — TRIED: Cosine warmdown (vs linear). RESULT: **+0.001 BPB worse** (1.1236). Muon optimizer prefers linear decay. Don't try other warmdown shapes.
- [x] **Smaller batch size (524K)** — TRIED: 524K tokens (vs 786K). RESULT: **+0.003 BPB worse** (1.1256). Got 10,034 steps (42% more) but noisier gradients devastated Muon optimizer. 786K is optimal.
- [-] **Larger batch size** — Abandoned: smaller batch already regressed; larger would give too few steps. 786K is the sweet spot.
- [-] **Gradient accumulation tweaks** — Abandoned: batch size experiment shows 786K is well-optimized.
- [x] **Muon weight decay tuning** — TRIED: WD=0.02 (half of 0.04). RESULT: **-0.0021 BPB (1.1186 vs 1.1207) but artifact 17.58MB (OVER 16MB)**. Lower WD = larger weights = worse compression. Need WD ~0.03-0.035 sweet spot. **HIGH PRIORITY: biggest BPB improvement found.**
- [x] **Muon WD=0.03 (compromise)** — TRIED: BPB 1.1187 (same as 0.02!), 16.52MB (OVER by 517KB). BPB plateaus 0.02-0.03.
- [x] **Muon WD=0.035** — TRIED: BPB 1.1204 base (only -0.0003), 16.01MB (OVER by 9KB). BPB cliff is between 0.03-0.035. Dead end.
- [-] **Muon WD=0.025** — Abandoned: 0.03 already same BPB as 0.02, going lower won't help. Need higher WD for size.
- [ ] **Different optimizers** — SOAP, Lion, Adalayer. Muon works well but alternatives exist.
- [ ] **Data ordering** — Smart curriculum over FineWeb shards. Some data is harder/more useful than others.
- [x] **Label smoothing** — TRIED: epsilon=0.02. RESULT: **+0.0212 BPB worse** (1.1444 vs 1.1232). Devastating at this model size — model too small to waste capacity softening targets.
- [x] **Multi-token prediction (MTP)** — TRIED: 1 head, weight=0.2. RESULT: **+0.0113 BPB worse** (1.1339 vs 1.1226). Auxiliary losses don't help at this scale — gradient interference + 1.5% compute overhead.

## Quantization & Compression

- [ ] **BitNet / ternary weights** — Train with ternary or binary weights from scratch. Zero quantization gap.
- [x] **Mixed precision per-layer (int5 MLP)** — TRIED: int5 for MLPs, int6 for attention. RESULT: **+0.019 BPB worse**. Int5 quantization gap is devastating. Not viable at this model size.
- [x] **Fix dead-coded QAT** — TRIED 5 times: QAT + 5 clips gives best BPB (1.1230) but artifact 80KB over 16MB. QAT + 10 clips fits under 16MB (15.93MB) but BPB is 1.1234 (worse than no-QAT best 1.1232). The GPTQ clip count interacts with QAT. Need to try 7-8 clips or find other ways to save ~100KB.
- [x] **More GPTQ clips (15)** — TRIED: 15 clips gives NO improvement over 10 clips. Quantization gap identical (0.0237 vs 0.0238). Search saturated at 10 points. Don't increase further.
- [-] **QAT from the start** — Abandoned: too expensive (model can't learn with full quantization noise from step 0).
- [-] **Int4/Int5 for some tensors** — Abandoned: int5 MLP experiment showed quantization error is too large.
- [-] **Aggressive post-quant pruning** — Abandoned: zeroing all ±1 quantized values saves 1.82MB but destroys BPB (+0.02). Need magnitude-aware pruning but code complexity is high.
- [x] **Better compression** — brotli-10 beats zstd-22 by 380-645KB. Already implemented and optimal.
- [x] **Grouped int6 quantization (G=128)** — TRIED: per-group scales (128 weights/group). RESULT: -0.0001 BPB on sliding window (noise). Artifact 3KB larger. Per-row GPTQ-lite already near-optimal. Muon produces approximately orthogonal weights with low within-row variance. **Not worth the complexity.**
- [ ] **Structured sparsity** — 2:4 or 4:8 structured sparsity for better compression ratios.
- [ ] **Magnitude-aware pruning** — Prune by |q * scale| instead of |q|. Could save 100KB with minimal BPB impact.
- [x] **BigramHash bucket tuning** — TRIED: 4096 buckets: **+0.0018 BPB worse** (undertrained). 3072 buckets: **-0.0002 base, tied TTT** (1.1198). TTT delta drops from -0.0012 to -0.0007 with 3072. Net wash. 2048 is optimal for our step count given TTT.
- [x] **BigramHash @512d (SOTA config)** — TRIED: 1536 buckets@512d, no projection (vs 2048@128d+proj). RESULT: **+0.001 BPB worse** (1.1217 base, 1.1213 TTT). TTT delta also worse (-0.0004 vs -0.0012). More params (786K vs 327K) undertrained. SOTA's bigram works only with their speed optimizations. Don't try larger bigrams without more steps.
- [x] **BigramHash 256d (doubled dim)** — TRIED: 2048 buckets@256d+proj (vs 128d+proj). RESULT: **+0.0002 base, +0.0003 TTT worse** (1.1209 base, 1.1201 TTT). TTT delta -0.0008 (vs -0.0012). Extra 328K params undertrained. **BigramHash is fully optimized at 2048×128d for ~7000 steps. Any expansion (buckets or dim) hurts. Dead end.**
- [x] **Value Embedding layer count** — TRIED: VE on 8,9,10 (3 layers) vs 9,10 (2 layers). RESULT: +0.0015 BPB worse. Layer 8 too early for token identity. 9,10 is optimal.
- [ ] **Vocabulary size optimization** — Larger vocab = fewer tokens = potentially better BPB, but more embedding params. Find the sweet spot.
- [ ] **Knowledge distillation into quantized model** — Train large, distill into the 16MB target.

## Evaluation

- [x] **Sliding window stride optimization** — TRIED stride=32 vs stride=64. RESULT: Only 0.00003 BPB difference. Not worth the 2x eval time. Stride=64 is optimal.
- [x] **Test-time training (TTT)** — TRIED: Full-model SGD TTT. SOTA defaults (lr=0.002, 3ep) caused catastrophic forgetting (+0.023 BPB). Conservative (lr=0.0005, 1ep) gives **-0.0012 BPB** (best delta). lr=0.001: -0.0010. Freeze 6 blocks: +0.0005 worse. Anchor alpha=0.0003: delta=-0.0007. **2 epochs at lr=0.0005: delta=-0.0007** (extra epoch causes forgetting). **3ep+2freeze (SOTA config): delta=+0.0003 WORSE** — 9 unfrozen blocks overfit with 3 epochs. SOTA's model handles 3ep because of stronger base (BigramHash 1536d, more steps). **All TTT configs exhausted. Best: 1ep/0freeze/lr=0.0005 → delta=-0.0012.**
- [x] **Longer eval context (4096)** — TRIED: eval_seq_len=4096 with NTK RoPE extrapolation (1024→4096, 4x). RESULT: **Sliding window CATASTROPHIC: 1.5502 BPB** (vs 1.1203 at 2048). RoPE NTK extrapolation fails at 4x even with 75% position-independent attention. Don't try eval_seq > 2048 without training at that length.
- [ ] **Ensembling within 16MB** — Multiple tiny models that vote? Probably not enough budget.
- [x] **Temperature scaling** — TRIED: Sweep T=[0.90, 0.95, 1.00, 1.05, 1.10] on sliding window eval. RESULT: **T=1.0 is optimal.** T=0.90: +0.010 BPB worse. T=0.95: +0.003. T=1.05: +0.0005. Model is well-calibrated. The ternary entry's T=0.90 was specific to ternary quantization's under-confidence. **Dead end.**

## Meta

- [ ] **Systematic leaderboard analysis** — Read every entry, build a table of techniques vs scores.
- [ ] **Ablation studies** — For known techniques, which ones actually contribute most?
- [ ] **Search the literature** — Recent papers on efficient LM training, parameter-efficient methods.
- [ ] **Analyze the baseline** — Profile where parameters are spent. What's the bottleneck?

## Priority Queue (next experiments)

1. ~~**Speed optimization (Parameter Banking)**~~ — TRIED: Replaced all nn.Linear with 4 contiguous banks, removed DDP for manual gradient sync. RESULT: **CATASTROPHIC — 182ms/step (2.1x slower), 3307 steps, val_bpb 1.1633**. Removing DDP eliminated critical communication-computation overlap. Forward pass was faster (~80ms steps 1-10) but sequential gradient sync after backward added ~100ms. **Simpler alternative: batch NS calls in Muon by grouping same-shape params, keep DDP.**
2. ~~**BigramHash @512d (1536 buckets)**~~ — TRIED: +0.001 BPB worse. Undertrained at ~7000 steps. TTT delta also regressed (-0.0004 vs -0.0012). Needs Parameter Banking for more steps first.
3. ~~**TTT 2 epochs at lr=0.0005**~~ — TRIED: delta=-0.0007, same as 1ep+anchor. Extra epoch causes forgetting. TTT tuning exhausted.
4. ~~**Earlier QAT (threshold 0.20-0.25)**~~ — TRIED: 0.20 gave 1.1234, +0.0008 worse. 0.15 is optimal.
5. ~~**grad_clip_norm=1.0**~~ — TRIED: +0.0007 BPB worse (1.1214). Helped locally but hurt at 8-GPU scale. 0.3 is optimal for DDP training.
6. **Cross-layer KV sharing** — Reuse KV from early layers in later layers. More info flow without more params.
9. ~~**TTT 3ep + 2 frozen blocks at lr=0.0005**~~ — TRIED: delta=+0.0003 (WORSE). 9 unfrozen blocks overfit with 3 epochs. SOTA's model handles this because of stronger base. All TTT configs exhausted.
10. **Batched NS in Muon (keep DDP)** — Group same-shape params for batched Newton-Schulz via torch.bmm. Reduces kernel launches from ~44 to ~4 without touching DDP. Could save 2-3ms/step → ~150 more steps.
7. ~~**Disable QAT entirely**~~ — TRIED: 0.0 gave 1.1233, +0.0007 worse. QAT 0.15 is the sweet spot — helps both model quality and quant gap.
8. ~~**TTT with Adam optimizer**~~ — TRIED: Adam TTT lr=0.001 gave BPB 1.2620 (+0.1416 regression). Adam is catastrophically wrong for few-shot TTT — variance estimates are noisy with so few steps, causing massive uncontrolled updates. SGD with momentum is far superior for TTT.

### Muon weight decay — MAJOR DISCOVERY (UPDATED)
- WD 0.04 (default): BPB 1.1207, 15.55MB — established baseline
- WD 0.03: BPB **1.1187** (-0.0020!), **16.52MB (OVER by 517KB)**
- WD 0.02 (halved): BPB **1.1186** (-0.0021!), **17.58MB (OVER)**
- BPB plateaus between 0.02-0.03, then jumps +0.002 at 0.04. Transition is between 0.03-0.04.
- WD=0.035: BPB 1.1204 base (-0.0003), 16.01MB (OVER by 9KB). BPB cliff is 0.03-0.035.
- **Dead end**: Can't fit beneficial WD range (≤0.03) under 16MB. Need ~500KB compression savings.

## Key Findings

### EMA decay tradeoff (UPDATED)
- EMA 0.997 + zstd: BPB 1.1232, 15.7-15.8MB (old best)
- EMA 0.998 + zstd: BPB 1.1229, 16.0-16.7MB (over 16MB)
- **EMA 0.998 + brotli: BPB 1.1226, 15.4-15.5MB (NEW BEST, under 16MB!)**
- EMA 0.999 + brotli: BPB 1.1293 — WAY too broad, averages outdated history
- Optimum is clearly 0.998. Don't go higher.
- Brotli compression eliminates the EMA 0.998 size penalty
- SWA disabled saves ~1ms/step → ~50 more training steps

### Compression: brotli vs zstd (NEW)
- brotli quality=10 beats zstd-22 by 380-645KB across all models
- Especially effective on EMA 0.998 weights (broader distributions)
- Compression: ~40s on trained model, decompression: <1s
- `pip install brotli` required on eval machine

### GPTQ clips tradeoff (UPDATED — 10 clips is optimal)
- 5 clips + brotli + EMA 0.998: 1.1236 BPB, 15.47MB — WORSE than 10 clips
- 10 clips + brotli + EMA 0.998: 1.1226 BPB, 15.47MB — BEST
- More clips = better optimal MSE search = better BPB. Previous "trend" was noise.
- 10 clips is confirmed optimal. Don't reduce.

### QAT threshold fully characterized
- QAT 0.20: 1.1234 — too much QAT hurts convergence
- QAT 0.15: 1.1226 — **optimal sweet spot** (helps model quality + quant gap)
- QAT 0.00: 1.1233 — no QAT gives worse model AND larger quant gap
- Don't revisit QAT tuning — it's settled at 0.15

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
| 2026-03-26 | 5clips_brotli_ema098 | 1.1236 | 15.47MB | **REGRESSED**: 5 clips worse than 10 clips (+0.001 BPB). 10 clips is optimal. |
| 2026-03-26 | ema_decay_0999 | 1.1293 | — | **REGRESSED**: EMA 0.999 way too broad (+0.0067 BPB). Optimum is 0.998. |
| 2026-03-26 | 15clips_brotli_ema098 | 1.1234 | 15.48MB | **REGRESSED**: 15 clips no better than 10. Quant gap identical. Slower pod (86ms/step). |
| 2026-03-26 | qat020_brotli_ema098 | 1.1234 | — | **REGRESSED**: Earlier QAT (0.20) hurts training quality AND quant gap. 0.15 is optimal. |
| 2026-03-27 | mtp1_head_aux_loss | 1.1339 | 15.47MB* | **REGRESSED**: MTP 1 head (weight 0.2) devastating (+0.0113 BPB). Aux losses bad at this scale. |
| 2026-03-27 | noqat_brotli_ema098 | 1.1233 | 15.47MB | **REGRESSED**: No QAT (+0.0007 BPB). QAT 0.15 confirmed optimal. |
| 2026-03-27 | bigram4096_brotli_ema098 | 1.1244 | — | **REGRESSED**: 4096 bigram buckets (+0.0018 BPB). Extra params undertrained. 2048 optimal. |
| 2026-03-27 | ve3layers_8_9_10 | 1.1241 | 15.47MB | **REGRESSED**: VE on 3 layers (+0.0015 BPB). Layer 8 too early for token identity. 9,10 optimal. |
| 2026-03-27 | batch524k_more_steps | 1.1256 | — | **REGRESSED**: Smaller batch (524K vs 786K) +0.003 BPB. Noisier gradients hurt Muon. 786K optimal. |
| 2026-03-27 | cosine_warmdown | 1.1236 | 15.69MB | **REGRESSED**: Cosine warmdown +0.001 BPB. Linear decay better for Muon optimizer. |
| 2026-03-27 | leaky_relu_05_squared | **1.1207** | 15.55MB | **NEW BEST!** LeakyReLU(0.5)² -0.0019 BPB. Preserves negative gradient flow. |
| 2026-03-28 | leaky_relu_07_squared | 1.1242 | — | **REGRESSED**: LeakyReLU(0.7)² +0.0035 BPB. Too symmetric; 0.5 is optimal slope. |
| 2026-03-28 | ttt_legal_score_first_v2 | 1.1436 | 15.55MB | **REGRESSED**: TTT with SOTA defaults (lr=0.002, 3ep) catastrophic forgetting. |
| 2026-03-28 | ttt_conservative_lr0005_ep1 | **1.1203** | 15.55MB | **NEW BEST!** TTT with lr=0.0005, 1ep: -0.0012 BPB improvement over sliding window. |
| 2026-03-28 | ttt_freeze6_lr0005_ep1 | 1.1208 | 15.55MB | **REGRESSED**: Freeze first 6 blocks +0.0005 BPB. Forgetting is distributed, not in early layers. |
| 2026-03-28 | ttt_anchor_alpha_0003 | **1.1200** | 15.55MB | **NEW BEST** absolute BPB but anchor reduces TTT delta (-0.0007 vs -0.0012). Improvement from base getting more steps (6976). |
| 2026-03-28 | ttt_lr001_noanchor | 1.1204 | 15.55MB | **REGRESSED**: Higher TTT LR (0.001 vs 0.0005) reduces TTT delta from -0.025 to -0.024. TTT saturated at lr=0.0005. |
| 2026-03-28 | ttt_2ep_lr0005 | **1.1198** | 15.56MB | **NEW BEST** absolute but TTT delta=-0.0007 (same as 1ep+anchor). 2 epochs causes forgetting, base improvement from run variance. |
| 2026-03-28 | bigram1536_512d_noproj | 1.1213 | 15.89MB | **REGRESSED**: BigramHash 1536@512d (SOTA config) +0.0015 BPB. Undertrained + TTT delta only -0.0004. Needs more steps. |
| 2026-03-28 | grad_clip_1.0 | 1.1214 | 15.53MB | **REGRESSED**: grad_clip 1.0 vs 0.3 (+0.0007 BPB). Helped locally but hurt at 8-GPU scale. 0.3 is optimal. |
| 2026-03-29 | ttt_adam_lr001 | 1.2620 | 15.55MB | **REGRESSED**: Adam TTT catastrophic (+0.1416 BPB). Base=1.1204 (fine), Adam destroys generalization in few-shot TTT. SGD is correct for TTT. |
| 2026-03-29 | grouped_int6_g128 | 1.1200 | 15.56MB | **NO IMPROVEMENT**: Grouped quant (G=128) gives -0.0001 on SW (noise). Per-row already optimal for orthogonal Muon weights. |
| 2026-03-29 | eval_seq4096_ntk_extrap | 1.1201 | 15.55MB | **REGRESSED**: 4096 eval context via NTK RoPE extrapolation. SW=1.5502 (catastrophic). 4x extrapolation fails. TTT unaffected (uses 2048). |
| 2026-03-29 | temp_sweep_eval | 1.1203 | 15.55MB | **NO IMPROVEMENT**: Temperature scaling T=[0.90-1.10] all worse than T=1.0. Model well-calibrated. Base=1.1212 (SW), TTT=1.1203. SwiGLU also tested locally: +0.007 BPB worse. |
| 2026-03-29 | param_banking_parallel_muon | 1.1633 | 14.57MB | **CATASTROPHIC**: Removed DDP for parameter banks. 182ms/step (2.1x slower), only 3307 steps. Sequential gradient sync after backward. |
| 2026-03-29 | bigram3072_brotli_ema098 | 1.1198 | 15.68MB | **TIED**: BigramHash 3072 buckets. Base improved -0.0002 (1.1205 vs 1.1207) but TTT delta dropped to -0.0007. Net wash. |
| 2026-03-29 | bigram256d_brotli_ema098 | 1.1201 | 15.76MB | **REGRESSED**: BigramHash 256d (2x dim). Base +0.0002, TTT delta -0.0008 (vs -0.0012). Extra 328K params undertrained. BigramHash fully optimized. |
| 2026-03-29 | ttt_3ep_2freeze | 1.1209 | 15.55MB | **REGRESSED**: TTT 3ep+2freeze (SOTA config). TTT delta=+0.0003 (WORSE than no TTT!). 9 unfrozen blocks overfit with 3 epochs. All TTT configs exhausted. |
| 2026-03-29 | warmdown3000_sota_match | 1.1205 | 15.72MB | **REGRESSED**: warmdown 3000 (SOTA config) vs 3500. Base 1.1211 (+0.0004), TTT delta -0.0006 (halved). EMA 0.998 needs longer warmdown. 3500 is optimal. |
| 2026-03-30 | muon_wd_002 | **1.1186** | 17.58MB (OVER) | **BEST BPB EVER but OVER 16MB**: Muon WD 0.02 (half of 0.04) gives -0.0021 BPB. Larger weights don't compress. Need WD ~0.03 sweet spot. |
| 2026-03-30 | muon_wd_003 | **1.1187** | 16.52MB (OVER) | **OVER by 517KB**: BPB same as WD=0.02 (plateau). Need WD=0.035 to fit under 16MB. |
| 2026-03-30 | muon_wd_0035 | 1.1196 | 16.01MB (OVER) | **OVER by 9KB**: Base 1.1204 (same as 0.04!). BPB cliff between 0.03-0.035. WD tuning dead end. |
