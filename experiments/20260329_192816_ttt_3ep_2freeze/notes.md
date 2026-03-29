# TTT 3 Epochs + 2 Frozen Blocks

## Hypothesis
SOTA uses TTT with 3 epochs and 2 frozen blocks, getting -0.0017 to -0.0021 TTT delta. We've only tried 1ep/0freeze (-0.0012), 2ep/0freeze (-0.0007), and 1ep/6freeze (-0.0005). The SOTA sweet spot of 2 frozen blocks might prevent early-layer forgetting while allowing deeper layers to adapt over 3 epochs.

## Change
- TTT_EPOCHS: 1 → 3
- TTT_FREEZE_BLOCKS: 0 → 2
- Everything else unchanged from best config (LeakyReLU 0.5², brotli, EMA 0.998, etc.)

## Results
- Base (sliding window): **1.1206** (normal, within run variance of best 1.1205-1.1207)
- TTT (3ep, 2freeze): **1.1209**
- TTT delta: **+0.0003** (REGRESSION — TTT made it WORSE)
- Artifact: 15,554,014 bytes (under 16MB)
- Steps: 6986, ms/step: 85.89

## Analysis
TTT with 3 epochs and 2 frozen blocks is catastrophically wrong for our model:
- 9 unfrozen blocks (3-10) getting 3 epochs of SGD at lr=0.0005 is too much adaptation
- The model overfits to each chunk, losing generalization
- Compare: 0 frozen + 1 epoch = -0.0012 (best), 0 frozen + 2 epoch = -0.0007 (some forgetting), 2 frozen + 3 epoch = +0.0003 (full overfitting)

## Why SOTA's TTT works differently
SOTA's model has:
- More training steps (7185 vs 6986) from Parameter Banking speed optimization
- BigramHash 1536d (stronger representation, more robust to TTT perturbation)
- Potentially different weight statistics from parameter banking
- Their model may be more "TTT-friendly" due to these architectural differences

## Key Learning
Our model's optimal TTT is firmly 1 epoch, 0 frozen blocks, lr=0.0005. Any increase in epochs causes forgetting that outweighs adaptation benefits. Freezing blocks doesn't help because our model has only 11 layers — freezing any is too costly.

**TTT tuning is fully exhausted for our architecture. All configurations tried:**
- 1ep/0freeze: -0.0012 (best)
- 2ep/0freeze: -0.0007
- 1ep/6freeze: -0.0005
- 3ep/2freeze: +0.0003
- anchor_alpha=0.0003: -0.0007
- lr=0.001: -0.0010
- Adam optimizer: catastrophic

## What to try next
Focus entirely on base model improvements. TTT is capped. Ideas:
- Cross-layer KV sharing (more information flow without more params)
- Speed optimization (batched NS in Muon, keep DDP) for more steps
- Novel architectural changes (MoE, factored embeddings)
