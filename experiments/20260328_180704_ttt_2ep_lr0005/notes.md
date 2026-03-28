# TTT 2 Epochs at lr=0.0005

## Hypothesis
Doubling TTT epochs from 1 to 2 at conservative lr=0.0005 would push TTT delta from -0.0012 toward -0.0015+.

## Changes
- `ttt_epochs`: 1 → 2 (only change)

## Results
- **val_bpb: 1.1198 (NEW BEST absolute, but TTT delta same as before)**
- Base (sliding window): 1.1204
- TTT delta: -0.0007 (same as 1ep+anchor, WORSE than 1ep no-anchor's -0.0012)
- Steps: 6943 (fewer than 6976 of last run)
- Artifact: 15.56MB (under 16MB)

## Analysis
The 2-epoch TTT gives TTT delta of only -0.0007, which is:
- Same as 1 epoch + anchor regularization
- WORSE than 1 epoch no-anchor (-0.0012)

This means 2 epochs causes the same amount of forgetting/drift that anchor regularization was preventing. The extra epoch doesn't help — it just trades adaptation for forgetting.

The absolute BPB of 1.1198 is a new best, but this is driven by the base model scoring 1.1204 (vs previous 1.1208), which is likely run-to-run variance in training steps/hardware speed.

## Conclusion
- 2 epochs at lr=0.0005 does NOT improve TTT effectiveness
- TTT is truly saturated at -0.0007 to -0.0012 for our architecture
- The SOTA getting -0.0021 must be due to architectural differences (Parameter Banking, BigramHash@512d)
- **Next priority: improve base model to close the gap, since TTT is capped**

## Key Insight from SOTA Analysis
The SOTA uses:
- BigramHash: 1536 buckets @ 512d (full model dim!) vs our 2048 @ 128d
- Parameter Banking: batched Linear layers → 83.3ms/step vs our 86ms
- These architectural differences likely make SOTA more TTT-resilient

## What to Try Next
1. **Parameter Banking** — ~200 more training steps, zero quality impact
2. **BigramHash @512d** — match SOTA's more expressive bigram embeddings
3. Focus on base model quality rather than TTT tuning
