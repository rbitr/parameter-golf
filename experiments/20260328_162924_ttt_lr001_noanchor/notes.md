# TTT lr=0.001, no anchor regularization

## Hypothesis
Doubling TTT learning rate from 0.0005 to 0.001 while keeping 1 epoch should improve TTT adaptation.
Removing anchor regularization (alpha=0.0003→0.0) which was damping good adaptation equally with bad drift.

## Changes
- TTT_LR: 0.0005 → 0.001
- TTT_ANCHOR_ALPHA: 0.0003 → 0.0

## Results
- val_bpb: 1.12042 (vs best 1.12004 with anchor, 1.12026 without anchor)
- Base BPB (pre-TTT): 1.14472
- TTT delta: -0.02430 (vs -0.02497 at lr=0.0005, -0.02441 with anchor)
- Steps: 6924
- Artifact: 15,552,315 bytes

## Analysis
Higher TTT learning rate (0.001 vs 0.0005) **slightly hurt** TTT effectiveness:
- TTT delta: -0.02430 (lr=0.001) vs -0.02497 (lr=0.0005) = 0.0007 BPB worse
- This suggests our model is already near optimal TTT LR at 0.0005
- Higher LR causes slightly more forgetting that outweighs better adaptation

Comparison of all TTT experiments:
| Config | Base BPB | TTT BPB | TTT Delta | Steps |
|---|---|---|---|---|
| lr=0.0005, 1ep, no anchor | 1.14523 | 1.12026 | -0.02497 | 6915 |
| lr=0.0005, 1ep, anchor=0.0003 | 1.14445 | 1.12004 | -0.02441 | 6976 |
| lr=0.001, 1ep, no anchor | 1.14472 | 1.12042 | -0.02430 | 6924 |
| lr=0.002, 3ep (SOTA settings) | — | 1.1436 | catastrophic | — |

## Key Insight
Our model's TTT optimal LR is 0.0005, not higher. The SOTA model tolerates lr=0.002 because of architectural differences (Parameter Banking, different BigramHash config). Our TTT improvement ceiling seems to be ~-0.025 BPB (on quantized model).

## What to Try Next
- TTT optimization is near-saturated for our model. Pivot to base model improvements.
- Parameter Banking for speed (83ms→86ms = ~200 more steps)
- MoE or cross-layer KV sharing for better base model quality
- Try lr=0.0003 to see if even lower is better (unlikely but cheap to test)
- Consider 2 epochs at lr=0.0005 as an alternative to higher LR
