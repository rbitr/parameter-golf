# TTT Legal Score-First v2

## Hypothesis
Adding test-time training (TTT) at eval time will improve BPB by ~0.002, similar to the SOTA entry.
Used the same hyperparameters as SOTA: SGD lr=0.002, momentum=0.9, 3 epochs, 32K chunks, freeze_blocks=0.

## Results
- **Sliding window BPB: 1.1208** (matches our best — base model unchanged)
- **TTT BPB: 1.1436** (MUCH WORSE — +0.023 BPB regression)
- TTT took 525s on 8xH100

## Analysis
TTT caused catastrophic forgetting. The BPB trajectory:
- Chunk 1 (no training): 1.157 (running avg of just first chunk)
- Chunk 51: 1.110 (improving rapidly — TTT IS initially helping!)
- Chunk 101: 1.118 (starting to degrade)
- Chunk 1891: 1.146 (model corrupted by excessive training)

The model adapts well to early chunks but then overwrites that knowledge when training on later chunks.
Total SGD steps: 1892 chunks × 3 epochs = 5676 steps at lr=0.002. This is too much for a 27M param model.

## What We Learned
1. TTT mechanically works — the implementation is correct
2. The SOTA's hyperparameters (lr=0.002, 3 epochs) are too aggressive for our model
3. The SOTA uses "Parallel Muon" parameter banking which may be more stable for SGD
4. Need much more conservative TTT: lower lr, fewer epochs

## Next Steps
Try conservative TTT: lr=0.0005, 1 epoch (12x less total training).
If that doesn't work, try LoRA-based TTT instead of full-model fine-tuning.
