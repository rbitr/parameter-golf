# Experiment: Label Smoothing (epsilon=0.02)

## Hypothesis
Label smoothing with small epsilon (0.02) can improve generalization, potentially giving 0.0001-0.0003 BPB improvement. Using proven 10-clip GPTQ config that fits under 16MB. Minimal code change (+256 bytes), low risk.

## Changes from current best (speed_cleanup_gptq10)
- Added `label_smoothing = 0.02` hyperparameter (env var LABEL_SMOOTHING)
- Modified `GPT.forward()` to use `F.cross_entropy(..., label_smoothing=ls)` during training only
- Set via class variable `GPT._label_smoothing` to avoid serialization issues

## Config
- 10 GPTQ clips (proven to fit under 16MB)
- QAT threshold 0.15
- EMA 0.997
- SWA enabled
- warmdown 3500
- seed 1337

## Results
- val_bpb: **1.1444** (REGRESSED +0.0212 from current best 1.1232)
- val_loss: 1.9323
- steps: 7038 (85.26 ms/step)
- model size: 15,757,288 bytes
- code size: 67,864 bytes
- total artifact: 15,825,152 bytes (under 16MB)
- cost: $7.01

## Analysis
- Label smoothing at epsilon=0.02 is **devastating** for this model size
- Train losses consistently ~0.2 higher throughout training (step 500: 2.60 vs 2.40 baseline)
- The model is too small — it needs every bit of capacity to fit the data distribution precisely
- Label smoothing wastes gradient signal by softening targets the model already struggles to match
- Even small epsilon values are too aggressive for 27M param models
- **CONCLUSION: Label smoothing is not viable at this model scale. Abandon this direction.**
