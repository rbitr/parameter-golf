# Experiment: rope_dims32_v2

Same as rope_dims32 but with `.cpu()` fix for torch.quantile. Still crashed — torch.quantile has a hard size limit regardless of device. See rope_dims32/notes.md for full analysis. Fixed properly with np.percentile() for next attempt.
