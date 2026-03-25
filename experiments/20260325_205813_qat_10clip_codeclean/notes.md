# Experiment: QAT + 10-Clip GPTQ + Code Cleanup

## Hypothesis
10 GPTQ clips produce smaller artifacts than 5 clips (per earlier findings). Combined with QAT and code cleanup, might fit under 16MB.

## Changes
- Same as match_sota_qat_cleanup but with 10 GPTQ clips instead of 5
- Code cleaned further (removed comments)

## Results
- **val_bpb: 1.1236** (worse than 5-clip QAT run's 1.1230)
- **artifact: 16,137,164 bytes** (137KB OVER 16MB)
- **steps: 7037**

## Analysis
- 10 clips produced LARGER artifact (16.14MB) than 5 clips (16.08MB) with QAT
- This is opposite to non-QAT behavior where 10 clips was often better
- QAT changes weight distributions in ways that affect optimal clip count
- BPB also worse (1.1236 vs 1.1230) — fewer steps (7037 vs 7048) partly explains this

## Key Learnings
1. With QAT, 5 clips > 10 clips for both artifact size and BPB
2. The QAT + artifact size problem persists regardless of clip count
