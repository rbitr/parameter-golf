# Experiment: QAT + 10 GPTQ Clips + Legacy Format

## Hypothesis
QAT (threshold=0.15) with 10 GPTQ clips and legacy torch.save format would produce an artifact under 16MB while achieving ~1.1230 BPB. The reasoning: QAT with 5 clips was 80KB over (16.08MB), and 10 clips compress ~130KB better than 5 clips.

## Changes from Best (speed_cleanup_gptq10)
- Enabled late_qat_threshold=0.15 (was 0.0)

## Results
- val_bpb (stride=64): **1.1234** (WORSE than best 1.1232 by 0.0002)
- val_bpb (stride=32): **1.1234** (negligible difference from stride=64)
- Artifact: **15.93MB** (UNDER 16MB ✓)
- Steps: 7035 (vs 6999 best, vs 7048 QAT+5clips)
- QAT enabled at step 6509

## Analysis
1. **Artifact fits!** QAT + 10 clips + legacy = 15.93MB. The hypothesis about compression was correct.
2. **BPB regressed.** 1.1234 vs 1.1232 (no QAT) and vs 1.1230 (QAT+5clips).
3. **Why BPB regressed with QAT?** Possible explanations:
   - 10 clips may pick suboptimal clip points for QAT-trained weights
   - QAT changes the weight distribution → GPTQ-lite interacts differently
   - Run variance (0.0002 BPB is within typical std=0.0005)
4. **Stride=32 vs 64**: Only 0.00003 BPB difference. Not worth the 2x eval time.

## Key Learnings
- QAT + 10 clips produces artifacts that reliably fit under 16MB
- The BPB benefit of QAT is not guaranteed with different GPTQ clip counts
- Previous best QAT (1.1230, 5 clips) may have been optimal clip count for QAT weights
- Need to find a way to make QAT + 5 clips fit under 16MB, OR accept QAT + 10 clips if repeated runs show better BPB

## Next Steps
1. Try QAT + 5 clips with reduced code size to fit under 16MB
2. Try QAT + 7-8 clips as middle ground
3. Consider speed optimization to get more training steps
4. Consider label smoothing as orthogonal improvement
