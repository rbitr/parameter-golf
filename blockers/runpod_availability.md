# Blocker: RunPod 8xH100 SXM Availability

## Issue
RunPod 8xH100 SXM pods frequently fail to become SSH-ready. 5 out of 8 attempts failed with pod readiness timeouts (status shows RUNNING but SSH port never becomes available).

## Impact
- $19.83 wasted on failed pods (out of $33.96 total spend)
- Only 2 successful evaluations completed
- Cannot iterate on improvements without reliable evaluation

## Successful Runs
- 2026-03-24 14:30 UTC: Pod ready in ~30s ✓
- 2026-03-24 14:50 UTC: Pod ready in ~22s ✓

## Failed Runs (no SSH)
- 13:56, 14:07, 15:23, 15:39, 15:55 UTC

## Possible Causes
1. 8xH100 capacity exhaustion on RunPod (time-of-day dependent?)
2. Template (y5cejece4j) issue with SSH startup on some nodes
3. Cloud type "SECURE" may have limited capacity

## Suggestions
1. Try running during off-peak hours
2. Try cloud_type="COMMUNITY" instead of "SECURE"
3. Check RunPod status page for known issues
4. Consider using runpodctl CLI for better pod management
