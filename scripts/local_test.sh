#!/bin/bash
set -euo pipefail

# Quick local A100 smoke test for Parameter Golf training script.
# Runs a short training job on 1 GPU to validate changes before RunPod evaluation.
#
# Usage:
#   scripts/local_test.sh                    # Default: 500 iters, 3 min cap
#   scripts/local_test.sh --fast             # Quick: 200 iters, 90s cap
#   scripts/local_test.sh --medium           # Medium: 1000 iters, 5 min cap

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKING_SCRIPT="${REPO_ROOT}/working/train_gpt.py"

# Defaults
ITERATIONS=500
MAX_WALLCLOCK=180
VAL_LOSS_EVERY=250
TRAIN_LOG_EVERY=50
SEED="${SEED:-1337}"

# Parse args
case "${1:-}" in
    --fast)
        ITERATIONS=200
        MAX_WALLCLOCK=90
        VAL_LOSS_EVERY=100
        ;;
    --medium)
        ITERATIONS=1000
        MAX_WALLCLOCK=300
        VAL_LOSS_EVERY=500
        ;;
esac

if [ ! -f "$WORKING_SCRIPT" ]; then
    echo "ERROR: Working script not found: $WORKING_SCRIPT"
    exit 1
fi

# Check for CUDA
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: CUDA not available"
    exit 1
fi

echo "=== Local Test ==="
echo "  Script:     $WORKING_SCRIPT"
echo "  Iterations: $ITERATIONS"
echo "  Max time:   ${MAX_WALLCLOCK}s"
echo "  Seed:       $SEED"
echo ""

# Check if data is downloaded
DATA_PATH="${REPO_ROOT}/data/datasets/fineweb10B_sp1024"
if [ ! -d "$DATA_PATH" ]; then
    echo "Data not found. Downloading (1 shard for local testing)..."
    python3 "${REPO_ROOT}/data/cached_challenge_fineweb.py" --variant sp1024 --train-shards 1
fi

# Run training on 1 GPU
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/tmp/pgolf_local_${TIMESTAMP}.log"

RUN_ID="local_${TIMESTAMP}" \
SEED="$SEED" \
ITERATIONS="$ITERATIONS" \
MAX_WALLCLOCK_SECONDS="$MAX_WALLCLOCK" \
VAL_LOSS_EVERY="$VAL_LOSS_EVERY" \
TRAIN_LOG_EVERY="$TRAIN_LOG_EVERY" \
DATA_PATH="$DATA_PATH" \
TOKENIZER_PATH="${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model" \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 "$WORKING_SCRIPT" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=== Local Test Complete ==="
echo "  Exit code: $EXIT_CODE"
echo "  Log: $LOG_FILE"

# Extract key metrics from log
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "  Key metrics:"
    grep -E "(val_loss|val_bpb|final_int8|Total submission)" "$LOG_FILE" | tail -5 | sed 's/^/    /'
fi

exit $EXIT_CODE
