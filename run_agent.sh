#!/bin/bash
set -euo pipefail

# Parameter Golf Agent Loop
# Runs Claude Code in a loop, each invocation picks up an experiment,
# implements it, tests it, records results, and exits.
#
# Usage:
#   ./run_agent.sh                  # Run indefinitely
#   ./run_agent.sh --max-runs 10    # Run at most 10 iterations
#
# Requirements:
#   - claude CLI installed and authenticated
#   - RUNPOD_API_KEY set in environment
#   - GPU available for local testing

AGENT_PROMPT="AGENT_PROMPT.md"
LOG_DIR="agent_logs"
MAX_CONSECUTIVE_FAILURES=3
COOLDOWN_SECONDS=30
BUDGET_LIMIT=900  # Soft limit - agent will also check
MAX_RUNS=0        # 0 = unlimited

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-runs)
            MAX_RUNS="$2"
            shift 2
            ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$LOG_DIR"

# Check prerequisites
if ! command -v claude &>/dev/null; then
    echo "ERROR: claude CLI not found. Install it first."
    exit 1
fi

if [ ! -f "$AGENT_PROMPT" ]; then
    echo "ERROR: $AGENT_PROMPT not found. Run from repo root."
    exit 1
fi

if [ -z "${RUNPOD_API_KEY:-}" ]; then
    echo "WARNING: RUNPOD_API_KEY not set. Agent won't be able to run RunPod evaluations."
fi

failure_count=0
run_count=0

echo "=== Parameter Golf Agent Loop ==="
echo "  Prompt:    $AGENT_PROMPT"
echo "  Log dir:   $LOG_DIR"
echo "  Max runs:  ${MAX_RUNS:-unlimited}"
echo "  Cooldown:  ${COOLDOWN_SECONDS}s"
echo ""

while true; do
    # Check run limit
    if [ "$MAX_RUNS" -gt 0 ] && [ "$run_count" -ge "$MAX_RUNS" ]; then
        echo "Reached max runs ($MAX_RUNS). Stopping."
        break
    fi

    # Check budget (quick parse of BUDGET.md)
    if [ -f "BUDGET.md" ]; then
        SPENT=$(grep -oP 'Cumulative:\s*\$?\K[\d.]+' BUDGET.md 2>/dev/null | tail -1 || echo "0")
        if (( $(echo "$SPENT > $BUDGET_LIMIT" | bc -l 2>/dev/null || echo 0) )); then
            echo "Budget limit reached (\$${SPENT} > \$${BUDGET_LIMIT}). Stopping."
            echo "Review experiments/ and decide whether to continue."
            break
        fi
    fi

    COMMIT=$(git rev-parse --short=8 HEAD)
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    TRANSCRIPT="${LOG_DIR}/agent_${TIMESTAMP}_${COMMIT}.json"
    LOGFILE="${LOG_DIR}/agent_${TIMESTAMP}_${COMMIT}.log"

    run_count=$((run_count + 1))
    echo "=== Run #${run_count} at $(date) on commit ${COMMIT} ===" | tee -a "$LOGFILE"

    # Run Claude with full transcript capture
    # --output-format json captures the full conversation
    # We also tee stderr to the log for any errors
    if claude --dangerously-skip-permissions \
              -p "$(cat "$AGENT_PROMPT")" \
              --model claude-opus-4-6 \
              --output-format json \
              2>> "$LOGFILE" \
              > "$TRANSCRIPT"; then
        failure_count=0
        echo "=== Run #${run_count} completed successfully at $(date) ===" | tee -a "$LOGFILE"

        # Extract a summary from the JSON transcript for the log
        if command -v jq &>/dev/null && [ -f "$TRANSCRIPT" ]; then
            # Get the last assistant message as a quick summary
            jq -r '.[] | select(.type == "assistant") | .message // empty' "$TRANSCRIPT" 2>/dev/null | tail -20 >> "$LOGFILE" || true
        fi
    else
        failure_count=$((failure_count + 1))
        echo "=== Run #${run_count} FAILED at $(date) (failure ${failure_count}/${MAX_CONSECUTIVE_FAILURES}) ===" | tee -a "$LOGFILE"

        if [ "$failure_count" -ge "$MAX_CONSECUTIVE_FAILURES" ]; then
            echo "Too many consecutive failures ($MAX_CONSECUTIVE_FAILURES). Stopping."
            echo "Check the last few logs in ${LOG_DIR}/ for details."
            exit 1
        fi
    fi

    NEW_COMMIT=$(git rev-parse --short=8 HEAD)
    if [ "$COMMIT" = "$NEW_COMMIT" ]; then
        echo "Warning: no new commit was made this run." | tee -a "$LOGFILE"
    fi

    # Report transcript size
    if [ -f "$TRANSCRIPT" ]; then
        SIZE=$(wc -c < "$TRANSCRIPT" | tr -d ' ')
        echo "  Transcript: $TRANSCRIPT (${SIZE} bytes)" | tee -a "$LOGFILE"
    fi

    echo "  Sleeping ${COOLDOWN_SECONDS}s..."
    sleep "$COOLDOWN_SECONDS"
done

echo ""
echo "=== Agent loop finished ==="
echo "  Total runs: $run_count"
echo "  Check experiments/ for results"
echo "  Check CURRENT_BEST.md for best score"
