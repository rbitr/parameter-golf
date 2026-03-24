#!/bin/bash
set -euo pipefail

# Setup script for the Parameter Golf agent on an A100 server.
# Run this once after cloning the repo.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Parameter Golf Agent Setup ==="

# 1. Python dependencies
echo ""
echo "1. Installing Python dependencies..."
pip install -r requirements.txt
pip install runpod

# 2. Claude Code CLI
echo ""
echo "2. Checking Claude Code CLI..."
if command -v claude &>/dev/null; then
    echo "   claude CLI found: $(claude --version 2>/dev/null || echo 'installed')"
else
    echo "   Installing Claude Code CLI..."
    npm install -g @anthropic-ai/claude-code
    if command -v claude &>/dev/null; then
        echo "   Installed successfully."
    else
        echo "   FAILED: claude not found after install."
        echo "   Install manually: npm install -g @anthropic-ai/claude-code"
        echo "   Or see: https://docs.anthropic.com/en/docs/claude-code"
    fi
fi

# 3. Environment variables
echo ""
echo "3. Checking environment..."
if [ -z "${RUNPOD_API_KEY:-}" ]; then
    echo "   WARNING: RUNPOD_API_KEY not set"
    echo "   Set it with: export RUNPOD_API_KEY=your_key_here"
else
    echo "   RUNPOD_API_KEY is set"
fi

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "   WARNING: ANTHROPIC_API_KEY not set (needed for claude CLI)"
    echo "   Set it with: export ANTHROPIC_API_KEY=your_key_here"
else
    echo "   ANTHROPIC_API_KEY is set"
fi

# 4. GPU check
echo ""
echo "4. Checking GPU..."
if python3 -c "import torch; print(f'   CUDA available: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')" 2>/dev/null; then
    :
else
    echo "   WARNING: PyTorch CUDA check failed. Make sure torch is installed with CUDA support."
fi

# 5. Download data
echo ""
echo "5. Downloading training data (1 shard for local testing)..."
if [ -d "$REPO_ROOT/data/datasets/fineweb10B_sp1024" ]; then
    SHARD_COUNT=$(ls "$REPO_ROOT/data/datasets/fineweb10B_sp1024"/fineweb_train_*.bin 2>/dev/null | wc -l)
    echo "   Data already exists ($SHARD_COUNT training shards)"
    echo "   To download more: python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10"
else
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
fi

# 6. Test RunPod connectivity
echo ""
echo "6. Testing RunPod API..."
if [ -n "${RUNPOD_API_KEY:-}" ]; then
    python3 scripts/test_runpod.py || echo "   RunPod test failed — check your API key"
else
    echo "   Skipped (RUNPOD_API_KEY not set)"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the agent loop:"
echo "  export RUNPOD_API_KEY=your_key"
echo "  export ANTHROPIC_API_KEY=your_key"
echo "  ./run_agent.sh"
