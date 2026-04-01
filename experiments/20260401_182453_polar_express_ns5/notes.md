# Experiment: Polar Express Newton-Schulz Coefficients

## Hypothesis
Replacing fixed NS coefficients (3.4445, -4.7750, 2.0315) with per-iteration Polar Express optimal quintic polynomials (Amsel et al., ICLR 2026) would give better gradient orthogonalization → better training quality → better BPB. The paper shows consistent improvements on GPT-2 scale models.

## Changes
1. Replaced fixed (a, b, c) tuple with 8 per-iteration Polar Express coefficients
2. Changed normalization from `X /= X.norm() + eps` to `X /= X.norm() * 1.01 + eps` (required by PE coefficients)
3. Kept muon_backend_steps=5 (same iteration count)

## Results
| Metric | Best (NS5 fixed) | Run 1 (PE) | Run 2 (PE + Cholesky fix) |
|--------|-------------------|------------|---------------------------|
| val_bpb | **1.1168** | CRASHED | CRASHED |
| steps | 6676 | 6716 | 6492 |
| exit_code | 0 | 1 | 1 |

## Crash Analysis

**Run 1:** Training completed (6716 steps, GOOD — even more than baseline 6676). Crashed during GPTQ quantization — Cholesky factorization failed because Hessian from AR self-generated calibration data was not positive-definite. The PE-trained model produces activations that create degenerate Hessians for some layers.

**Run 2:** Added try/except with 10x stronger damping around Cholesky. This fixed the Cholesky issue but caused NaN propagation in the GPTQ error compensation loop (`err = (w - q*sf) / d` where d ≈ 0 from over-damped Hinv). `best_q` remained None because all MSE values were NaN.

## Root Cause
The Polar Express normalization (`* 1.01`) scales singular values to [0, 1/1.01], producing weight matrices with different spectral properties than standard NS. During AR self-generation (temp=0.8), these weights produce activations where some input features are highly correlated or near-zero, causing:
1. Rank-deficient H = X^T X matrices
2. Cholesky decomposition failure even with 1% damping
3. Over-damping (10%) fixes Cholesky but makes Hinv diagonal elements too small → NaN

## Key Learning
- Polar Express TRAINING works fine (6716 steps, loss decreasing normally)
- The issue is at the GPTQ quantization stage, not training
- The NS coefficients affect weight spectral properties which cascade into quantization quality
- Any future NS coefficient changes need to be validated with the FULL pipeline including GPTQ
- A more robust GPTQ implementation (eigenvalue clamping instead of diagonal damping) could fix this

## What to Try Next
- If revisiting Polar Express: fix GPTQ to use eigendecomposition-based regularization (clamp eigenvalues > eps instead of adding diagonal damping)
- Or: use Polar Express for only the last 3-4 NS iterations, keeping the first 1-2 as standard NS
- Or: abandon Polar Express and focus on other approaches (SSM hybrid, structured sparsity, speed optimization)
