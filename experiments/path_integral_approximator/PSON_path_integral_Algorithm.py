"""
PSON Path Integral Algorithm
============================

PSON algorithm for sparse path integral approximation.

HOW PSON DIFFERS FROM CLASSICAL PATH INTEGRAL METHODS:

| Method                | Approach                              | PSON Difference                    |
|-----------------------|---------------------------------------|-------------------------------------|
| Monte Carlo (PIMC)    | Sample many random paths, average     | Optimizes *fixed sparse set*        |
| Time-Slicing          | Discretize time into N slices         | Works with irregular spacing        |
| Stationary Phase      | Focus on classical paths (δS=0)       | Explores *all* paths with noise     |
| Laplace's Method      | Gaussian around maximum               | No smoothness assumptions           |
| Wick Rotation         | t → iτ (imaginary time)               | Works in real space                 |

Most methods either:
1. Sample many random paths (expensive, high variance)
2. Focus on classical paths (misses off-shell contributions)

PSON is unique: It *optimizes* a sparse fixed set of paths using precision-scaled
orthogonal noise. This is closer to importance sampling but with guaranteed descent.

TESTED ALTERNATIVES (all either hurt or showed negligible improvement):
- Antithetic variates: Lower MSE but introduces BIAS (skews distribution shape)
- Adaptive error precision: Marginal improvement within noise
- Importance weighting: Marginal improvement within noise
- Stationary phase focus: Small improvement but adds complexity
- Temperature annealing: Hurt performance
- ASPIC smoothing: Hurt performance
- MPPI weighting: Hurt performance

CONCLUSION: Baseline PSON is near-optimal for this problem. Simpler is better.

Usage:
    uv run python experiments/PSON_path_integral_Algorithm.py
    uv run python experiments/PSON_path_integral_Algorithm.py --plot

Artifacts:
    - pson_path_integral_results.json
    - pson_path_integral_comparison.png
"""

import argparse
import json
from typing import List, Dict
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Core Functions
# =============================================================================

def generate_primes(n: int) -> List[int]:
    """Generate first n prime numbers."""
    primes = []
    candidate = 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1 if candidate == 2 else 2
    return primes


def build_sparse_gaps(count: int) -> np.ndarray:
    """Build sparse gaps from primes."""
    return np.array([p * 10.0 for p in generate_primes(count)])


def build_dense_gaps(sparse_gaps: np.ndarray, num_dense: int) -> np.ndarray:
    """Build dense uniform gaps spanning sparse range."""
    return np.linspace(sparse_gaps.min(), sparse_gaps.max(), num_dense)


# Screen geometry
_X_SCREEN = np.linspace(-0.005, 0.005, 500)
_THETA = _X_SCREEN / 1.0


def simulate_intensity(
    gaps: np.ndarray,
    phases: np.ndarray,
    lambda_nm: float = 633.0,
) -> np.ndarray:
    """Simulate interference intensity pattern."""
    k = 2 * np.pi / (lambda_nm * 1e-9)
    theta = _THETA
    
    intensities = []
    for gap, phase in zip(gaps, phases):
        d = gap * 1e-6
        phi = k * d * np.sin(theta) + phase
        field1 = 0.5 * np.exp(0j)
        field2 = 0.5 * np.exp(1j * phi)
        I = np.abs(field1 + field2) ** 2
        intensities.append(I)
    
    return np.mean(intensities, axis=0)


def compute_precision(gaps: np.ndarray) -> np.ndarray:
    """Compute precision from gap irregularity."""
    gap_mean = np.mean(gaps)
    gap_var = np.var(gaps) + 1e-8
    irregularity = (gaps - gap_mean) ** 2 / gap_var
    return np.clip(1.0 / (1.0 + irregularity), 1e-4, 1.0)


def compute_weights(gaps: np.ndarray) -> np.ndarray:
    """Compute non-local credit weights from irregularity."""
    gap_mean = np.mean(gaps)
    gap_var = np.var(gaps) + 1e-8
    irregularity = (gaps - gap_mean) ** 2 / gap_var
    weights = irregularity.copy()
    if weights.sum() < 1e-8:
        weights = np.ones_like(weights)
    return weights / weights.sum()


def project_orthogonal(
    grad: np.ndarray,
    precision: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Project noise orthogonal to gradient with metric."""
    z = rng.normal(0, 1, size=grad.shape[0])
    Mz = precision * z
    Mg = precision * grad
    denom = np.dot(grad, Mg) + 1e-12
    if abs(denom) < 1e-18:
        return z
    alpha = np.dot(grad, Mz) / denom
    return z - alpha * grad


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((a - b) ** 2))


def visibility(I: np.ndarray) -> float:
    """Fringe visibility."""
    return (I.max() - I.min()) / (I.max() + I.min() + 1e-12)


# =============================================================================
# PSON Path Integral Optimizer
# =============================================================================

@dataclass
class OptResult:
    """Optimization result."""
    final_mse: float
    init_mse: float
    mse_improvement: float
    final_visibility: float
    target_visibility: float
    accept_rate: float
    mse_curve: List[float]
    phases: np.ndarray


def run_pson_path_integral(
    sparse_gaps: np.ndarray,
    I_target: np.ndarray,
    steps: int = 300,
    w: float = 0.2,
    lr: float = 0.1,
    noise_scale: float = 0.02,
    seed: int = 42,
) -> OptResult:
    """
    Run PSON optimizer for path integral approximation.
    
    Args:
        sparse_gaps: Sparse path positions (from primes)
        I_target: Target intensity pattern (from dense paths)
        steps: Optimization steps
        w: Non-local gradient weight
        lr: Learning rate
        noise_scale: PSON noise scale
        seed: Random seed
    
    Returns:
        OptResult with final phases and metrics
    """
    rng = np.random.default_rng(seed)
    n = len(sparse_gaps)
    
    # Initialize
    phases = np.zeros(n)
    weights = compute_weights(sparse_gaps)
    precision = compute_precision(sparse_gaps)
    
    # Track progress
    mse_curve = []
    accepted = 0
    attempts = 0
    
    # Initial MSE
    I_init = simulate_intensity(sparse_gaps, phases)
    init_mse = mse(I_init, I_target)
    mse_curve.append(init_mse)
    
    for step in range(steps):
        # Current state
        I_cur = simulate_intensity(sparse_gaps, phases)
        E_cur = mse(I_cur, I_target)
        
        # Non-local gradient
        grad = -w * E_cur * weights
        
        # Deterministic proposal
        proposal = phases - lr * grad
        
        # PSON noise (orthogonal, precision-scaled)
        delta = project_orthogonal(grad, precision, rng)
        noise = (delta / (np.sqrt(precision) + 1e-12)) * noise_scale
        candidate = proposal + noise
        
        # Accept/reject
        attempts += 1
        I_new = simulate_intensity(sparse_gaps, candidate)
        E_new = mse(I_new, I_target)
        
        if E_new <= E_cur:
            phases = candidate
            accepted += 1
            mse_curve.append(E_new)
        else:
            # Try deterministic fallback
            attempts += 1
            I_det = simulate_intensity(sparse_gaps, proposal)
            E_det = mse(I_det, I_target)
            if E_det <= E_cur:
                phases = proposal
                accepted += 1
                mse_curve.append(E_det)
            else:
                mse_curve.append(E_cur)
    
    # Final results
    I_final = simulate_intensity(sparse_gaps, phases)
    final_mse = mse(I_final, I_target)
    
    return OptResult(
        final_mse=final_mse,
        init_mse=init_mse,
        mse_improvement=(init_mse - final_mse) / (init_mse + 1e-12) * 100,
        final_visibility=visibility(I_final),
        target_visibility=visibility(I_target),
        accept_rate=accepted / attempts if attempts > 0 else 0,
        mse_curve=mse_curve,
        phases=phases,
    )


# =============================================================================
# Main
# =============================================================================

def run_experiment(
    n_sparse: int = 25,
    n_dense: int = 200,
    steps: int = 300,
    seeds: List[int] = [42, 123, 456],
    plot: bool = False,
) -> Dict:
    """Run PSON path integral experiment."""
    
    sparse_gaps = build_sparse_gaps(n_sparse)
    dense_gaps = build_dense_gaps(sparse_gaps, n_dense)
    I_target = simulate_intensity(dense_gaps, np.zeros(n_dense))
    
    print("=" * 60)
    print("PSON PATH INTEGRAL")
    print("=" * 60)
    print(f"Sparse: {n_sparse} paths, Dense: {n_dense} paths")
    print(f"Steps: {steps}, Seeds: {seeds}")
    print()
    
    mses = []
    improvements = []
    last_result = None
    
    for seed in seeds:
        result = run_pson_path_integral(
            sparse_gaps=sparse_gaps,
            I_target=I_target,
            steps=steps,
            seed=seed,
        )
        mses.append(result.final_mse)
        improvements.append(result.mse_improvement)
        last_result = result
        print(f"  Seed {seed}: MSE={result.final_mse:.6f}, Improvement={result.mse_improvement:+.1f}%")
    
    print()
    print("-" * 60)
    print(f"Mean MSE: {np.mean(mses):.6f} (+/- {np.std(mses):.6f})")
    print(f"Mean Improvement: {np.mean(improvements):+.1f}%")
    
    if plot and last_result is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        I_init = simulate_intensity(sparse_gaps, np.zeros(n_sparse))
        I_final = simulate_intensity(sparse_gaps, last_result.phases)
        
        ax.plot(I_target, label='Dense Target (200 paths)', linewidth=2, alpha=0.8)
        ax.plot(I_init, label='Sparse Init (25 paths)', linestyle='--', alpha=0.6)
        ax.plot(I_final, label='Sparse Final (PSON optimized)', linewidth=2)
        ax.set_xlabel('Screen Position')
        ax.set_ylabel('Intensity')
        ax.set_title(f'PSON Path Integral Approximation\nMSE: {last_result.final_mse:.6f} ({last_result.mse_improvement:+.1f}% improvement)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pson_path_integral_comparison.png', dpi=150)
        plt.close()
        print("\nPlot saved: pson_path_integral_comparison.png")
    
    return {
        "mean_mse": float(np.mean(mses)),
        "std_mse": float(np.std(mses)),
        "mean_improvement": float(np.mean(improvements)),
        "mses": [float(m) for m in mses],
    }


def main():
    parser = argparse.ArgumentParser(description="PSON Path Integral Algorithm")
    parser.add_argument("--sparse", type=int, default=25)
    parser.add_argument("--dense", type=int, default=200)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--plot", action="store_true", help="Generate comparison plot")
    args = parser.parse_args()
    
    seeds = [int(x) for x in args.seeds.split(",")]
    
    results = run_experiment(
        n_sparse=args.sparse,
        n_dense=args.dense,
        steps=args.steps,
        seeds=seeds,
        plot=args.plot,
    )
    
    with open("pson_path_integral_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved: pson_path_integral_results.json")


if __name__ == "__main__":
    main()
