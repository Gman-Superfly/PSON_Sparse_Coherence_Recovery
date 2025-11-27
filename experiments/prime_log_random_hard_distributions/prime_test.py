"""
Prime Test: Are prime-based gaps actually better than other distributions?

Compares:
1. PRIME gaps (our default)
2. UNIFORM gaps (evenly spaced)
3. RANDOM gaps (random positions)
4. FIBONACCI gaps (another structured sequence)
5. LOG gaps (logarithmically spaced)

For each, we test PSON optimization on sparse path integral approximation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import json


# =============================================================================
# Gap Generation Functions
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


def build_prime_gaps(count: int) -> np.ndarray:
    """Prime-based gaps."""
    return np.array([p * 10.0 for p in generate_primes(count)])


def build_uniform_gaps(count: int, min_val: float = 20.0, max_val: float = 1000.0) -> np.ndarray:
    """Uniformly spaced gaps."""
    return np.linspace(min_val, max_val, count)


def build_random_gaps(count: int, min_val: float = 20.0, max_val: float = 1000.0, seed: int = 42) -> np.ndarray:
    """Random gaps (sorted)."""
    rng = np.random.default_rng(seed)
    gaps = rng.uniform(min_val, max_val, count)
    return np.sort(gaps)


def build_log_gaps(count: int, min_val: float = 20.0, max_val: float = 1000.0) -> np.ndarray:
    """Logarithmically spaced gaps."""
    return np.geomspace(min_val, max_val, count)


# =============================================================================
# Physics Simulation
# =============================================================================

_X_SCREEN = np.linspace(-0.005, 0.005, 500)
_THETA = _X_SCREEN / 1.0


def simulate_intensity(gaps: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """Simulate interference intensity pattern."""
    k = 2 * np.pi / (633.0 * 1e-9)
    intensities = []
    for gap, phase in zip(gaps, phases):
        d = gap * 1e-6
        phi = k * d * np.sin(_THETA) + phase
        I = np.abs(0.5 + 0.5 * np.exp(1j * phi)) ** 2
        intensities.append(I)
    return np.mean(intensities, axis=0)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def visibility(I: np.ndarray) -> float:
    return (I.max() - I.min()) / (I.max() + I.min() + 1e-12)


# =============================================================================
# PSON Optimizer
# =============================================================================

def compute_precision(gaps: np.ndarray) -> np.ndarray:
    gap_mean = np.mean(gaps)
    gap_var = np.var(gaps) + 1e-8
    irregularity = (gaps - gap_mean) ** 2 / gap_var
    return np.clip(1.0 / (1.0 + irregularity), 1e-4, 1.0)


def compute_weights(gaps: np.ndarray) -> np.ndarray:
    gap_mean = np.mean(gaps)
    gap_var = np.var(gaps) + 1e-8
    irregularity = (gaps - gap_mean) ** 2 / gap_var
    weights = irregularity.copy()
    if weights.sum() < 1e-8:
        weights = np.ones_like(weights)
    return weights / weights.sum()


def project_orthogonal(grad: np.ndarray, precision: np.ndarray, rng) -> np.ndarray:
    z = rng.normal(0, 1, size=grad.shape[0])
    Mz = precision * z
    Mg = precision * grad
    denom = np.dot(grad, Mg) + 1e-12
    if abs(denom) < 1e-18:
        return z
    alpha = np.dot(grad, Mz) / denom
    return z - alpha * grad


def run_pson(
    sparse_gaps: np.ndarray,
    I_target: np.ndarray,
    steps: int = 300,
    seed: int = 42,
) -> Tuple[float, float, np.ndarray]:
    """Run PSON optimization. Returns (init_mse, final_mse, final_phases)."""
    rng = np.random.default_rng(seed)
    n = len(sparse_gaps)
    
    phases = np.zeros(n)
    weights = compute_weights(sparse_gaps)
    precision = compute_precision(sparse_gaps)
    
    w, lr, noise_scale = 0.2, 0.1, 0.02
    
    I_init = simulate_intensity(sparse_gaps, phases)
    init_mse = mse(I_init, I_target)
    
    for step in range(steps):
        I_cur = simulate_intensity(sparse_gaps, phases)
        E_cur = mse(I_cur, I_target)
        
        grad = -w * E_cur * weights
        proposal = phases - lr * grad
        
        delta = project_orthogonal(grad, precision, rng)
        noise = (delta / (np.sqrt(precision) + 1e-12)) * noise_scale
        candidate = proposal + noise
        
        I_new = simulate_intensity(sparse_gaps, candidate)
        E_new = mse(I_new, I_target)
        
        if E_new <= E_cur:
            phases = candidate
        else:
            I_det = simulate_intensity(sparse_gaps, proposal)
            E_det = mse(I_det, I_target)
            if E_det <= E_cur:
                phases = proposal
    
    I_final = simulate_intensity(sparse_gaps, phases)
    final_mse = mse(I_final, I_target)
    
    return init_mse, final_mse, phases


# =============================================================================
# Main Test
# =============================================================================

def main():
    n_sparse = 25
    n_dense = 200
    steps = 300
    seeds = [42, 123, 456]
    
    # Test with MULTIPLE dense reference types
    dense_configs = {
        "Uniform Dense": np.linspace(20.0, 1000.0, n_dense),
        "Random Dense": build_random_gaps(n_dense, 20.0, 1000.0, seed=999),
        "Prime Dense": build_prime_gaps(n_dense),
    }
    
    all_results = {}
    
    for dense_name, dense_gaps in dense_configs.items():
        print("\n" + "=" * 70)
        print(f"TESTING WITH: {dense_name} reference")
        print("=" * 70)
        
        I_target = simulate_intensity(dense_gaps, np.zeros(n_dense))
        results, best = run_comparison(dense_name, dense_gaps, I_target, n_sparse, steps, seeds)
        all_results[dense_name] = {"results": results, "best": best}
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Best sparse distribution for each dense reference")
    print("=" * 70)
    for dense_name, data in all_results.items():
        print(f"  {dense_name:<20} -> Best sparse: {data['best']}")
    
    # Save all results
    with open('prime_test_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved: prime_test_results.json")
    
def run_comparison(dense_name, dense_gaps, I_target, n_sparse, steps, seeds):
    """Run comparison for a specific dense reference."""
    
    # Gap distributions to test
    gap_configs = {
        "Prime": build_prime_gaps(n_sparse),
        "Uniform": build_uniform_gaps(n_sparse, 20.0, 1000.0),
        "Random": build_random_gaps(n_sparse, 20.0, 1000.0),
        "Logarithmic": build_log_gaps(n_sparse, 20.0, 1000.0),
    }
    
    print(f"Sparse: {n_sparse} paths, Dense: {len(dense_gaps)} paths, Steps: {steps}")
    print()
    
    results = {}
    
    for name, gaps in gap_configs.items():
        mses = []
        improvements = []
        
        for seed in seeds:
            init_mse, final_mse, _ = run_pson(gaps, I_target, steps, seed)
            mses.append(final_mse)
            improvements.append((init_mse - final_mse) / (init_mse + 1e-12) * 100)
        
        mean_mse = np.mean(mses)
        std_mse = np.std(mses)
        mean_imp = np.mean(improvements)
        
        # Also compute irregularity measure
        gap_mean = np.mean(gaps)
        gap_var = np.var(gaps)
        irregularity = gap_var / (gap_mean ** 2)  # Coefficient of variation squared
        
        results[name] = {
            "mean_mse": float(mean_mse),
            "std_mse": float(std_mse),
            "mean_improvement": float(mean_imp),
            "irregularity": float(irregularity),
        }
        
        print(f"{name:<12} MSE: {mean_mse:.6f} (+/- {std_mse:.6f})  Improvement: {mean_imp:+.1f}%")
    
    # Find best
    best = min(results.keys(), key=lambda k: results[k]["mean_mse"])
    worst = max(results.keys(), key=lambda k: results[k]["mean_mse"])
    
    print()
    print(f"BEST:  {best} (MSE: {results[best]['mean_mse']:.6f})")
    print(f"WORST: {worst} (MSE: {results[worst]['mean_mse']:.6f})")
    
    return results, best


if __name__ == "__main__":
    main()

