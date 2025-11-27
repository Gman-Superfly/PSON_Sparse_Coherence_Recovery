"""
Energy Evaluation Speedup Test
==============================

Tests various optimizations for the interference simulation bottleneck.

CURRENT BOTTLENECK: Python for-loop over gaps in simulate_interference()

OPTIMIZATIONS TESTED:
1. Vectorized numpy (no Python loops)
2. Reduced θ resolution (200 → 50 points)
3. Combined vectorized + reduced resolution
4. Numba JIT compilation
5. Early rejection (abort if energy already exceeds threshold)

Usage:
    uv run python experiments/speedup_test.py
"""

import numpy as np
import time
from typing import Tuple, Callable
import json

# Try to import numba (optional)
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Numba not available - skipping JIT tests")


# =============================================================================
# Original Implementation (SLOW - for-loop)
# =============================================================================

def simulate_original(gaps: np.ndarray, phases: np.ndarray, n_points: int = 200) -> np.ndarray:
    """Original implementation with Python for-loop."""
    k = 2 * np.pi / 0.5
    theta = np.linspace(-0.01, 0.01, n_points)
    
    I_total = np.zeros(n_points)
    for i, (gap, phase) in enumerate(zip(gaps, phases)):
        phi = k * gap * np.sin(theta) + phase
        E = np.abs(0.5 + 0.5 * np.exp(1j * phi))**2
        I_total += E
    
    return I_total / len(gaps)


def visibility(I: np.ndarray) -> float:
    """Calculate visibility."""
    return (I.max() - I.min()) / (I.max() + I.min() + 1e-12)


def energy_original(gaps: np.ndarray, phases: np.ndarray) -> float:
    """Original energy function."""
    I = simulate_original(gaps, phases)
    return (1.0 - visibility(I)) ** 2


# =============================================================================
# Optimization 1: Fully Vectorized (no Python loops)
# =============================================================================

def simulate_vectorized(gaps: np.ndarray, phases: np.ndarray, n_points: int = 200) -> np.ndarray:
    """Fully vectorized - no Python loops."""
    k = 2 * np.pi / 0.5
    theta = np.linspace(-0.01, 0.01, n_points)
    
    # Broadcasting: gaps (N,1) * theta (1,M) -> phi (N,M)
    phi = k * gaps[:, np.newaxis] * np.sin(theta[np.newaxis, :]) + phases[:, np.newaxis]
    
    # Compute intensity for all gaps at once
    E = np.abs(0.5 + 0.5 * np.exp(1j * phi))**2  # Shape: (N, M)
    
    # Average over gaps
    I_total = np.mean(E, axis=0)  # Shape: (M,)
    
    return I_total


def energy_vectorized(gaps: np.ndarray, phases: np.ndarray) -> float:
    """Vectorized energy function."""
    I = simulate_vectorized(gaps, phases)
    return (1.0 - visibility(I)) ** 2


# =============================================================================
# Optimization 2: Reduced Resolution
# =============================================================================

def simulate_reduced(gaps: np.ndarray, phases: np.ndarray, n_points: int = 50) -> np.ndarray:
    """Reduced θ resolution (50 instead of 200)."""
    return simulate_original(gaps, phases, n_points=n_points)


def energy_reduced(gaps: np.ndarray, phases: np.ndarray) -> float:
    """Reduced resolution energy."""
    I = simulate_reduced(gaps, phases)
    return (1.0 - visibility(I)) ** 2


# =============================================================================
# Optimization 3: Vectorized + Reduced
# =============================================================================

def simulate_fast(gaps: np.ndarray, phases: np.ndarray, n_points: int = 50) -> np.ndarray:
    """Vectorized + reduced resolution."""
    k = 2 * np.pi / 0.5
    theta = np.linspace(-0.01, 0.01, n_points)
    
    phi = k * gaps[:, np.newaxis] * np.sin(theta[np.newaxis, :]) + phases[:, np.newaxis]
    E = np.abs(0.5 + 0.5 * np.exp(1j * phi))**2
    
    return np.mean(E, axis=0)


def energy_fast(gaps: np.ndarray, phases: np.ndarray) -> float:
    """Fast energy (vectorized + reduced)."""
    I = simulate_fast(gaps, phases)
    return (1.0 - visibility(I)) ** 2


# =============================================================================
# Optimization 4: Numba JIT (if available)
# =============================================================================

if HAS_NUMBA:
    @jit(nopython=True, fastmath=True)
    def simulate_numba(gaps: np.ndarray, phases: np.ndarray, n_points: int = 200) -> np.ndarray:
        """Numba JIT-compiled version."""
        k = 2 * np.pi / 0.5
        theta = np.linspace(-0.01, 0.01, n_points)
        
        I_total = np.zeros(n_points)
        for i in range(len(gaps)):
            for j in range(n_points):
                phi = k * gaps[i] * np.sin(theta[j]) + phases[i]
                E = (0.5 + 0.5 * np.cos(phi))**2 + (0.5 * np.sin(phi))**2
                I_total[j] += E
        
        return I_total / len(gaps)
    
    def energy_numba(gaps: np.ndarray, phases: np.ndarray) -> float:
        """Numba energy."""
        I = simulate_numba(gaps, phases)
        return (1.0 - visibility(I)) ** 2


# =============================================================================
# Optimization 5: Precomputed + Cached
# =============================================================================

class CachedSimulator:
    """Simulator with precomputed constants and optional caching."""
    
    def __init__(self, n_points: int = 200, cache_size: int = 100):
        self.n_points = n_points
        self.k = 2 * np.pi / 0.5
        self.theta = np.linspace(-0.01, 0.01, n_points)
        self.sin_theta = np.sin(self.theta)
        self._cache = {}
        self._cache_size = cache_size
    
    def simulate(self, gaps: np.ndarray, phases: np.ndarray) -> np.ndarray:
        """Simulate with precomputed constants."""
        phi = self.k * gaps[:, np.newaxis] * self.sin_theta[np.newaxis, :] + phases[:, np.newaxis]
        E = np.abs(0.5 + 0.5 * np.exp(1j * phi))**2
        return np.mean(E, axis=0)
    
    def energy(self, gaps: np.ndarray, phases: np.ndarray) -> float:
        """Energy with precomputed constants."""
        I = self.simulate(gaps, phases)
        return (1.0 - visibility(I)) ** 2


# =============================================================================
# Optimization 6: Ultra-fast (minimal resolution + vectorized)
# =============================================================================

def simulate_ultra(gaps: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """Ultra-fast: only 20 θ points, fully vectorized."""
    k = 2 * np.pi / 0.5
    theta = np.linspace(-0.01, 0.01, 20)
    
    phi = k * gaps[:, np.newaxis] * np.sin(theta) + phases[:, np.newaxis]
    E = np.abs(0.5 + 0.5 * np.exp(1j * phi))**2
    
    return np.mean(E, axis=0)


def energy_ultra(gaps: np.ndarray, phases: np.ndarray) -> float:
    """Ultra-fast energy."""
    I = simulate_ultra(gaps, phases)
    return (1.0 - visibility(I)) ** 2


# =============================================================================
# Benchmark
# =============================================================================

def benchmark_energy(name: str, energy_fn: Callable, gaps: np.ndarray, n_evals: int = 1000) -> dict:
    """Benchmark an energy function."""
    rng = np.random.default_rng(42)
    
    # Warmup
    for _ in range(10):
        phases = rng.uniform(0, 2*np.pi, len(gaps))
        _ = energy_fn(gaps, phases)
    
    # Timed runs
    start = time.perf_counter()
    energies = []
    for _ in range(n_evals):
        phases = rng.uniform(0, 2*np.pi, len(gaps))
        e = energy_fn(gaps, phases)
        energies.append(e)
    elapsed = time.perf_counter() - start
    
    return {
        "name": name,
        "total_time": elapsed,
        "time_per_eval_ms": 1000 * elapsed / n_evals,
        "evals_per_sec": n_evals / elapsed,
        "mean_energy": float(np.mean(energies)),
    }


def main():
    print("=" * 70)
    print("ENERGY EVALUATION SPEEDUP TEST")
    print("=" * 70)
    
    # Test configuration
    n_gaps = 25
    n_evals = 1000
    
    # Generate prime-gap array
    def gen_primes(n):
        primes = []
        candidate = 2
        while len(primes) < n:
            is_prime = all(candidate % p != 0 for p in primes)
            if is_prime:
                primes.append(candidate)
            candidate += 1
        return primes
    
    gaps = np.array([p * 10.0 for p in gen_primes(n_gaps)])
    
    print(f"\nConfiguration: {n_gaps} gaps, {n_evals} evaluations each")
    print("-" * 70)
    
    # Benchmark all methods
    methods = [
        ("Original (for-loop, 200 pts)", lambda g, p: energy_original(g, p)),
        ("Vectorized (200 pts)", lambda g, p: energy_vectorized(g, p)),
        ("Reduced (for-loop, 50 pts)", lambda g, p: energy_reduced(g, p)),
        ("Fast (vectorized, 50 pts)", lambda g, p: energy_fast(g, p)),
        ("Ultra (vectorized, 20 pts)", lambda g, p: energy_ultra(g, p)),
    ]
    
    # Add cached version
    cached_sim = CachedSimulator(n_points=200)
    methods.append(("Cached (precomputed, 200 pts)", lambda g, p: cached_sim.energy(g, p)))
    
    cached_sim_fast = CachedSimulator(n_points=50)
    methods.append(("Cached Fast (precomputed, 50 pts)", lambda g, p: cached_sim_fast.energy(g, p)))
    
    # Add Numba if available
    if HAS_NUMBA:
        # Warmup JIT compilation
        phases_warmup = np.zeros(n_gaps)
        _ = energy_numba(gaps, phases_warmup)
        methods.append(("Numba JIT (200 pts)", lambda g, p: energy_numba(g, p)))
    
    results = []
    baseline_time = None
    
    for name, fn in methods:
        result = benchmark_energy(name, fn, gaps, n_evals)
        results.append(result)
        
        if baseline_time is None:
            baseline_time = result["time_per_eval_ms"]
            speedup = 1.0
        else:
            speedup = baseline_time / result["time_per_eval_ms"]
        
        print(f"{name:<35} {result['time_per_eval_ms']:>8.3f} ms/eval  {speedup:>6.1f}x  E={result['mean_energy']:.4f}")
    
    # Check accuracy
    print("\n" + "=" * 70)
    print("ACCURACY CHECK (energy values should be similar)")
    print("=" * 70)
    
    rng = np.random.default_rng(123)
    test_phases = rng.uniform(0, 2*np.pi, n_gaps)
    
    baseline_energy = energy_original(gaps, test_phases)
    print(f"\nBaseline energy: {baseline_energy:.6f}")
    
    for name, fn in methods[1:]:  # Skip baseline
        e = fn(gaps, test_phases)
        error = abs(e - baseline_energy) / (baseline_energy + 1e-10) * 100
        status = "OK" if error < 5 else "WARN" if error < 20 else "BAD"
        print(f"{name:<35} E={e:.6f}  error={error:.2f}%  [{status}]")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    best = min(results, key=lambda r: r["time_per_eval_ms"])
    print(f"\nFastest: {best['name']}")
    print(f"  Time: {best['time_per_eval_ms']:.3f} ms/eval")
    print(f"  Speedup: {baseline_time / best['time_per_eval_ms']:.1f}x over original")
    print(f"  Throughput: {best['evals_per_sec']:.0f} evals/sec")
    
    # Save results
    output = {
        "config": {"n_gaps": n_gaps, "n_evals": n_evals},
        "results": results,
        "best": best["name"],
        "best_speedup": baseline_time / best["time_per_eval_ms"],
    }
    
    with open("speedup_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved: speedup_results.json")


if __name__ == "__main__":
    main()

