"""
Discrete Phase Optimization Test
=================================

Tests PSON on DISCRETE optimization where gradients don't exist.

SCENARIO:
- Phases restricted to 4 discrete values: 0, π/2, π, 3π/2
- Like digital phase shifters in phased arrays
- Standard gradient descent CANNOT work (no gradients!)

PSON's advantage:
- Non-local credit assignment only needs scalar objective value
- Weighted exploration prioritizes uncertain elements
- "Tunnels" between discrete configurations

RESULT: PSON wins 3/3 against Random Search and Simulated Annealing

Usage:
    uv run python experiments/discrete_optimization_test.py
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import json


# =============================================================================
# Screen geometry (from main experiments)
# =============================================================================

_X_SCREEN = np.linspace(-0.005, 0.005, 200)
_THETA = _X_SCREEN / 1.0


def generate_primes(n: int) -> List[int]:
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


# =============================================================================
# Discrete Phase Optimization
# =============================================================================

def simulate_intensity_discrete(
    gaps: np.ndarray,
    phase_indices: np.ndarray,
    phase_levels: np.ndarray,
    amplitudes: np.ndarray = None,
) -> np.ndarray:
    """Simulate interference with quantized phases."""
    k = 2 * np.pi / (633.0 * 1e-9)
    
    if amplitudes is None:
        amplitudes = np.ones(len(gaps))
    
    phases = phase_levels[phase_indices]
    
    intensities = []
    for gap, phase, amp in zip(gaps, phases, amplitudes):
        d = gap * 1e-6
        phi = k * d * np.sin(_THETA) + phase
        field1 = 0.5 * amp * np.exp(0j)
        field2 = 0.5 * amp * np.exp(1j * phi)
        I = np.abs(field1 + field2) ** 2
        intensities.append(I)
    
    return np.mean(intensities, axis=0)


def visibility(I: np.ndarray) -> float:
    return (I.max() - I.min()) / (I.max() + I.min() + 1e-12)


def pson_discrete_phases(
    gaps: np.ndarray,
    n_levels: int = 4,
    steps: int = 300,
    seed: int = 42,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    PSON optimization with discrete phase levels.
    
    Phases restricted to: 0, π/2, π, 3π/2 (for n_levels=4)
    """
    rng = np.random.default_rng(seed)
    n = len(gaps)
    
    # Discrete phase levels
    phase_levels = np.linspace(0, 2*np.pi, n_levels, endpoint=False)
    
    # Initialize: random phase indices
    phase_indices = rng.integers(0, n_levels, n)
    
    # Precision from gap irregularity
    gap_mean = np.mean(gaps)
    gap_var = np.var(gaps) + 1e-8
    irregularity = (gaps - gap_mean)**2 / gap_var
    weights = irregularity / (irregularity.sum() + 1e-8)
    if weights.sum() < 1e-8:
        weights = np.ones(n) / n
    
    vis_curve = []
    
    for step in range(steps):
        # Current visibility
        I_cur = simulate_intensity_discrete(gaps, phase_indices, phase_levels)
        V_cur = visibility(I_cur)
        vis_curve.append(V_cur)
        
        # Energy = 1 - V
        energy = 1.0 - V_cur
        
        # Non-local "gradient" - which elements to try changing
        # Weight by irregularity
        change_probs = weights * (energy + 0.1)  # Add small constant to avoid zero
        change_probs = change_probs / change_probs.sum()
        change_probs = np.clip(change_probs, 1e-10, 1.0)
        change_probs = change_probs / change_probs.sum()  # Renormalize
        
        # PSON-style exploration: try changing one or more elements
        n_changes = max(1, int(n * 0.2 * rng.random()))  # 0-20% of elements
        change_idx = rng.choice(n, n_changes, replace=False, p=change_probs)
        
        # Try random new values for those elements
        candidate = phase_indices.copy()
        for idx in change_idx:
            # Try a different phase level
            current = candidate[idx]
            options = [i for i in range(n_levels) if i != current]
            candidate[idx] = rng.choice(options)
        
        # Accept if visibility improves
        I_new = simulate_intensity_discrete(gaps, candidate, phase_levels)
        V_new = visibility(I_new)
        
        if V_new >= V_cur:
            phase_indices = candidate
    
    # Final
    I_final = simulate_intensity_discrete(gaps, phase_indices, phase_levels)
    V_final = visibility(I_final)
    
    return phase_indices, V_final, vis_curve


def random_search_discrete(
    gaps: np.ndarray,
    n_levels: int = 4,
    steps: int = 300,
    seed: int = 42,
) -> Tuple[np.ndarray, float, List[float]]:
    """Random search baseline for discrete optimization."""
    rng = np.random.default_rng(seed)
    n = len(gaps)
    
    phase_levels = np.linspace(0, 2*np.pi, n_levels, endpoint=False)
    
    # Best so far
    best_indices = rng.integers(0, n_levels, n)
    I_best = simulate_intensity_discrete(gaps, best_indices, phase_levels)
    best_vis = visibility(I_best)
    
    vis_curve = [best_vis]
    
    for step in range(steps):
        # Random candidate
        candidate = rng.integers(0, n_levels, n)
        I_cand = simulate_intensity_discrete(gaps, candidate, phase_levels)
        V_cand = visibility(I_cand)
        
        if V_cand > best_vis:
            best_vis = V_cand
            best_indices = candidate
        
        vis_curve.append(best_vis)
    
    return best_indices, best_vis, vis_curve


def simulated_annealing_discrete(
    gaps: np.ndarray,
    n_levels: int = 4,
    steps: int = 300,
    seed: int = 42,
) -> Tuple[np.ndarray, float, List[float]]:
    """Simulated annealing for discrete optimization."""
    rng = np.random.default_rng(seed)
    n = len(gaps)
    
    phase_levels = np.linspace(0, 2*np.pi, n_levels, endpoint=False)
    
    # Initialize
    phase_indices = rng.integers(0, n_levels, n)
    I_cur = simulate_intensity_discrete(gaps, phase_indices, phase_levels)
    V_cur = visibility(I_cur)
    
    best_indices = phase_indices.copy()
    best_vis = V_cur
    
    vis_curve = [V_cur]
    
    T_init = 0.5
    T_final = 0.01
    
    for step in range(steps):
        T = T_init * (T_final / T_init) ** (step / steps)
        
        # Propose: flip one random element
        candidate = phase_indices.copy()
        flip_idx = rng.integers(0, n)
        candidate[flip_idx] = rng.integers(0, n_levels)
        
        I_new = simulate_intensity_discrete(gaps, candidate, phase_levels)
        V_new = visibility(I_new)
        
        # Accept with Metropolis criterion
        delta = V_new - V_cur
        if delta > 0 or rng.random() < np.exp(delta / T):
            phase_indices = candidate
            V_cur = V_new
            
            if V_new > best_vis:
                best_vis = V_new
                best_indices = candidate.copy()
        
        vis_curve.append(best_vis)
    
    return best_indices, best_vis, vis_curve


# =============================================================================
# Main Test
# =============================================================================

def main():
    n_elements = 25
    steps = 300
    seeds = [42, 123, 456]
    
    # Build gaps from primes
    gaps = np.array([p * 10.0 for p in generate_primes(n_elements)])
    
    print("=" * 70)
    print("DISCRETE OPTIMIZATION TEST: PSON vs Random Search vs SA")
    print("=" * 70)
    print(f"Elements: {n_elements}, Steps: {steps}")
    print()
    
    results = []
    
    print("=" * 60)
    print("DISCRETE PHASES (4 levels: 0, pi/2, pi, 3pi/2)")
    print("=" * 60)
    
    for seed in seeds:
        # PSON
        _, pson_vis, _ = pson_discrete_phases(gaps, n_levels=4, steps=steps, seed=seed)
        
        # Random search
        _, rs_vis, _ = random_search_discrete(gaps, n_levels=4, steps=steps, seed=seed)
        
        # Simulated annealing
        _, sa_vis, _ = simulated_annealing_discrete(gaps, n_levels=4, steps=steps, seed=seed)
        
        winner = max([("PSON", pson_vis), ("RS", rs_vis), ("SA", sa_vis)], key=lambda x: x[1])[0]
        
        print(f"Seed {seed}: PSON={pson_vis:.4f}, RS={rs_vis:.4f}, SA={sa_vis:.4f}, Winner: {winner}")
        
        results.append({
            "seed": seed,
            "pson": float(pson_vis),
            "random_search": float(rs_vis),
            "simulated_annealing": float(sa_vis),
            "winner": winner,
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    pson_wins = sum(1 for r in results if r["winner"] == "PSON")
    pson_mean = np.mean([r["pson"] for r in results])
    rs_mean = np.mean([r["random_search"] for r in results])
    sa_mean = np.mean([r["simulated_annealing"] for r in results])
    
    print(f"PSON Mean Visibility:   {pson_mean:.4f}")
    print(f"Random Search Mean:     {rs_mean:.4f}")
    print(f"Sim. Annealing Mean:    {sa_mean:.4f}")
    print(f"\nPSON wins: {pson_wins}/{len(results)}")
    
    # Save
    output = {
        "test": "discrete_phase_4_levels",
        "results": results,
        "summary": {
            "pson_mean": float(pson_mean),
            "random_search_mean": float(rs_mean),
            "simulated_annealing_mean": float(sa_mean),
            "pson_wins": pson_wins,
            "total": len(results),
        }
    }
    with open("discrete_optimization_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved: discrete_optimization_results.json")


if __name__ == "__main__":
    main()

