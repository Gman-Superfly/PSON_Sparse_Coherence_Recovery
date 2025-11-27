"""
PSON Optical Coherence: Large-Scale Scaling Test
====================================================

Scale up sparse optical coherence recovery from 100 elements to 4096+
to find PSON's sweet spot in the optical domain.

Key optimization: Use SPSA (Simultaneous Perturbation) for gradient estimation
instead of finite differences - reduces O(n) to O(2) function evals per iteration.

Usage:
    uv run python experiments/optical_scaling/pson_optical_scaling_test.py
"""

import numpy as np
import json
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class OpticalScalingConfig:
    """Configuration for large-scale optical tests."""
    n_elements: int = 1024
    wavelength: float = 1.0
    n_theta_points: int = 200
    n_iterations: int = 300
    seed: int = 42


def generate_prime_gaps(n: int, seed: int = 42) -> np.ndarray:
    """Generate prime-gap positions for sparse array."""
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i in range(limit + 1) if is_prime[i]]
    
    limit = max(1000, n * 20)
    primes = sieve(limit)
    gaps = np.array([primes[i+1] - primes[i] for i in range(min(n, len(primes)-1))])
    
    if len(gaps) < n:
        rng = np.random.default_rng(seed)
        extra = rng.integers(2, 10, size=n - len(gaps))
        gaps = np.concatenate([gaps, extra])
    
    gaps = gaps[:n].astype(float)
    gaps = gaps / np.mean(gaps) * 2.0
    return gaps


def simulate_interference_vectorized(
    gaps: np.ndarray,
    phases: np.ndarray,
    theta: np.ndarray,
    wavelength: float = 1.0
) -> np.ndarray:
    """Vectorized interference simulation."""
    k = 2 * np.pi / wavelength
    positions = np.cumsum(gaps)
    positions = positions - positions[0]
    
    phi = k * positions[:, np.newaxis] * np.sin(theta[np.newaxis, :]) + phases[:, np.newaxis]
    E = np.exp(1j * phi)
    E_total = np.sum(E, axis=0)
    I = np.abs(E_total) ** 2 / len(gaps) ** 2
    return I


def compute_visibility(I: np.ndarray) -> float:
    """Compute fringe visibility."""
    I_max = np.max(I)
    I_min = np.min(I)
    if I_max + I_min < 1e-10:
        return 0.0
    return (I_max - I_min) / (I_max + I_min)


def generate_target_signal(n: int, signal_type: str, seed: int = 42) -> np.ndarray:
    """Generate target phase signal."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    
    if signal_type == "zeta":
        phases = np.zeros(n)
        for k in range(1, 20):
            phases += np.sin(2 * np.pi * k * t) / k
        phases = phases / (np.max(np.abs(phases)) + 1e-10) * np.pi
    elif signal_type == "turbulence":
        freqs = np.fft.fftfreq(n)
        # Avoid divide by zero - use small epsilon for zero frequencies
        safe_freqs = np.where(np.abs(freqs) > 1e-10, np.abs(freqs), 1e-10)
        spectrum = safe_freqs ** (-5/6)
        spectrum[np.abs(freqs) < 1e-10] = 0  # Zero out DC component
        spectrum = spectrum * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        phases = np.real(np.fft.ifft(spectrum))
        max_phase = np.max(np.abs(phases))
        if max_phase > 1e-10:
            phases = phases / max_phase * np.pi
        else:
            phases = rng.uniform(-np.pi, np.pi, n)
    elif signal_type == "chirp":
        phases = np.pi * np.sin(2 * np.pi * (t + t**2))
    else:
        phases = np.sin(2 * np.pi * t) + 0.5 * np.sin(4 * np.pi * t)
        phases = phases / (np.max(np.abs(phases)) + 1e-10) * np.pi
    
    return phases


def run_pson_optical(
    gaps: np.ndarray,
    target_phases: np.ndarray,
    config: OpticalScalingConfig,
    method: str = "PSON",  # "GD", "PSON", "PSON-Sub"
) -> Tuple[float, float]:
    """
    Run optimization for optical coherence.
    Uses SPSA for efficient gradient estimation.
    
    Returns:
        final_visibility, time_sec
    """
    rng = np.random.default_rng(config.seed)
    n = len(gaps)
    theta = np.linspace(-np.pi/4, np.pi/4, config.n_theta_points)
    
    # Target intensity
    I_target = simulate_interference_vectorized(gaps, target_phases, theta, config.wavelength)
    
    def energy(ph):
        I = simulate_interference_vectorized(gaps, ph, theta, config.wavelength)
        return float(np.mean((I - I_target) ** 2))
    
    # Initialize
    phases = rng.uniform(-np.pi, np.pi, n)
    
    # Precision from gap irregularity
    gap_mean = np.mean(gaps)
    irregularity = np.abs(gaps - gap_mean) / (gap_mean + 1e-8)
    precision = 1.0 / (1.0 + irregularity)
    
    best_phases = phases.copy()
    best_energy = energy(phases)
    
    t0 = time.perf_counter()
    
    # Setup for subspace method
    if method == "PSON-Sub" and n > 128:
        k_sub = min(64, n // 8)
        # Random orthonormal basis
        V = rng.standard_normal((n, k_sub))
        V, _ = np.linalg.qr(V)
    
    for it in range(config.n_iterations):
        cur_energy = energy(phases)
        
        if cur_energy < best_energy:
            best_energy = cur_energy
            best_phases = phases.copy()
        
        # SPSA gradient estimation (O(2) function evals instead of O(n))
        eps = 0.1 * (1.0 / (1 + it / 100))  # Decaying perturbation
        
        if method == "PSON-Sub" and n > 128:
            # Work in subspace
            delta = rng.choice([-1, 1], k_sub)
            delta_full = V @ delta
            
            e_plus = energy(phases + eps * delta_full)
            e_minus = energy(phases - eps * delta_full)
            grad_spsa = (e_plus - e_minus) / (2 * eps) * delta
            
            # Step in subspace
            lr = 0.3
            phases_sub = V.T @ phases
            proposal_sub = phases_sub - lr * grad_spsa
            
            if method == "PSON-Sub":
                # PSON exploration in subspace
                energy_ratio = min(cur_energy / (best_energy + 1e-10), 2.0)
                noise_scale = 0.2 * energy_ratio
                noise_sub = noise_scale * rng.standard_normal(k_sub)
                candidate_sub = proposal_sub + noise_sub
            else:
                candidate_sub = proposal_sub
            
            # Lift back
            candidate = V @ candidate_sub + (phases - V @ (V.T @ phases))
            
        else:
            # Standard method with SPSA
            delta = rng.choice([-1, 1], n)
            e_plus = energy(phases + eps * delta)
            e_minus = energy(phases - eps * delta)
            grad_spsa = (e_plus - e_minus) / (2 * eps + 1e-10) * delta
            
            lr = 0.2
            proposal = phases - lr * grad_spsa
            
            if method in ["PSON", "PSON-Sub"]:
                # PSON exploration
                energy_ratio = min(cur_energy / (best_energy + 1e-10), 2.0)
                change_probs = (1.0 - precision) * (0.2 + 0.8 * energy_ratio)
                change_probs = change_probs / (np.sum(change_probs) + 1e-10)
                
                n_changes = max(1, int(n * 0.2))
                change_idx = rng.choice(n, min(n_changes, n), replace=False, p=change_probs)
                
                noise_scale = 0.3 * energy_ratio
                noise = noise_scale * rng.standard_normal(len(change_idx))
                
                candidate = proposal.copy()
                candidate[change_idx] += noise / (np.sqrt(precision[change_idx]) + 0.1)
            else:
                candidate = proposal
        
        # Wrap phases
        candidate = np.mod(candidate + np.pi, 2 * np.pi) - np.pi
        
        # Monotonic acceptance (for PSON methods)
        if method == "GD":
            phases = candidate
        else:
            cand_energy = energy(candidate)
            if cand_energy <= cur_energy:
                phases = candidate
    
    elapsed = time.perf_counter() - t0
    
    # Final visibility
    I_final = simulate_interference_vectorized(gaps, best_phases, theta, config.wavelength)
    final_vis = compute_visibility(I_final)
    
    return final_vis, elapsed


def main():
    print("=" * 80)
    print("PSON Optical Coherence: Large-Scale Scaling (SPSA)")
    print("=" * 80)
    
    sizes = [100, 256, 512, 1024, 2048, 4096]
    signal_types = ["zeta", "turbulence", "chirp"]
    seeds = [42, 123, 456]
    methods = ["GD", "PSON", "PSON-Sub"]
    
    all_results = {}
    
    for n_elements in sizes:
        print(f"\n{'='*60}")
        print(f"Testing {n_elements} elements...")
        print(f"{'='*60}")
        
        config = OpticalScalingConfig(
            n_elements=n_elements,
            n_iterations=300,
        )
        
        results_for_size = {"n_elements": n_elements, "signals": {}}
        
        for signal_type in signal_types:
            print(f"\n  Signal: {signal_type}")
            
            method_results = {m: {"vis": [], "time": []} for m in methods}
            
            for seed in seeds:
                config.seed = seed
                gaps = generate_prime_gaps(n_elements, seed)
                target = generate_target_signal(n_elements, signal_type, seed)
                
                for method in methods:
                    vis, elapsed = run_pson_optical(gaps, target, config, method=method)
                    method_results[method]["vis"].append(vis)
                    method_results[method]["time"].append(elapsed)
            
            # Compute stats and print
            signal_results = {}
            for method in methods:
                mean_vis = float(np.mean(method_results[method]["vis"]))
                std_vis = float(np.std(method_results[method]["vis"]))
                mean_time = float(np.mean(method_results[method]["time"]))
                signal_results[method] = {
                    "mean_vis": mean_vis,
                    "std_vis": std_vis,
                    "mean_time": mean_time,
                }
                print(f"    {method:<10}: V={mean_vis:.4f} +/- {std_vis:.4f}, T={mean_time:.2f}s")
            
            # Winner
            vis_means = {m: signal_results[m]["mean_vis"] for m in methods}
            winner = max(vis_means.items(), key=lambda x: x[1])[0]
            signal_results["winner"] = winner
            print(f"    Winner: {winner}")
            
            results_for_size["signals"][signal_type] = signal_results
        
        all_results[f"n_{n_elements}"] = results_for_size
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Win Rate by Array Size")
    print("=" * 80)
    
    print(f"\n{'Size':<8} {'GD':<10} {'PSON':<10} {'PSON-Sub':<12} {'Best'}")
    print("-" * 50)
    
    for size_key, results in all_results.items():
        n = results["n_elements"]
        wins = {m: 0 for m in methods}
        for sig_data in results["signals"].values():
            wins[sig_data["winner"]] += 1
        
        best = max(wins.items(), key=lambda x: x[1])[0]
        print(f"{n:<8} {wins['GD']:<10} {wins['PSON']:<10} {wins['PSON-Sub']:<12} {best}")
    
    # Total PSON variants win rate
    total_tests = len(sizes) * len(signal_types)
    pson_wins = sum(
        1 for r in all_results.values()
        for sig_data in r["signals"].values()
        if sig_data["winner"] in ["PSON", "PSON-Sub"]
    )
    print(f"\nPSON (any variant) total wins: {pson_wins}/{total_tests} ({100*pson_wins/total_tests:.0f}%)")
    
    # Save
    with open("pson_optical_scaling_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved: pson_optical_scaling_results.json")


if __name__ == "__main__":
    main()
