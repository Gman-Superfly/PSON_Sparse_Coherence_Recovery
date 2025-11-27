"""
Spatial Light Modulator (SLM) Test
==================================

Tests PSON on phase-only hologram optimization.

PROBLEM:
- NxN grid of phase pixels (typical: 512x512 to 1920x1080)
- Each pixel has discrete phase levels (typically 8-bit = 256 levels)
- Goal: Shape output beam to match target intensity pattern

DISCRETE CONSTRAINT:
- Phase levels: 0, 2π/K, 4π/K, ..., 2π(K-1)/K
- We test with reduced resolution and fewer levels for speed

PSON ADVANTAGE:
- Handles irregular pixel responses (dead pixels, nonlinear response)
- Non-local credit assignment for pattern matching
- Doesn't require gradient through FFT

Usage:
    uv run python experiments/discrete_applications/spatial_light_modulator_test.py
"""

import numpy as np
import json
from typing import Tuple, List

# =============================================================================
# SLM Physics (Fourier Optics)
# =============================================================================

def slm_to_farfield(phases: np.ndarray) -> np.ndarray:
    """
    Compute far-field intensity from SLM phase pattern.
    Uses Fraunhofer approximation: far-field ∝ |FFT(exp(iφ))|²
    """
    field = np.exp(1j * phases)
    farfield = np.fft.fftshift(np.fft.fft2(field))
    return np.abs(farfield) ** 2


def pattern_similarity(output: np.ndarray, target: np.ndarray) -> float:
    """
    Compute similarity between output and target patterns.
    Uses normalized correlation (higher is better, max 1.0).
    """
    output_norm = output / (np.linalg.norm(output) + 1e-10)
    target_norm = target / (np.linalg.norm(target) + 1e-10)
    return float(np.sum(output_norm * target_norm))


def efficiency(output: np.ndarray, target: np.ndarray) -> float:
    """
    Compute diffraction efficiency: power in target region / total power.
    """
    # Mask target region (where target > 10% of max)
    mask = target > 0.1 * np.max(target)
    power_in_target = np.sum(output[mask])
    total_power = np.sum(output)
    return float(power_in_target / (total_power + 1e-10))


# =============================================================================
# Target Patterns
# =============================================================================

def create_target_spot(size: int, position: Tuple[int, int] = None) -> np.ndarray:
    """Create a focused spot target."""
    target = np.zeros((size, size))
    if position is None:
        position = (size // 2, size // 2)
    # Gaussian spot
    y, x = np.ogrid[:size, :size]
    r2 = (x - position[1])**2 + (y - position[0])**2
    sigma = size / 20
    target = np.exp(-r2 / (2 * sigma**2))
    return target


def create_target_line(size: int, angle: float = 0) -> np.ndarray:
    """Create a line pattern target."""
    target = np.zeros((size, size))
    center = size // 2
    # Line through center at given angle
    for i in range(size):
        x = int(center + (i - center) * np.cos(angle))
        y = int(center + (i - center) * np.sin(angle))
        if 0 <= x < size and 0 <= y < size:
            target[y, x] = 1.0
    # Blur slightly
    from scipy.ndimage import gaussian_filter
    target = gaussian_filter(target, sigma=1)
    return target


def create_target_ring(size: int, radius: float = None) -> np.ndarray:
    """Create a ring pattern target."""
    if radius is None:
        radius = size / 6
    y, x = np.ogrid[:size, :size]
    center = size // 2
    r = np.sqrt((x - center)**2 + (y - center)**2)
    target = np.exp(-((r - radius) / 2)**2)
    return target


# =============================================================================
# PSON SLM Optimizer
# =============================================================================

def pson_slm(
    target: np.ndarray,
    n_levels: int = 16,  # 4-bit phase
    steps: int = 200,
    seed: int = 42,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    PSON optimization for SLM phase pattern.
    """
    rng = np.random.default_rng(seed)
    size = target.shape[0]
    n_pixels = size * size
    
    # Initialize with random phases (quantized)
    phase_levels = np.array([i * 2 * np.pi / n_levels for i in range(n_levels)])
    phases = rng.choice(phase_levels, (size, size))
    
    # Weights: uniform (no irregularity info for SLM)
    # But we can weight by target intensity (prioritize important regions)
    target_norm = target / (np.max(target) + 1e-10)
    weights = target_norm.flatten()
    weights = weights / (weights.sum() + 1e-8)
    if weights.sum() < 1e-8:
        weights = np.ones(n_pixels) / n_pixels
    
    def compute_objective(ph):
        output = slm_to_farfield(ph)
        return pattern_similarity(output, target)
    
    best_phases = phases.copy()
    best_obj = compute_objective(phases)
    obj_curve = []
    
    for step in range(steps):
        obj_cur = compute_objective(phases)
        obj_curve.append(obj_cur)
        
        if obj_cur > best_obj:
            best_obj = obj_cur
            best_phases = phases.copy()
        
        energy = 1.0 - obj_cur
        
        # PSON: weighted exploration
        change_probs = weights * (0.2 + 0.8 * energy)
        change_probs = change_probs / change_probs.sum()
        
        # Change more pixels when far from optimum
        n_changes = max(1, int(n_pixels * 0.05 * (0.5 + energy)))
        change_idx = rng.choice(n_pixels, min(n_changes, n_pixels), replace=False, p=change_probs)
        
        candidate = phases.copy().flatten()
        for idx in change_idx:
            current_level = int(round(candidate[idx] * n_levels / (2 * np.pi))) % n_levels
            if rng.random() < 0.6:
                delta = rng.choice([-1, 1])
                new_level = (current_level + delta) % n_levels
            else:
                new_level = rng.integers(0, n_levels)
            candidate[idx] = phase_levels[new_level]
        candidate = candidate.reshape((size, size))
        
        obj_new = compute_objective(candidate)
        if obj_new >= obj_cur:
            phases = candidate
    
    return best_phases, best_obj, obj_curve


def gerchberg_saxton_discrete(
    target: np.ndarray,
    n_levels: int = 16,
    steps: int = 200,
    seed: int = 42,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Gerchberg-Saxton algorithm with phase quantization.
    Classic iterative Fourier transform algorithm.
    """
    rng = np.random.default_rng(seed)
    size = target.shape[0]
    
    phase_levels = np.array([i * 2 * np.pi / n_levels for i in range(n_levels)])
    
    # Initialize with random phase
    phases = rng.choice(phase_levels, (size, size))
    
    def quantize(p):
        idx = np.round(p * n_levels / (2 * np.pi)).astype(int) % n_levels
        return phase_levels[idx]
    
    def compute_objective(ph):
        output = slm_to_farfield(ph)
        return pattern_similarity(output, target)
    
    # Input plane: unit amplitude
    input_amp = np.ones((size, size))
    target_amp = np.sqrt(target + 1e-10)  # Target amplitude
    
    obj_curve = []
    best_obj = 0
    best_phases = phases.copy()
    
    for step in range(steps):
        # Forward propagate
        field = input_amp * np.exp(1j * phases)
        farfield = np.fft.fftshift(np.fft.fft2(field))
        
        # Apply target amplitude constraint
        farfield_phase = np.angle(farfield)
        farfield = target_amp * np.exp(1j * farfield_phase)
        
        # Backward propagate
        nearfield = np.fft.ifft2(np.fft.ifftshift(farfield))
        
        # Apply input amplitude constraint and quantize phase
        phases = quantize(np.angle(nearfield))
        
        obj = compute_objective(phases)
        obj_curve.append(obj)
        
        if obj > best_obj:
            best_obj = obj
            best_phases = phases.copy()
    
    return best_phases, best_obj, obj_curve


def random_search_slm(
    target: np.ndarray,
    n_levels: int = 16,
    steps: int = 200,
    seed: int = 42,
) -> Tuple[np.ndarray, float, List[float]]:
    """Random search baseline."""
    rng = np.random.default_rng(seed)
    size = target.shape[0]
    
    phase_levels = np.array([i * 2 * np.pi / n_levels for i in range(n_levels)])
    
    def compute_objective(ph):
        output = slm_to_farfield(ph)
        return pattern_similarity(output, target)
    
    best_phases = rng.choice(phase_levels, (size, size))
    best_obj = compute_objective(best_phases)
    obj_curve = []
    
    for step in range(steps):
        phases = rng.choice(phase_levels, (size, size))
        obj = compute_objective(phases)
        obj_curve.append(best_obj)
        
        if obj > best_obj:
            best_obj = obj
            best_phases = phases.copy()
    
    return best_phases, best_obj, obj_curve


# =============================================================================
# Main Test
# =============================================================================

def main():
    print("=" * 70)
    print("SPATIAL LIGHT MODULATOR TEST: PSON vs GS vs Random Search")
    print("=" * 70)
    
    # Small size for speed (real SLMs are 512x512 or larger)
    size = 32
    n_levels = 16  # 4-bit
    steps = 200
    seeds = [42, 123, 456]
    
    # Target patterns
    targets = {
        "spot_center": create_target_spot(size),
        "spot_offset": create_target_spot(size, (size//2 + 5, size//2 + 5)),
        "ring": create_target_ring(size),
    }
    
    results = []
    
    for target_name, target in targets.items():
        print(f"\nTarget: {target_name}")
        print("-" * 50)
        
        target_results = {"target": target_name, "runs": []}
        
        for seed in seeds:
            # PSON
            _, pson_sim, _ = pson_slm(target, n_levels=n_levels, steps=steps, seed=seed)
            
            # Gerchberg-Saxton
            _, gs_sim, _ = gerchberg_saxton_discrete(target, n_levels=n_levels, steps=steps, seed=seed)
            
            # Random Search
            _, rs_sim, _ = random_search_slm(target, n_levels=n_levels, steps=steps, seed=seed)
            
            winner = max([("PSON", pson_sim), ("GS", gs_sim), ("RS", rs_sim)], key=lambda x: x[1])[0]
            
            print(f"  Seed {seed}: PSON={pson_sim:.4f}, GS={gs_sim:.4f}, RS={rs_sim:.4f}, Winner: {winner}")
            
            target_results["runs"].append({
                "seed": seed,
                "pson": float(pson_sim),
                "gs": float(gs_sim),
                "rs": float(rs_sim),
                "winner": winner,
            })
        
        pson_wins = sum(1 for r in target_results["runs"] if r["winner"] == "PSON")
        target_results["summary"] = {
            "pson_mean": float(np.mean([r["pson"] for r in target_results["runs"]])),
            "gs_mean": float(np.mean([r["gs"] for r in target_results["runs"]])),
            "rs_mean": float(np.mean([r["rs"] for r in target_results["runs"]])),
            "pson_wins": pson_wins,
        }
        
        results.append(target_results)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_pson_wins = sum(r["summary"]["pson_wins"] for r in results)
    total_runs = sum(len(r["runs"]) for r in results)
    print(f"PSON wins: {total_pson_wins}/{total_runs}")
    
    for r in results:
        print(f"  {r['target']}: PSON={r['summary']['pson_mean']:.4f}, GS={r['summary']['gs_mean']:.4f}")
    
    # Save
    with open("slm_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved: slm_results.json")


if __name__ == "__main__":
    main()

