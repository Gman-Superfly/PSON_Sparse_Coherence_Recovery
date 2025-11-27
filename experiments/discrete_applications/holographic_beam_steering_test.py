"""
Holographic Beam Steering Test (LiDAR)
======================================

Tests PSON on discrete phase optimization for solid-state LiDAR.

PROBLEM:
- Optical Phased Array (OPA) for beam steering without mechanical parts
- Each emitter has discrete phase control
- Goal: Steer beam to target angle with minimal sidelobes

DISCRETE CONSTRAINT:
- Typical: 5-6 bit phase shifters (32-64 levels)
- Trade-off: more bits = better performance but slower/more power

PSON ADVANTAGE:
- Monotonic descent = safe for real-time scanning
- Handles element failures gracefully
- Non-local credit for multi-objective (main beam + sidelobe suppression)

REAL-WORLD APPLICATION:
- Autonomous vehicles (LiDAR)
- Free-space optical communication
- Laser machining

Usage:
    uv run python experiments/discrete_applications/holographic_beam_steering_test.py
"""

import numpy as np
import json
from typing import Tuple, List

# =============================================================================
# Optical Phased Array Physics
# =============================================================================

def compute_far_field_1d(
    positions: np.ndarray,    # Element positions (in wavelengths)
    phases: np.ndarray,       # Phase of each element (radians)
    theta: np.ndarray,        # Observation angles (radians)
    wavelength: float = 1.0,  # Wavelength (arbitrary units)
) -> np.ndarray:
    """
    Compute 1D far-field pattern from optical phased array.
    """
    k = 2 * np.pi / wavelength
    
    # Far-field = sum of phasors
    E = np.zeros(len(theta), dtype=complex)
    for pos, phi in zip(positions, phases):
        E += np.exp(1j * (k * pos * np.sin(theta) + phi))
    
    return np.abs(E) ** 2


def beam_steering_objective(
    positions: np.ndarray,
    phases: np.ndarray,
    target_theta: float,
    sidelobe_weight: float = 0.3,
) -> float:
    """
    Combined objective: maximize main beam, minimize sidelobes.
    
    Returns value between 0 and 1 (higher is better).
    """
    theta = np.linspace(-np.pi/2, np.pi/2, 181)
    pattern = compute_far_field_1d(positions, phases, theta)
    
    # Normalize
    n = len(positions)
    max_possible = n ** 2
    pattern_norm = pattern / max_possible
    
    # Main beam: power at target angle
    target_idx = np.argmin(np.abs(theta - target_theta))
    main_beam = pattern_norm[target_idx]
    
    # Sidelobes: peak outside main beam region (±5°)
    mask_width = 5  # degrees
    mask_start = max(0, target_idx - mask_width)
    mask_end = min(len(theta), target_idx + mask_width)
    
    sidelobe_pattern = pattern_norm.copy()
    sidelobe_pattern[mask_start:mask_end] = 0
    peak_sidelobe = np.max(sidelobe_pattern)
    
    # Combined objective
    objective = main_beam - sidelobe_weight * peak_sidelobe
    
    return float(max(0, objective))


def steering_efficiency(
    positions: np.ndarray,
    phases: np.ndarray,
    target_theta: float,
) -> float:
    """
    Beam steering efficiency: power at target / total radiated power.
    """
    theta = np.linspace(-np.pi/2, np.pi/2, 361)
    pattern = compute_far_field_1d(positions, phases, theta)
    
    target_idx = np.argmin(np.abs(theta - target_theta))
    
    # Main beam region (±2°)
    width = 4
    start = max(0, target_idx - width)
    end = min(len(theta), target_idx + width)
    
    main_power = np.sum(pattern[start:end])
    total_power = np.sum(pattern)
    
    return float(main_power / (total_power + 1e-10))


# =============================================================================
# PSON Beam Steering Optimizer
# =============================================================================

def pson_beam_steering(
    positions: np.ndarray,
    target_theta: float,
    n_levels: int = 32,  # 5-bit
    steps: int = 300,
    seed: int = 42,
    sidelobe_weight: float = 0.3,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    PSON optimization for beam steering with discrete phases.
    """
    rng = np.random.default_rng(seed)
    n = len(positions)
    
    phase_levels = np.array([i * 2 * np.pi / n_levels for i in range(n_levels)])
    
    # Initialize with ideal steering phases (quantized)
    k = 2 * np.pi
    ideal_phases = -k * positions * np.sin(target_theta)
    phases = np.array([phase_levels[int(round(p * n_levels / (2 * np.pi))) % n_levels] for p in ideal_phases])
    
    # Weights: position-based (edge elements matter more for sidelobes)
    edge_distance = np.abs(positions - np.mean(positions))
    weights = edge_distance / (edge_distance.sum() + 1e-8)
    weights = 0.5 * weights + 0.5 / n  # Mix with uniform
    
    def compute_objective(ph):
        return beam_steering_objective(positions, ph, target_theta, sidelobe_weight)
    
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
        
        n_changes = max(1, int(n * 0.15 * (0.3 + energy)))
        change_idx = rng.choice(n, min(n_changes, n), replace=False, p=change_probs)
        
        candidate = phases.copy()
        for idx in change_idx:
            current_level = int(round(phases[idx] * n_levels / (2 * np.pi))) % n_levels
            # Smaller perturbations for fine-tuning
            if rng.random() < 0.8:
                delta = rng.choice([-1, 1])
                new_level = (current_level + delta) % n_levels
            else:
                delta = rng.choice([-2, -1, 1, 2])
                new_level = (current_level + delta) % n_levels
            candidate[idx] = phase_levels[new_level]
        
        obj_new = compute_objective(candidate)
        if obj_new >= obj_cur:
            phases = candidate
    
    return best_phases, best_obj, obj_curve


def gradient_quantized(
    positions: np.ndarray,
    target_theta: float,
    n_levels: int = 32,
    steps: int = 300,
    seed: int = 42,
    sidelobe_weight: float = 0.3,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Gradient descent with post-quantization.
    Common approach: optimize continuous, then quantize.
    """
    rng = np.random.default_rng(seed)
    n = len(positions)
    
    phase_levels = np.array([i * 2 * np.pi / n_levels for i in range(n_levels)])
    
    # Start with ideal steering
    k = 2 * np.pi
    phases = -k * positions * np.sin(target_theta)
    
    def compute_objective(ph):
        return beam_steering_objective(positions, ph, target_theta, sidelobe_weight)
    
    def quantize(ph):
        idx = np.round(ph * n_levels / (2 * np.pi)).astype(int) % n_levels
        return phase_levels[idx]
    
    lr = 0.1
    obj_curve = []
    
    for step in range(steps):
        # Compute gradient numerically
        grad = np.zeros(n)
        eps = 0.01
        for i in range(n):
            phases_plus = phases.copy()
            phases_plus[i] += eps
            phases_minus = phases.copy()
            phases_minus[i] -= eps
            grad[i] = (compute_objective(phases_plus) - compute_objective(phases_minus)) / (2 * eps)
        
        phases = phases + lr * grad
        phases = phases % (2 * np.pi)
        
        # Quantize for evaluation
        quantized = quantize(phases)
        obj_curve.append(compute_objective(quantized))
    
    final_phases = quantize(phases)
    return final_phases, compute_objective(final_phases), obj_curve


def random_search_beam(
    positions: np.ndarray,
    target_theta: float,
    n_levels: int = 32,
    steps: int = 300,
    seed: int = 42,
    sidelobe_weight: float = 0.3,
) -> Tuple[np.ndarray, float, List[float]]:
    """Random search baseline."""
    rng = np.random.default_rng(seed)
    n = len(positions)
    
    phase_levels = np.array([i * 2 * np.pi / n_levels for i in range(n_levels)])
    
    def compute_objective(ph):
        return beam_steering_objective(positions, ph, target_theta, sidelobe_weight)
    
    best_phases = rng.choice(phase_levels, n)
    best_obj = compute_objective(best_phases)
    obj_curve = []
    
    for step in range(steps):
        phases = rng.choice(phase_levels, n)
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
    print("HOLOGRAPHIC BEAM STEERING (LiDAR) TEST")
    print("=" * 70)
    
    results = []
    
    configs = [
        {"name": "16-element, 5-bit, 0deg", "n_elements": 16, "n_levels": 32, "target_deg": 0},
        {"name": "16-element, 5-bit, 30deg", "n_elements": 16, "n_levels": 32, "target_deg": 30},
        {"name": "32-element, 5-bit, 30deg", "n_elements": 32, "n_levels": 32, "target_deg": 30},
        {"name": "32-element, 4-bit, 30deg", "n_elements": 32, "n_levels": 16, "target_deg": 30},
        {"name": "64-element, 6-bit, 45deg", "n_elements": 64, "n_levels": 64, "target_deg": 45},
    ]
    
    seeds = [42, 123, 456]
    
    for config in configs:
        print(f"\n{config['name']}")
        print("-" * 50)
        
        n_elements = config["n_elements"]
        n_levels = config["n_levels"]
        target_theta = np.deg2rad(config["target_deg"])
        
        # Half-wavelength spacing
        positions = np.arange(n_elements) * 0.5
        
        config_results = {"config": config["name"], "runs": []}
        
        for seed in seeds:
            # PSON
            _, pson_obj, _ = pson_beam_steering(
                positions, target_theta, n_levels=n_levels, steps=300, seed=seed
            )
            
            # Gradient + quantize
            _, gd_obj, _ = gradient_quantized(
                positions, target_theta, n_levels=n_levels, steps=300, seed=seed
            )
            
            # Random Search
            _, rs_obj, _ = random_search_beam(
                positions, target_theta, n_levels=n_levels, steps=300, seed=seed
            )
            
            winner = max([("PSON", pson_obj), ("GD+Q", gd_obj), ("RS", rs_obj)], key=lambda x: x[1])[0]
            
            print(f"  Seed {seed}: PSON={pson_obj:.4f}, GD+Q={gd_obj:.4f}, RS={rs_obj:.4f}, Winner: {winner}")
            
            config_results["runs"].append({
                "seed": seed,
                "pson": float(pson_obj),
                "gd_quantized": float(gd_obj),
                "rs": float(rs_obj),
                "winner": winner,
            })
        
        pson_wins = sum(1 for r in config_results["runs"] if r["winner"] == "PSON")
        config_results["summary"] = {
            "pson_mean": float(np.mean([r["pson"] for r in config_results["runs"]])),
            "gd_mean": float(np.mean([r["gd_quantized"] for r in config_results["runs"]])),
            "rs_mean": float(np.mean([r["rs"] for r in config_results["runs"]])),
            "pson_wins": pson_wins,
        }
        
        results.append(config_results)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_pson_wins = sum(r["summary"]["pson_wins"] for r in results)
    total_runs = sum(len(r["runs"]) for r in results)
    print(f"PSON wins: {total_pson_wins}/{total_runs}")
    
    for r in results:
        s = r["summary"]
        print(f"  {r['config']}: PSON={s['pson_mean']:.4f}, GD+Q={s['gd_mean']:.4f}")
    
    # Save
    with open("beam_steering_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved: beam_steering_results.json")


if __name__ == "__main__":
    main()

