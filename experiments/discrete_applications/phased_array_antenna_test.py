"""
Phased Array Antenna Test (5G/6G, Radar)
=========================================

Tests PSON on discrete phase optimization for phased array antennas.

PROBLEM:
- N antenna elements arranged in a line or grid
- Each element has a digital phase shifter with K discrete levels
- Goal: Maximize beam gain in target direction (or minimize sidelobes)

DISCRETE CONSTRAINT:
- Phase levels: 0, 2π/K, 4π/K, ..., 2π(K-1)/K
- Typical: 3-bit (8 levels), 4-bit (16 levels), 6-bit (64 levels)

PSON ADVANTAGE:
- Non-local credit assignment handles "which elements matter most"
- Weighted exploration prioritizes elements with irregular spacing/response
- Monotonic descent = safe for real-time reconfiguration

Usage:
    uv run python experiments/discrete_applications/phased_array_antenna_test.py
"""

import numpy as np
import json
from typing import Tuple, List

# =============================================================================
# Phased Array Physics
# =============================================================================

def array_factor(
    positions: np.ndarray,  # Element positions (in wavelengths)
    phases: np.ndarray,     # Phase of each element (radians)
    theta: np.ndarray,      # Observation angles (radians)
    amplitudes: np.ndarray = None,  # Optional amplitude weights
) -> np.ndarray:
    """
    Compute array factor for a phased array.
    
    AF(θ) = Σ_n a_n * exp(j * (k * d_n * sin(θ) + φ_n))
    
    Returns magnitude squared (power pattern).
    """
    if amplitudes is None:
        amplitudes = np.ones(len(positions))
    
    k = 2 * np.pi  # k*λ = 2π, positions already in wavelengths
    
    # Compute array factor
    AF = np.zeros(len(theta), dtype=complex)
    for n, (d_n, phi_n, a_n) in enumerate(zip(positions, phases, amplitudes)):
        AF += a_n * np.exp(1j * (k * d_n * np.sin(theta) + phi_n))
    
    return np.abs(AF) ** 2


def beam_gain(
    positions: np.ndarray,
    phases: np.ndarray,
    target_theta: float,  # Target steering angle (radians)
    amplitudes: np.ndarray = None,
) -> float:
    """Compute gain in target direction (normalized)."""
    theta = np.array([target_theta])
    AF = array_factor(positions, phases, theta, amplitudes)
    
    # Normalize by max possible (all in phase)
    n = len(positions)
    max_gain = n ** 2
    
    return float(AF[0] / max_gain)


def sidelobe_level(
    positions: np.ndarray,
    phases: np.ndarray,
    target_theta: float,
    amplitudes: np.ndarray = None,
    theta_resolution: int = 361,
) -> float:
    """
    Compute peak sidelobe level relative to main beam.
    Lower is better (returns negative dB).
    """
    theta = np.linspace(-np.pi/2, np.pi/2, theta_resolution)
    AF = array_factor(positions, phases, theta, amplitudes)
    
    # Find main beam (should be near target)
    main_idx = np.argmin(np.abs(theta - target_theta))
    main_power = AF[main_idx]
    
    # Mask out main beam region (±5°)
    mask_width = int(5 * theta_resolution / 180)
    AF_sidelobes = AF.copy()
    AF_sidelobes[max(0, main_idx - mask_width):min(len(AF), main_idx + mask_width)] = 0
    
    peak_sidelobe = np.max(AF_sidelobes)
    
    if main_power < 1e-10:
        return -100.0  # No main beam
    
    sll_db = 10 * np.log10(peak_sidelobe / main_power + 1e-10)
    return float(sll_db)


# =============================================================================
# PSON Discrete Phase Optimizer
# =============================================================================

def quantize_phase(phase: float, n_levels: int) -> float:
    """Quantize phase to nearest discrete level."""
    level_size = 2 * np.pi / n_levels
    level = round(phase / level_size) % n_levels
    return level * level_size


def pson_phased_array(
    positions: np.ndarray,
    target_theta: float,
    n_levels: int = 8,  # 3-bit phase shifter
    steps: int = 300,
    seed: int = 42,
    objective: str = "gain",  # "gain" or "sidelobe"
) -> Tuple[np.ndarray, float, List[float]]:
    """
    PSON optimization for phased array with discrete phases.
    """
    rng = np.random.default_rng(seed)
    n = len(positions)
    
    # Initialize with steering phases (quantized)
    k = 2 * np.pi
    ideal_phases = -k * positions * np.sin(target_theta)
    phases = np.array([quantize_phase(p, n_levels) for p in ideal_phases])
    
    # Precision from position irregularity
    pos_mean = np.mean(positions)
    pos_var = np.var(positions) + 1e-8
    irregularity = (positions - pos_mean)**2 / pos_var
    weights = irregularity / (irregularity.sum() + 1e-8)
    if weights.sum() < 1e-8:
        weights = np.ones(n) / n
    
    phase_levels = np.array([i * 2 * np.pi / n_levels for i in range(n_levels)])
    
    def compute_objective(ph):
        if objective == "gain":
            return beam_gain(positions, ph, target_theta)
        else:  # sidelobe
            return -sidelobe_level(positions, ph, target_theta)  # Negate: lower SLL is better
    
    best_phases = phases.copy()
    best_obj = compute_objective(phases)
    obj_curve = []
    
    for step in range(steps):
        obj_cur = compute_objective(phases)
        obj_curve.append(obj_cur if objective == "gain" else -obj_cur)
        
        if obj_cur > best_obj:
            best_obj = obj_cur
            best_phases = phases.copy()
        
        energy = 1.0 - obj_cur
        
        # PSON: weighted exploration
        change_probs = weights * (0.3 + 0.7 * energy)
        change_probs = change_probs / change_probs.sum()
        
        n_changes = max(1, int(n * 0.2 * rng.random()))
        change_idx = rng.choice(n, n_changes, replace=False, p=change_probs)
        
        candidate = phases.copy()
        for idx in change_idx:
            # Try adjacent phase levels (more likely) or random jump
            current_level = int(round(phases[idx] * n_levels / (2 * np.pi))) % n_levels
            if rng.random() < 0.7:
                # Adjacent level
                delta = rng.choice([-1, 1])
                new_level = (current_level + delta) % n_levels
            else:
                # Random level
                new_level = rng.integers(0, n_levels)
            candidate[idx] = phase_levels[new_level]
        
        obj_new = compute_objective(candidate)
        if obj_new >= obj_cur:
            phases = candidate
    
    return best_phases, best_obj, obj_curve


def random_search_phased_array(
    positions: np.ndarray,
    target_theta: float,
    n_levels: int = 8,
    steps: int = 300,
    seed: int = 42,
    objective: str = "gain",
) -> Tuple[np.ndarray, float, List[float]]:
    """Random search baseline."""
    rng = np.random.default_rng(seed)
    n = len(positions)
    
    phase_levels = np.array([i * 2 * np.pi / n_levels for i in range(n_levels)])
    
    def compute_objective(ph):
        if objective == "gain":
            return beam_gain(positions, ph, target_theta)
        else:
            return -sidelobe_level(positions, ph, target_theta)
    
    best_phases = rng.choice(phase_levels, n)
    best_obj = compute_objective(best_phases)
    obj_curve = []
    
    for step in range(steps):
        phases = rng.choice(phase_levels, n)
        obj = compute_objective(phases)
        obj_curve.append(best_obj if objective == "gain" else -best_obj)
        
        if obj > best_obj:
            best_obj = obj
            best_phases = phases.copy()
    
    return best_phases, best_obj, obj_curve


# =============================================================================
# Main Test
# =============================================================================

def main():
    print("=" * 70)
    print("PHASED ARRAY ANTENNA TEST: PSON vs Random Search")
    print("=" * 70)
    
    results = []
    
    # Test configurations
    configs = [
        {"name": "8-element ULA, 3-bit", "n_elements": 8, "n_levels": 8, "spacing": "uniform"},
        {"name": "16-element ULA, 4-bit", "n_elements": 16, "n_levels": 16, "spacing": "uniform"},
        {"name": "8-element sparse, 3-bit", "n_elements": 8, "n_levels": 8, "spacing": "sparse"},
        {"name": "16-element sparse, 4-bit", "n_elements": 16, "n_levels": 16, "spacing": "sparse"},
    ]
    
    target_theta = np.deg2rad(30)  # Steer to 30°
    seeds = [42, 123, 456]
    
    for config in configs:
        print(f"\n{config['name']}")
        print("-" * 50)
        
        n = config["n_elements"]
        n_levels = config["n_levels"]
        
        # Create element positions
        if config["spacing"] == "uniform":
            positions = np.arange(n) * 0.5  # Half-wavelength spacing
        else:
            # Sparse/irregular spacing
            rng = np.random.default_rng(0)
            positions = np.sort(rng.uniform(0, n * 0.5, n))
        
        config_results = {"config": config["name"], "runs": []}
        
        for seed in seeds:
            # PSON
            _, pson_gain, _ = pson_phased_array(
                positions, target_theta, n_levels=n_levels, steps=300, seed=seed, objective="gain"
            )
            
            # Random Search
            _, rs_gain, _ = random_search_phased_array(
                positions, target_theta, n_levels=n_levels, steps=300, seed=seed, objective="gain"
            )
            
            winner = "PSON" if pson_gain > rs_gain else "RS"
            print(f"  Seed {seed}: PSON={pson_gain:.4f}, RS={rs_gain:.4f}, Winner: {winner}")
            
            config_results["runs"].append({
                "seed": seed,
                "pson_gain": float(pson_gain),
                "rs_gain": float(rs_gain),
                "winner": winner,
            })
        
        pson_wins = sum(1 for r in config_results["runs"] if r["winner"] == "PSON")
        pson_mean = np.mean([r["pson_gain"] for r in config_results["runs"]])
        rs_mean = np.mean([r["rs_gain"] for r in config_results["runs"]])
        
        config_results["summary"] = {
            "pson_mean": float(pson_mean),
            "rs_mean": float(rs_mean),
            "pson_wins": pson_wins,
            "total": len(seeds),
        }
        
        print(f"  Summary: PSON mean={pson_mean:.4f}, RS mean={rs_mean:.4f}, PSON wins {pson_wins}/{len(seeds)}")
        results.append(config_results)
    
    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    
    total_pson_wins = sum(r["summary"]["pson_wins"] for r in results)
    total_runs = sum(r["summary"]["total"] for r in results)
    print(f"PSON wins: {total_pson_wins}/{total_runs} ({100*total_pson_wins/total_runs:.1f}%)")
    
    # Save results
    with open("phased_array_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved: phased_array_results.json")


if __name__ == "__main__":
    main()

