"""
Acoustic Beamforming Test (Ultrasound/Sonar)
============================================

Tests PSON on discrete delay/phase optimization for acoustic arrays.

PROBLEM:
- Array of transducers (speakers or ultrasonic emitters)
- Each element has discrete time delay (quantized)
- Goal: Focus acoustic beam at target location

DISCRETE CONSTRAINT:
- Time delays quantized to ADC/DAC resolution
- Typical: 8-12 bit timing (256-4096 levels)
- Phase = 2π * f * delay

PSON ADVANTAGE:
- Handles element response variations
- Non-local credit for multi-frequency operation
- Safe for real-time focusing (medical ultrasound)

APPLICATIONS:
- Medical ultrasound imaging
- HIFU (High-Intensity Focused Ultrasound) therapy
- Sonar systems
- Parametric speaker arrays

Usage:
    uv run python experiments/discrete_applications/acoustic_beamforming_test.py
"""

import numpy as np
import json
from typing import Tuple, List

# =============================================================================
# Acoustic Physics
# =============================================================================

def compute_acoustic_field(
    element_positions: np.ndarray,  # (N, 2) array of x, y positions
    delays: np.ndarray,              # Time delays for each element
    focal_point: np.ndarray,         # Target focus point (x, y)
    frequency: float = 1e6,          # Frequency in Hz (1 MHz for ultrasound)
    speed_of_sound: float = 1500,    # m/s in tissue
) -> float:
    """
    Compute acoustic pressure at focal point.
    
    Uses simple delay-and-sum model.
    """
    wavelength = speed_of_sound / frequency
    k = 2 * np.pi / wavelength
    
    # Distance from each element to focal point
    distances = np.sqrt(np.sum((element_positions - focal_point) ** 2, axis=1))
    
    # Propagation time
    prop_times = distances / speed_of_sound
    
    # Total phase at focal point (propagation + applied delay)
    total_phase = k * distances - 2 * np.pi * frequency * delays
    
    # Sum of phasors (complex pressure)
    pressure = np.sum(np.exp(1j * total_phase))
    
    return np.abs(pressure) ** 2


def compute_focusing_efficiency(
    element_positions: np.ndarray,
    delays: np.ndarray,
    focal_point: np.ndarray,
    frequency: float = 1e6,
    speed_of_sound: float = 1500,
    grid_size: int = 21,
    region_half_width: float = 0.01,  # 1 cm half-width
) -> Tuple[float, float]:
    """
    Compute focusing efficiency and sidelobe level.
    
    Returns:
        efficiency: Power at focal point / max possible
        peak_sidelobe_ratio: Peak sidelobe / main peak (lower is better)
    """
    n = len(element_positions)
    
    # Grid around focal point
    x = np.linspace(focal_point[0] - region_half_width, focal_point[0] + region_half_width, grid_size)
    y = np.linspace(focal_point[1] - region_half_width, focal_point[1] + region_half_width, grid_size)
    
    field = np.zeros((grid_size, grid_size))
    
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            point = np.array([xi, yj])
            field[i, j] = compute_acoustic_field(element_positions, delays, point, frequency, speed_of_sound)
    
    # Main peak (should be at center)
    center = grid_size // 2
    main_peak = field[center, center]
    
    # Max possible (perfect focusing)
    max_possible = n ** 2
    efficiency = main_peak / max_possible
    
    # Sidelobe: max outside center region
    mask = np.ones_like(field, dtype=bool)
    mask[center-2:center+3, center-2:center+3] = False
    peak_sidelobe = np.max(field[mask]) if np.any(mask) else 0
    
    sidelobe_ratio = peak_sidelobe / (main_peak + 1e-10)
    
    return float(efficiency), float(sidelobe_ratio)


# =============================================================================
# PSON Acoustic Optimizer
# =============================================================================

def pson_acoustic_focus(
    element_positions: np.ndarray,
    focal_point: np.ndarray,
    n_levels: int = 256,  # 8-bit delay quantization
    frequency: float = 1e6,
    speed_of_sound: float = 1500,
    steps: int = 300,
    seed: int = 42,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    PSON optimization for acoustic focusing with discrete delays.
    """
    rng = np.random.default_rng(seed)
    n = len(element_positions)
    
    wavelength = speed_of_sound / frequency
    period = 1.0 / frequency
    
    # Delay levels (one period range)
    delay_levels = np.array([i * period / n_levels for i in range(n_levels)])
    
    # Initialize with geometric focusing delays (quantized)
    distances = np.sqrt(np.sum((element_positions - focal_point) ** 2, axis=1))
    max_dist = np.max(distances)
    ideal_delays = (max_dist - distances) / speed_of_sound
    # Wrap to one period and quantize
    ideal_delays = ideal_delays % period
    delays = np.array([delay_levels[int(round(d * n_levels / period)) % n_levels] for d in ideal_delays])
    
    # Weights: distance-based (farther elements need more precision)
    dist_weights = distances / (distances.sum() + 1e-8)
    weights = 0.7 * dist_weights + 0.3 / n
    
    def compute_objective(dl):
        return compute_acoustic_field(element_positions, dl, focal_point, frequency, speed_of_sound)
    
    max_possible = n ** 2
    
    best_delays = delays.copy()
    best_obj = compute_objective(delays)
    obj_curve = []
    
    for step in range(steps):
        obj_cur = compute_objective(delays)
        obj_curve.append(obj_cur / max_possible)
        
        if obj_cur > best_obj:
            best_obj = obj_cur
            best_delays = delays.copy()
        
        energy = 1.0 - obj_cur / max_possible
        
        # PSON: weighted exploration
        change_probs = weights * (0.2 + 0.8 * energy)
        change_probs = change_probs / change_probs.sum()
        
        n_changes = max(1, int(n * 0.15 * (0.3 + energy)))
        change_idx = rng.choice(n, min(n_changes, n), replace=False, p=change_probs)
        
        candidate = delays.copy()
        for idx in change_idx:
            current_level = int(round(delays[idx] * n_levels / period)) % n_levels
            if rng.random() < 0.7:
                delta = rng.choice([-1, 1])
            else:
                delta = rng.choice([-3, -2, -1, 1, 2, 3])
            new_level = (current_level + delta) % n_levels
            candidate[idx] = delay_levels[new_level]
        
        obj_new = compute_objective(candidate)
        if obj_new >= obj_cur:
            delays = candidate
    
    return best_delays, best_obj / max_possible, obj_curve


def geometric_focus_quantized(
    element_positions: np.ndarray,
    focal_point: np.ndarray,
    n_levels: int = 256,
    frequency: float = 1e6,
    speed_of_sound: float = 1500,
) -> Tuple[np.ndarray, float]:
    """
    Simple geometric focusing with quantization.
    Baseline: compute ideal delays and quantize.
    """
    n = len(element_positions)
    period = 1.0 / frequency
    
    delay_levels = np.array([i * period / n_levels for i in range(n_levels)])
    
    distances = np.sqrt(np.sum((element_positions - focal_point) ** 2, axis=1))
    max_dist = np.max(distances)
    ideal_delays = (max_dist - distances) / speed_of_sound
    ideal_delays = ideal_delays % period
    
    delays = np.array([delay_levels[int(round(d * n_levels / period)) % n_levels] for d in ideal_delays])
    
    obj = compute_acoustic_field(element_positions, delays, focal_point, frequency, speed_of_sound)
    
    return delays, obj / (n ** 2)


def random_search_acoustic(
    element_positions: np.ndarray,
    focal_point: np.ndarray,
    n_levels: int = 256,
    frequency: float = 1e6,
    speed_of_sound: float = 1500,
    steps: int = 300,
    seed: int = 42,
) -> Tuple[np.ndarray, float, List[float]]:
    """Random search baseline."""
    rng = np.random.default_rng(seed)
    n = len(element_positions)
    
    period = 1.0 / frequency
    delay_levels = np.array([i * period / n_levels for i in range(n_levels)])
    
    def compute_objective(dl):
        return compute_acoustic_field(element_positions, dl, focal_point, frequency, speed_of_sound)
    
    max_possible = n ** 2
    
    best_delays = rng.choice(delay_levels, n)
    best_obj = compute_objective(best_delays)
    obj_curve = []
    
    for step in range(steps):
        delays = rng.choice(delay_levels, n)
        obj = compute_objective(delays)
        obj_curve.append(best_obj / max_possible)
        
        if obj > best_obj:
            best_obj = obj
            best_delays = delays.copy()
    
    return best_delays, best_obj / max_possible, obj_curve


# =============================================================================
# Array Geometries
# =============================================================================

def create_linear_array(n_elements: int, pitch: float = 0.0003) -> np.ndarray:
    """Linear array with given pitch (default 0.3mm for ultrasound)."""
    x = np.arange(n_elements) * pitch - (n_elements - 1) * pitch / 2
    y = np.zeros(n_elements)
    return np.column_stack([x, y])


def create_ring_array(n_elements: int, radius: float = 0.01) -> np.ndarray:
    """Circular ring array (common for HIFU)."""
    angles = np.linspace(0, 2 * np.pi, n_elements, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack([x, y])


def create_sparse_array(n_elements: int, aperture: float = 0.01, seed: int = 0) -> np.ndarray:
    """Sparse random array (irregular spacing)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-aperture/2, aperture/2, n_elements)
    y = rng.uniform(-aperture/2, aperture/2, n_elements)
    return np.column_stack([x, y])


# =============================================================================
# Main Test
# =============================================================================

def main():
    print("=" * 70)
    print("ACOUSTIC BEAMFORMING TEST (Ultrasound/Sonar)")
    print("=" * 70)
    
    results = []
    
    configs = [
        {"name": "16-linear, 8-bit", "geometry": "linear", "n_elements": 16, "n_levels": 256},
        {"name": "32-linear, 8-bit", "geometry": "linear", "n_elements": 32, "n_levels": 256},
        {"name": "16-ring, 8-bit", "geometry": "ring", "n_elements": 16, "n_levels": 256},
        {"name": "16-sparse, 8-bit", "geometry": "sparse", "n_elements": 16, "n_levels": 256},
        {"name": "32-sparse, 6-bit", "geometry": "sparse", "n_elements": 32, "n_levels": 64},
    ]
    
    focal_point = np.array([0.0, 0.03])  # 3 cm depth
    seeds = [42, 123, 456]
    
    for config in configs:
        print(f"\n{config['name']}")
        print("-" * 50)
        
        n_elements = config["n_elements"]
        n_levels = config["n_levels"]
        
        # Create array geometry
        if config["geometry"] == "linear":
            positions = create_linear_array(n_elements)
        elif config["geometry"] == "ring":
            positions = create_ring_array(n_elements)
        else:
            positions = create_sparse_array(n_elements)
        
        config_results = {"config": config["name"], "runs": []}
        
        for seed in seeds:
            # Geometric focusing (baseline)
            _, geo_obj = geometric_focus_quantized(positions, focal_point, n_levels=n_levels)
            
            # PSON
            _, pson_obj, _ = pson_acoustic_focus(
                positions, focal_point, n_levels=n_levels, steps=300, seed=seed
            )
            
            # Random Search
            _, rs_obj, _ = random_search_acoustic(
                positions, focal_point, n_levels=n_levels, steps=300, seed=seed
            )
            
            winner = max([("PSON", pson_obj), ("Geo", geo_obj), ("RS", rs_obj)], key=lambda x: x[1])[0]
            
            improvement = 100 * (pson_obj - geo_obj) / (geo_obj + 1e-10)
            
            print(f"  Seed {seed}: PSON={pson_obj:.4f} (+{improvement:.1f}%), Geo={geo_obj:.4f}, RS={rs_obj:.4f}, Winner: {winner}")
            
            config_results["runs"].append({
                "seed": seed,
                "pson": float(pson_obj),
                "geometric": float(geo_obj),
                "rs": float(rs_obj),
                "winner": winner,
                "pson_improvement_pct": float(improvement),
            })
        
        pson_wins = sum(1 for r in config_results["runs"] if r["winner"] == "PSON")
        config_results["summary"] = {
            "pson_mean": float(np.mean([r["pson"] for r in config_results["runs"]])),
            "geometric_mean": float(np.mean([r["geometric"] for r in config_results["runs"]])),
            "rs_mean": float(np.mean([r["rs"] for r in config_results["runs"]])),
            "pson_wins": pson_wins,
            "avg_improvement_pct": float(np.mean([r["pson_improvement_pct"] for r in config_results["runs"]])),
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
        print(f"  {r['config']}: PSON={s['pson_mean']:.4f} (+{s['avg_improvement_pct']:.1f}%), Geo={s['geometric_mean']:.4f}")
    
    # Save
    with open("acoustic_beamforming_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved: acoustic_beamforming_results.json")


if __name__ == "__main__":
    main()

