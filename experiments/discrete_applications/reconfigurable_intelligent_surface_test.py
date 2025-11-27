"""
Reconfigurable Intelligent Surface (RIS) Test
==============================================

Tests PSON on discrete phase optimization for RIS in wireless communications.

PROBLEM:
- Large array of passive reflecting elements (100s to 1000s)
- Each element has 1-2 bit phase control (2-4 levels)
- Goal: Configure phases to maximize signal at receiver

DISCRETE CONSTRAINT:
- 1-bit: phases {0, π}
- 2-bit: phases {0, π/2, π, 3π/2}
- Very coarse quantization!

PSON ADVANTAGE:
- Handles massive discrete search space (2^N or 4^N configurations)
- Only needs scalar RSSI feedback (received signal strength)
- Non-local credit guides which elements to flip

REAL-WORLD RELEVANCE:
- Hot research area in 6G communications
- RIS enables smart radio environments
- Key challenge: optimize with only received power feedback

Usage:
    uv run python experiments/discrete_applications/reconfigurable_intelligent_surface_test.py
"""

import numpy as np
import json
from typing import Tuple, List

# =============================================================================
# RIS Channel Model
# =============================================================================

def generate_ris_channels(
    n_elements: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate RIS channel matrices.
    
    Returns:
        h_tx: Channel from transmitter to RIS elements (n_elements,)
        h_rx: Channel from RIS elements to receiver (n_elements,)
    """
    rng = np.random.default_rng(seed)
    
    # Rayleigh fading channels (complex)
    h_tx = (rng.standard_normal(n_elements) + 1j * rng.standard_normal(n_elements)) / np.sqrt(2)
    h_rx = (rng.standard_normal(n_elements) + 1j * rng.standard_normal(n_elements)) / np.sqrt(2)
    
    # Add path loss (distance-dependent)
    # Assume RIS elements have slightly different distances
    distances = 1 + 0.1 * rng.standard_normal(n_elements)
    path_loss = 1 / (distances ** 2)
    
    h_tx = h_tx * np.sqrt(path_loss)
    h_rx = h_rx * np.sqrt(path_loss)
    
    return h_tx, h_rx


def compute_received_power(
    h_tx: np.ndarray,
    h_rx: np.ndarray,
    phases: np.ndarray,
    transmit_power: float = 1.0,
) -> float:
    """
    Compute received signal power through RIS.
    
    P_rx = |h_rx^H * Φ * h_tx|^2 * P_tx
    
    where Φ = diag(exp(jφ)) is the RIS phase shift matrix.
    """
    # RIS reflection coefficients
    phi = np.exp(1j * phases)
    
    # Cascaded channel
    h_cascaded = h_rx.conj() * phi * h_tx
    
    # Received power
    channel_gain = np.abs(np.sum(h_cascaded)) ** 2
    
    return float(channel_gain * transmit_power)


def compute_snr_db(
    h_tx: np.ndarray,
    h_rx: np.ndarray,
    phases: np.ndarray,
    noise_power: float = 1e-3,
) -> float:
    """Compute SNR in dB."""
    p_rx = compute_received_power(h_tx, h_rx, phases)
    snr = p_rx / noise_power
    return float(10 * np.log10(snr + 1e-10))


# =============================================================================
# PSON RIS Optimizer
# =============================================================================

def pson_ris(
    h_tx: np.ndarray,
    h_rx: np.ndarray,
    n_bits: int = 2,  # 1-bit or 2-bit
    steps: int = 500,
    seed: int = 42,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    PSON optimization for RIS with discrete phases.
    """
    rng = np.random.default_rng(seed)
    n = len(h_tx)
    
    n_levels = 2 ** n_bits
    phase_levels = np.array([i * 2 * np.pi / n_levels for i in range(n_levels)])
    
    # Initialize randomly
    phases = rng.choice(phase_levels, n)
    
    # Weights: based on channel strength (stronger channels = more important)
    channel_strength = np.abs(h_tx * h_rx)
    weights = channel_strength / (channel_strength.sum() + 1e-8)
    
    def compute_objective(ph):
        return compute_received_power(h_tx, h_rx, ph)
    
    # Normalize objective (estimate max power)
    max_power_estimate = np.sum(np.abs(h_tx) * np.abs(h_rx)) ** 2
    
    best_phases = phases.copy()
    best_obj = compute_objective(phases)
    obj_curve = []
    
    for step in range(steps):
        obj_cur = compute_objective(phases)
        obj_curve.append(obj_cur)
        
        if obj_cur > best_obj:
            best_obj = obj_cur
            best_phases = phases.copy()
        
        # Normalized energy (0 to 1)
        energy = 1.0 - min(obj_cur / max_power_estimate, 1.0)
        
        # PSON: weighted exploration
        change_probs = weights * (0.2 + 0.8 * energy)
        change_probs = change_probs / change_probs.sum()
        
        n_changes = max(1, int(n * 0.1 * (0.5 + energy)))
        change_idx = rng.choice(n, min(n_changes, n), replace=False, p=change_probs)
        
        candidate = phases.copy()
        for idx in change_idx:
            current_level = int(round(phases[idx] * n_levels / (2 * np.pi))) % n_levels
            if rng.random() < 0.7:
                delta = rng.choice([-1, 1])
                new_level = (current_level + delta) % n_levels
            else:
                new_level = rng.integers(0, n_levels)
            candidate[idx] = phase_levels[new_level]
        
        obj_new = compute_objective(candidate)
        if obj_new >= obj_cur:
            phases = candidate
    
    return best_phases, best_obj, obj_curve


def exhaustive_1bit(
    h_tx: np.ndarray,
    h_rx: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Exhaustive search for 1-bit RIS (only feasible for small N).
    Optimal solution for comparison.
    """
    n = len(h_tx)
    
    if n > 20:
        raise ValueError("Exhaustive search only for N <= 20")
    
    best_phases = None
    best_power = -np.inf
    
    for i in range(2 ** n):
        # Convert integer to binary phase pattern
        phases = np.array([(i >> j) & 1 for j in range(n)]) * np.pi
        power = compute_received_power(h_tx, h_rx, phases)
        
        if power > best_power:
            best_power = power
            best_phases = phases.copy()
    
    return best_phases, best_power


def random_search_ris(
    h_tx: np.ndarray,
    h_rx: np.ndarray,
    n_bits: int = 2,
    steps: int = 500,
    seed: int = 42,
) -> Tuple[np.ndarray, float, List[float]]:
    """Random search baseline."""
    rng = np.random.default_rng(seed)
    n = len(h_tx)
    
    n_levels = 2 ** n_bits
    phase_levels = np.array([i * 2 * np.pi / n_levels for i in range(n_levels)])
    
    def compute_objective(ph):
        return compute_received_power(h_tx, h_rx, ph)
    
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


def greedy_ris(
    h_tx: np.ndarray,
    h_rx: np.ndarray,
    n_bits: int = 2,
    seed: int = 42,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Greedy element-by-element optimization.
    Common practical approach.
    """
    rng = np.random.default_rng(seed)
    n = len(h_tx)
    
    n_levels = 2 ** n_bits
    phase_levels = np.array([i * 2 * np.pi / n_levels for i in range(n_levels)])
    
    # Initialize randomly
    phases = rng.choice(phase_levels, n)
    obj_curve = []
    
    def compute_objective(ph):
        return compute_received_power(h_tx, h_rx, ph)
    
    # Multiple passes
    for _ in range(5):
        for i in range(n):
            best_level = phases[i]
            best_obj = compute_objective(phases)
            
            for level in phase_levels:
                candidate = phases.copy()
                candidate[i] = level
                obj = compute_objective(candidate)
                
                if obj > best_obj:
                    best_obj = obj
                    best_level = level
            
            phases[i] = best_level
            obj_curve.append(compute_objective(phases))
    
    return phases, compute_objective(phases), obj_curve


# =============================================================================
# Main Test
# =============================================================================

def main():
    print("=" * 70)
    print("RECONFIGURABLE INTELLIGENT SURFACE (RIS) TEST")
    print("=" * 70)
    
    results = []
    
    configs = [
        {"name": "16-element, 1-bit", "n_elements": 16, "n_bits": 1},
        {"name": "16-element, 2-bit", "n_elements": 16, "n_bits": 2},
        {"name": "64-element, 1-bit", "n_elements": 64, "n_bits": 1},
        {"name": "64-element, 2-bit", "n_elements": 64, "n_bits": 2},
        {"name": "256-element, 2-bit", "n_elements": 256, "n_bits": 2},
    ]
    
    seeds = [42, 123, 456]
    
    for config in configs:
        print(f"\n{config['name']}")
        print("-" * 50)
        
        n_elements = config["n_elements"]
        n_bits = config["n_bits"]
        
        config_results = {"config": config["name"], "runs": []}
        
        for seed in seeds:
            # Generate channels
            h_tx, h_rx = generate_ris_channels(n_elements, seed=seed)
            
            # Optimal (only for small 1-bit)
            if n_elements <= 16 and n_bits == 1:
                _, opt_power = exhaustive_1bit(h_tx, h_rx)
            else:
                opt_power = None
            
            # PSON
            _, pson_power, _ = pson_ris(h_tx, h_rx, n_bits=n_bits, steps=500, seed=seed)
            
            # Greedy
            _, greedy_power, _ = greedy_ris(h_tx, h_rx, n_bits=n_bits, seed=seed)
            
            # Random Search
            _, rs_power, _ = random_search_ris(h_tx, h_rx, n_bits=n_bits, steps=500, seed=seed)
            
            winner = max([("PSON", pson_power), ("Greedy", greedy_power), ("RS", rs_power)], key=lambda x: x[1])[0]
            
            if opt_power:
                pson_gap = 100 * (opt_power - pson_power) / opt_power
                print(f"  Seed {seed}: PSON={pson_power:.4f} (gap={pson_gap:.1f}%), Greedy={greedy_power:.4f}, RS={rs_power:.4f}, Winner: {winner}")
            else:
                print(f"  Seed {seed}: PSON={pson_power:.4f}, Greedy={greedy_power:.4f}, RS={rs_power:.4f}, Winner: {winner}")
            
            config_results["runs"].append({
                "seed": seed,
                "pson": float(pson_power),
                "greedy": float(greedy_power),
                "rs": float(rs_power),
                "optimal": float(opt_power) if opt_power else None,
                "winner": winner,
            })
        
        pson_wins = sum(1 for r in config_results["runs"] if r["winner"] == "PSON")
        config_results["summary"] = {
            "pson_mean": float(np.mean([r["pson"] for r in config_results["runs"]])),
            "greedy_mean": float(np.mean([r["greedy"] for r in config_results["runs"]])),
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
        print(f"  {r['config']}: PSON={s['pson_mean']:.4f}, Greedy={s['greedy_mean']:.4f}, RS={s['rs_mean']:.4f}")
    
    # Save
    with open("ris_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved: ris_results.json")


if __name__ == "__main__":
    main()

