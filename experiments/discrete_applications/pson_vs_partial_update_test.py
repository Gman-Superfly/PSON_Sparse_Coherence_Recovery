"""
PSON vs Partial Update (PU) Adaptive Algorithms
================================================

Head-to-head comparison of PSON against state-of-the-art PU-LMS variants
for phased array beamforming.

PU-LMS VARIANTS:
1. Sequential-LMS: Cycles through elements in order
2. M-max LMS: Selects M elements with largest gradient magnitudes
3. Periodic-LMS: Updates every K-th element in repeating pattern
4. Stochastic-LMS: Randomly selects M elements each iteration

COMPARISON METRICS:
- Convergence speed (iterations to reach target MSE)
- Final beam gain at target angle
- Null depth at interferer angles
- Steady-state MSE
- Computational complexity (updates per iteration)

SCENARIO:
- Desired signal at 0° (or specified target angle)
- Interferers at ±30° (for null steering test)
- SNR: 10-20 dB
- Discrete phase levels (3-5 bit quantization)

Usage:
    uv run python experiments/discrete_applications/pson_vs_partial_update_test.py
"""

import numpy as np
import json
from typing import Tuple, List, Dict
from dataclasses import dataclass

# =============================================================================
# Phased Array Model
# =============================================================================

@dataclass
class ArrayConfig:
    """Phased array configuration."""
    n_elements: int = 16
    n_phase_levels: int = 32  # 5-bit
    element_spacing: float = 0.5  # wavelengths
    target_angle: float = 0.0  # degrees
    interferer_angles: List[float] = None  # degrees
    snr_db: float = 15.0
    
    def __post_init__(self):
        if self.interferer_angles is None:
            self.interferer_angles = [-30.0, 30.0]


def steering_vector(n_elements: int, spacing: float, angle_deg: float) -> np.ndarray:
    """
    Compute steering vector for a ULA.
    
    a(θ) = [1, e^(j*k*d*sin(θ)), e^(j*2*k*d*sin(θ)), ...]
    """
    angle_rad = np.deg2rad(angle_deg)
    k = 2 * np.pi  # k*λ = 2π
    n = np.arange(n_elements)
    return np.exp(1j * k * spacing * n * np.sin(angle_rad))


def array_response(weights: np.ndarray, config: ArrayConfig, angles: np.ndarray) -> np.ndarray:
    """Compute array response (beam pattern) at given angles."""
    response = np.zeros(len(angles), dtype=complex)
    for i, angle in enumerate(angles):
        sv = steering_vector(config.n_elements, config.element_spacing, angle)
        response[i] = np.dot(weights.conj(), sv)
    return np.abs(response) ** 2


def generate_received_signal(
    config: ArrayConfig,
    n_samples: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate received signal at array elements.
    
    Returns:
        X: (n_samples, n_elements) received signal matrix
        d: (n_samples,) desired signal
    """
    rng = np.random.default_rng(seed)
    n = config.n_elements
    
    # Desired signal
    s_desired = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) / np.sqrt(2)
    a_desired = steering_vector(n, config.element_spacing, config.target_angle)
    
    # Interferer signals
    interferer_power = 10 ** (-config.snr_db / 10)  # Interferers at lower power
    s_interferers = []
    a_interferers = []
    for angle in config.interferer_angles:
        s_int = np.sqrt(interferer_power) * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) / np.sqrt(2)
        a_int = steering_vector(n, config.element_spacing, angle)
        s_interferers.append(s_int)
        a_interferers.append(a_int)
    
    # Noise
    noise_power = 10 ** (-config.snr_db / 10)
    noise = np.sqrt(noise_power) * (rng.standard_normal((n_samples, n)) + 1j * rng.standard_normal((n_samples, n))) / np.sqrt(2)
    
    # Received signal: X = desired + interferers + noise
    X = np.outer(s_desired, a_desired)
    for s_int, a_int in zip(s_interferers, a_interferers):
        X += np.outer(s_int, a_int)
    X += noise
    
    return X, s_desired


def quantize_phases(phases: np.ndarray, n_levels: int) -> np.ndarray:
    """Quantize phases to discrete levels."""
    level_size = 2 * np.pi / n_levels
    levels = np.round(phases / level_size) * level_size
    return levels % (2 * np.pi)


def compute_mse(weights: np.ndarray, X: np.ndarray, d: np.ndarray) -> float:
    """Compute mean square error."""
    y = X @ weights
    e = d - y
    return float(np.mean(np.abs(e) ** 2))


def compute_beam_metrics(weights: np.ndarray, config: ArrayConfig) -> Dict[str, float]:
    """Compute beam quality metrics."""
    angles = np.linspace(-90, 90, 361)
    pattern = array_response(weights, config, angles)
    pattern_db = 10 * np.log10(pattern + 1e-10)
    
    # Main lobe gain (at target)
    target_idx = np.argmin(np.abs(angles - config.target_angle))
    main_gain = pattern_db[target_idx]
    
    # Null depths (at interferers)
    null_depths = []
    for int_angle in config.interferer_angles:
        int_idx = np.argmin(np.abs(angles - int_angle))
        null_depths.append(pattern_db[int_idx])
    
    # Sidelobe level (peak outside main beam ±10°)
    main_mask = np.abs(angles - config.target_angle) < 10
    sidelobe_pattern = pattern_db.copy()
    sidelobe_pattern[main_mask] = -100
    peak_sidelobe = np.max(sidelobe_pattern)
    
    return {
        "main_gain_db": float(main_gain),
        "null_depth_db": float(np.mean(null_depths)),
        "sidelobe_db": float(peak_sidelobe),
    }


# =============================================================================
# PU-LMS Variants
# =============================================================================

def full_lms(
    X: np.ndarray,
    d: np.ndarray,
    config: ArrayConfig,
    mu: float = 0.01,
    n_iterations: int = 300,
) -> Tuple[np.ndarray, List[float], int]:
    """
    Full LMS algorithm (baseline).
    Updates ALL elements every iteration.
    """
    n = config.n_elements
    n_levels = config.n_phase_levels
    n_samples = len(d)
    
    # Initialize weights
    phases = np.zeros(n)
    weights = np.exp(1j * phases)
    
    mse_curve = []
    updates_per_iter = n
    
    for iteration in range(n_iterations):
        # Cycle through samples
        sample_idx = iteration % n_samples
        x = X[sample_idx]
        
        # Output and error
        y = np.dot(weights.conj(), x)
        e = d[sample_idx] - y
        
        # Full update (all elements)
        grad = e.conj() * x
        phases = phases + mu * np.real(grad * np.exp(-1j * phases))
        
        # Quantize
        phases = quantize_phases(phases, n_levels)
        weights = np.exp(1j * phases)
        
        mse = compute_mse(weights, X, d)
        mse_curve.append(mse)
    
    return weights, mse_curve, updates_per_iter


def sequential_lms(
    X: np.ndarray,
    d: np.ndarray,
    config: ArrayConfig,
    mu: float = 0.01,
    n_iterations: int = 300,
) -> Tuple[np.ndarray, List[float], int]:
    """
    Sequential PU-LMS: Updates one element per iteration in cyclic order.
    """
    n = config.n_elements
    n_levels = config.n_phase_levels
    n_samples = len(d)
    
    phases = np.zeros(n)
    weights = np.exp(1j * phases)
    
    mse_curve = []
    updates_per_iter = 1
    
    for iteration in range(n_iterations):
        sample_idx = iteration % n_samples
        x = X[sample_idx]
        
        y = np.dot(weights.conj(), x)
        e = d[sample_idx] - y
        
        # Sequential: update only element (iteration % n)
        update_idx = iteration % n
        grad = e.conj() * x[update_idx]
        phases[update_idx] = phases[update_idx] + mu * np.real(grad * np.exp(-1j * phases[update_idx]))
        
        phases = quantize_phases(phases, n_levels)
        weights = np.exp(1j * phases)
        
        mse = compute_mse(weights, X, d)
        mse_curve.append(mse)
    
    return weights, mse_curve, updates_per_iter


def mmax_lms(
    X: np.ndarray,
    d: np.ndarray,
    config: ArrayConfig,
    mu: float = 0.01,
    n_iterations: int = 300,
    m_fraction: float = 0.3,  # Update top 30%
) -> Tuple[np.ndarray, List[float], int]:
    """
    M-max PU-LMS: Updates M elements with largest gradient magnitudes.
    """
    n = config.n_elements
    n_levels = config.n_phase_levels
    n_samples = len(d)
    m = max(1, int(n * m_fraction))
    
    phases = np.zeros(n)
    weights = np.exp(1j * phases)
    
    mse_curve = []
    updates_per_iter = m
    
    for iteration in range(n_iterations):
        sample_idx = iteration % n_samples
        x = X[sample_idx]
        
        y = np.dot(weights.conj(), x)
        e = d[sample_idx] - y
        
        # Compute all gradients
        grad = e.conj() * x
        grad_phase = np.real(grad * np.exp(-1j * phases))
        
        # Select M elements with largest gradient magnitudes
        top_m_idx = np.argsort(np.abs(grad_phase))[-m:]
        
        # Update only selected elements
        phases[top_m_idx] = phases[top_m_idx] + mu * grad_phase[top_m_idx]
        
        phases = quantize_phases(phases, n_levels)
        weights = np.exp(1j * phases)
        
        mse = compute_mse(weights, X, d)
        mse_curve.append(mse)
    
    return weights, mse_curve, updates_per_iter


def periodic_lms(
    X: np.ndarray,
    d: np.ndarray,
    config: ArrayConfig,
    mu: float = 0.01,
    n_iterations: int = 300,
    period: int = 3,  # Update every 3rd element
) -> Tuple[np.ndarray, List[float], int]:
    """
    Periodic PU-LMS: Updates every K-th element in repeating pattern.
    """
    n = config.n_elements
    n_levels = config.n_phase_levels
    n_samples = len(d)
    
    phases = np.zeros(n)
    weights = np.exp(1j * phases)
    
    mse_curve = []
    m = len(range(0, n, period))
    updates_per_iter = m
    
    for iteration in range(n_iterations):
        sample_idx = iteration % n_samples
        x = X[sample_idx]
        
        y = np.dot(weights.conj(), x)
        e = d[sample_idx] - y
        
        grad = e.conj() * x
        grad_phase = np.real(grad * np.exp(-1j * phases))
        
        # Periodic: update elements 0, period, 2*period, ...
        # Shift pattern each iteration
        offset = iteration % period
        update_idx = np.arange(offset, n, period)
        
        phases[update_idx] = phases[update_idx] + mu * grad_phase[update_idx]
        
        phases = quantize_phases(phases, n_levels)
        weights = np.exp(1j * phases)
        
        mse = compute_mse(weights, X, d)
        mse_curve.append(mse)
    
    return weights, mse_curve, updates_per_iter


def stochastic_lms(
    X: np.ndarray,
    d: np.ndarray,
    config: ArrayConfig,
    mu: float = 0.01,
    n_iterations: int = 300,
    m_fraction: float = 0.3,
    seed: int = 42,
) -> Tuple[np.ndarray, List[float], int]:
    """
    Stochastic PU-LMS: Randomly selects M elements each iteration.
    """
    rng = np.random.default_rng(seed)
    n = config.n_elements
    n_levels = config.n_phase_levels
    n_samples = len(d)
    m = max(1, int(n * m_fraction))
    
    phases = np.zeros(n)
    weights = np.exp(1j * phases)
    
    mse_curve = []
    updates_per_iter = m
    
    for iteration in range(n_iterations):
        sample_idx = iteration % n_samples
        x = X[sample_idx]
        
        y = np.dot(weights.conj(), x)
        e = d[sample_idx] - y
        
        grad = e.conj() * x
        grad_phase = np.real(grad * np.exp(-1j * phases))
        
        # Randomly select M elements
        update_idx = rng.choice(n, m, replace=False)
        
        phases[update_idx] = phases[update_idx] + mu * grad_phase[update_idx]
        
        phases = quantize_phases(phases, n_levels)
        weights = np.exp(1j * phases)
        
        mse = compute_mse(weights, X, d)
        mse_curve.append(mse)
    
    return weights, mse_curve, updates_per_iter


# =============================================================================
# PSON Algorithm (adapted for LMS-style comparison)
# =============================================================================

def pson_beamforming(
    X: np.ndarray,
    d: np.ndarray,
    config: ArrayConfig,
    n_iterations: int = 300,
    seed: int = 42,
) -> Tuple[np.ndarray, List[float], int]:
    """
    PSON algorithm for beamforming.
    
    Key differences from PU-LMS:
    - Non-local credit assignment (uses global MSE, not local gradients)
    - Precision-scaled exploration (uncertain elements explore more)
    - Monotonic descent (only accepts improvements)
    """
    rng = np.random.default_rng(seed)
    n = config.n_elements
    n_levels = config.n_phase_levels
    phase_levels = np.array([i * 2 * np.pi / n_levels for i in range(n_levels)])
    
    # Initialize with steering phases toward target
    sv_target = steering_vector(n, config.element_spacing, config.target_angle)
    phases = np.angle(sv_target)
    phases = quantize_phases(phases, n_levels)
    weights = np.exp(1j * phases)
    
    # Precision from element position (edge elements = less certain)
    positions = np.arange(n) * config.element_spacing
    pos_center = np.mean(positions)
    irregularity = np.abs(positions - pos_center)
    precision = 1.0 / (1.0 + irregularity / np.max(irregularity))
    weights_pson = (1.0 - precision) / (np.sum(1.0 - precision) + 1e-8)
    
    mse_curve = []
    best_mse = compute_mse(weights, X, d)
    best_phases = phases.copy()
    
    # Average updates (adaptive)
    avg_updates = 0
    
    for iteration in range(n_iterations):
        mse_cur = compute_mse(weights, X, d)
        mse_curve.append(mse_cur)
        
        if mse_cur < best_mse:
            best_mse = mse_cur
            best_phases = phases.copy()
        
        # Energy for exploration scaling
        energy = mse_cur / (best_mse + 1e-10)
        energy = min(energy, 2.0)  # Cap
        
        # PSON: weighted exploration
        change_probs = weights_pson * (0.2 + 0.8 * (energy - 1.0 + 1.0))
        change_probs = np.clip(change_probs, 0.01, None)
        change_probs = change_probs / change_probs.sum()
        
        # Number of changes scales with energy
        n_changes = max(1, int(n * 0.2 * (0.3 + 0.7 * min(energy, 1.5))))
        avg_updates += n_changes
        
        change_idx = rng.choice(n, min(n_changes, n), replace=False, p=change_probs)
        
        candidate_phases = phases.copy()
        for idx in change_idx:
            current_level = int(round(phases[idx] * n_levels / (2 * np.pi))) % n_levels
            # PSON: precision-scaled step size
            step = 1 if precision[idx] > 0.5 else rng.choice([1, 2])
            delta = rng.choice([-step, step])
            new_level = (current_level + delta) % n_levels
            candidate_phases[idx] = phase_levels[new_level]
        
        candidate_weights = np.exp(1j * candidate_phases)
        mse_new = compute_mse(candidate_weights, X, d)
        
        # Monotonic descent: only accept if better
        if mse_new <= mse_cur:
            phases = candidate_phases
            weights = candidate_weights
    
    avg_updates = avg_updates / n_iterations
    
    return np.exp(1j * best_phases), mse_curve, int(avg_updates)


# =============================================================================
# Main Test
# =============================================================================

def run_comparison(config: ArrayConfig, n_iterations: int = 300, seed: int = 42) -> Dict:
    """Run full comparison of all algorithms."""
    
    # Generate signal
    X, d = generate_received_signal(config, n_samples=100, seed=seed)
    
    results = {}
    
    # Full LMS (baseline)
    weights, mse_curve, updates = full_lms(X, d, config, n_iterations=n_iterations)
    metrics = compute_beam_metrics(weights, config)
    results["Full-LMS"] = {
        "final_mse": float(mse_curve[-1]),
        "convergence_iter": int(np.argmin(np.array(mse_curve) > mse_curve[-1] * 1.1) or n_iterations),
        "updates_per_iter": updates,
        **metrics,
    }
    
    # Sequential PU-LMS
    weights, mse_curve, updates = sequential_lms(X, d, config, n_iterations=n_iterations)
    metrics = compute_beam_metrics(weights, config)
    results["Sequential-LMS"] = {
        "final_mse": float(mse_curve[-1]),
        "convergence_iter": int(np.argmin(np.array(mse_curve) > mse_curve[-1] * 1.1) or n_iterations),
        "updates_per_iter": updates,
        **metrics,
    }
    
    # M-max PU-LMS
    weights, mse_curve, updates = mmax_lms(X, d, config, n_iterations=n_iterations)
    metrics = compute_beam_metrics(weights, config)
    results["M-max-LMS"] = {
        "final_mse": float(mse_curve[-1]),
        "convergence_iter": int(np.argmin(np.array(mse_curve) > mse_curve[-1] * 1.1) or n_iterations),
        "updates_per_iter": updates,
        **metrics,
    }
    
    # Periodic PU-LMS
    weights, mse_curve, updates = periodic_lms(X, d, config, n_iterations=n_iterations)
    metrics = compute_beam_metrics(weights, config)
    results["Periodic-LMS"] = {
        "final_mse": float(mse_curve[-1]),
        "convergence_iter": int(np.argmin(np.array(mse_curve) > mse_curve[-1] * 1.1) or n_iterations),
        "updates_per_iter": updates,
        **metrics,
    }
    
    # Stochastic PU-LMS
    weights, mse_curve, updates = stochastic_lms(X, d, config, n_iterations=n_iterations, seed=seed)
    metrics = compute_beam_metrics(weights, config)
    results["Stochastic-LMS"] = {
        "final_mse": float(mse_curve[-1]),
        "convergence_iter": int(np.argmin(np.array(mse_curve) > mse_curve[-1] * 1.1) or n_iterations),
        "updates_per_iter": updates,
        **metrics,
    }
    
    # PSON
    weights, mse_curve, updates = pson_beamforming(X, d, config, n_iterations=n_iterations, seed=seed)
    metrics = compute_beam_metrics(weights, config)
    results["PSON"] = {
        "final_mse": float(mse_curve[-1]),
        "convergence_iter": int(np.argmin(np.array(mse_curve) > mse_curve[-1] * 1.1) or n_iterations),
        "updates_per_iter": updates,
        **metrics,
    }
    
    return results


def main():
    print("=" * 80)
    print("PSON vs PARTIAL UPDATE (PU) ADAPTIVE ALGORITHMS")
    print("Head-to-Head Comparison for Phased Array Beamforming")
    print("=" * 80)
    
    all_results = []
    
    # Test configurations
    configs = [
        ArrayConfig(n_elements=16, n_phase_levels=32, target_angle=0, interferer_angles=[-30, 30], snr_db=15),
        ArrayConfig(n_elements=16, n_phase_levels=16, target_angle=30, interferer_angles=[-20, 45], snr_db=10),
        ArrayConfig(n_elements=32, n_phase_levels=32, target_angle=0, interferer_angles=[-30, 30], snr_db=15),
        ArrayConfig(n_elements=32, n_phase_levels=64, target_angle=45, interferer_angles=[0, -45], snr_db=20),
    ]
    
    seeds = [42, 123, 456]
    
    for i, config in enumerate(configs):
        config_name = f"{config.n_elements}-elem, {int(np.log2(config.n_phase_levels))}-bit, {config.target_angle}deg, SNR={config.snr_db}dB"
        print(f"\n{'='*70}")
        print(f"Configuration {i+1}: {config_name}")
        print(f"{'='*70}")
        
        config_results = {"config": config_name, "runs": []}
        
        for seed in seeds:
            results = run_comparison(config, n_iterations=300, seed=seed)
            
            # Find winner by MSE
            mse_winner = min(results.items(), key=lambda x: x[1]["final_mse"])[0]
            # Find winner by main beam gain
            gain_winner = max(results.items(), key=lambda x: x[1]["main_gain_db"])[0]
            # Find winner by null depth (more negative = better)
            null_winner = min(results.items(), key=lambda x: x[1]["null_depth_db"])[0]
            
            print(f"\nSeed {seed}:")
            print(f"  {'Algorithm':<18} {'MSE':>10} {'Gain(dB)':>10} {'Null(dB)':>10} {'Updates':>8}")
            print(f"  {'-'*60}")
            for name, res in results.items():
                marker = "*" if name == mse_winner else " "
                print(f" {marker}{name:<17} {res['final_mse']:>10.4f} {res['main_gain_db']:>10.1f} {res['null_depth_db']:>10.1f} {res['updates_per_iter']:>8}")
            
            print(f"  Winners: MSE={mse_winner}, Gain={gain_winner}, Null={null_winner}")
            
            config_results["runs"].append({
                "seed": seed,
                "results": results,
                "winners": {"mse": mse_winner, "gain": gain_winner, "null": null_winner},
            })
        
        # Summarize
        pson_mse_wins = sum(1 for r in config_results["runs"] if r["winners"]["mse"] == "PSON")
        pson_gain_wins = sum(1 for r in config_results["runs"] if r["winners"]["gain"] == "PSON")
        
        config_results["summary"] = {
            "pson_mse_wins": pson_mse_wins,
            "pson_gain_wins": pson_gain_wins,
            "total_runs": len(seeds),
        }
        
        print(f"\nPSON Summary: MSE wins {pson_mse_wins}/{len(seeds)}, Gain wins {pson_gain_wins}/{len(seeds)}")
        
        all_results.append(config_results)
    
    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    
    total_mse_wins = {"Full-LMS": 0, "Sequential-LMS": 0, "M-max-LMS": 0, 
                      "Periodic-LMS": 0, "Stochastic-LMS": 0, "PSON": 0}
    total_gain_wins = {"Full-LMS": 0, "Sequential-LMS": 0, "M-max-LMS": 0,
                       "Periodic-LMS": 0, "Stochastic-LMS": 0, "PSON": 0}
    
    for config_result in all_results:
        for run in config_result["runs"]:
            total_mse_wins[run["winners"]["mse"]] += 1
            total_gain_wins[run["winners"]["gain"]] += 1
    
    total_runs = sum(total_mse_wins.values())
    
    print(f"\nMSE Wins (lower is better):")
    for name in sorted(total_mse_wins.keys(), key=lambda x: -total_mse_wins[x]):
        pct = 100 * total_mse_wins[name] / total_runs
        bar = "*" * int(pct / 5)
        print(f"  {name:<18} {total_mse_wins[name]:>3}/{total_runs} ({pct:>5.1f}%) {bar}")
    
    print(f"\nGain Wins (higher is better):")
    for name in sorted(total_gain_wins.keys(), key=lambda x: -total_gain_wins[x]):
        pct = 100 * total_gain_wins[name] / total_runs
        bar = "*" * int(pct / 5)
        print(f"  {name:<18} {total_gain_wins[name]:>3}/{total_runs} ({pct:>5.1f}%) {bar}")
    
    # Efficiency comparison
    print(f"\nComputational Efficiency (updates per iteration):")
    print(f"  Full-LMS:       N (all elements)")
    print(f"  Sequential-LMS: 1 (minimal)")
    print(f"  M-max-LMS:      ~0.3N (30%)")
    print(f"  Periodic-LMS:   ~0.33N (1/period)")
    print(f"  Stochastic-LMS: ~0.3N (30%)")
    print(f"  PSON:           ~0.2N (adaptive, 20% avg)")
    
    # Save results
    with open("pson_vs_pu_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nResults saved: pson_vs_pu_results.json")


if __name__ == "__main__":
    main()

