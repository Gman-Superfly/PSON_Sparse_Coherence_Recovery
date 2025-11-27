"""
PSON vs PU-BAA (November 2025 Research)
=======================================

Comparison against the PU-BAA approach from:
    Shubber, Jamel, Nahar (2025). "Beamforming Array Antenna: New Innovative 
    Research Using Partial Update Adaptive Algorithms"
    AIP Conference Proceedings, Volume 3350, DOI: 10.1063/5.0298348

KEY DIFFERENCES FROM OUR PREVIOUS TEST:
1. Uses NLMS (Normalized LMS) not just LMS
2. Includes multipath fading (3 taps)
3. Matches their exact simulation setup:
   - N=16 elements, λ/2 spacing
   - Desired signal at 0°, interferers at ±30°
   - SNR=15 dB
   - 3-tap multipath channel

THEIR REPORTED RESULTS:
| Method           | Convergence | MSE (dB) | Null Depth | Complexity |
|------------------|-------------|----------|------------|------------|
| Full NLMS        | 200-300     | -25      | -40 dB     | 0%         |
| M-max PU-NLMS    | 100-150     | -24      | -38 dB     | 70%        |
| Periodic PU-NLMS | 120-180     | -25      | -39 dB     | 65%        |
| Stochastic PU    | 130-170     | -24.5    | -39 dB     | 75%        |

Usage:
    uv run python experiments/discrete_applications/pson_vs_pu_baa_test.py
"""

import numpy as np
import json
from typing import Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class PUBAAConfig:
    """Configuration matching the PU-BAA paper."""
    n_elements: int = 16
    element_spacing: float = 0.5  # λ/2
    desired_angle: float = 0.0  # degrees
    interferer_angles: List[float] = None
    snr_db: float = 15.0
    n_multipath_taps: int = 3
    n_iterations: int = 500
    
    def __post_init__(self):
        if self.interferer_angles is None:
            self.interferer_angles = [-30.0, 30.0]


def steering_vector(n_elements: int, spacing: float, angle_deg: float) -> np.ndarray:
    """Compute steering vector for ULA."""
    angle_rad = np.deg2rad(angle_deg)
    k = 2 * np.pi
    n = np.arange(n_elements)
    return np.exp(1j * k * spacing * n * np.sin(angle_rad))


def generate_multipath_channel(n_taps: int, seed: int = 42) -> np.ndarray:
    """Generate multipath fading channel (Rayleigh)."""
    rng = np.random.default_rng(seed)
    # Rayleigh fading coefficients with exponential decay
    h = (rng.standard_normal(n_taps) + 1j * rng.standard_normal(n_taps)) / np.sqrt(2)
    decay = np.exp(-np.arange(n_taps) / 2)  # Exponential power decay
    h = h * decay
    h = h / np.linalg.norm(h)  # Normalize
    return h


def generate_received_signal(
    config: PUBAAConfig,
    n_samples: int = 1000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate received signal with multipath fading.
    
    Returns:
        X: (n_samples, n_elements) received signal matrix
        d: (n_samples,) desired signal
        channel: multipath channel coefficients
    """
    rng = np.random.default_rng(seed)
    n = config.n_elements
    
    # Multipath channel
    channel = generate_multipath_channel(config.n_multipath_taps, seed)
    
    # Desired signal (QPSK-like)
    s_desired = (rng.choice([-1, 1], n_samples) + 1j * rng.choice([-1, 1], n_samples)) / np.sqrt(2)
    a_desired = steering_vector(n, config.element_spacing, config.desired_angle)
    
    # Apply multipath to desired signal
    s_desired_multipath = np.convolve(s_desired, channel, mode='same')
    
    # Interferer signals
    interference_power = 10 ** (-config.snr_db / 20)  # Relative to desired
    X = np.outer(s_desired_multipath, a_desired)
    
    for angle in config.interferer_angles:
        s_int = interference_power * (rng.choice([-1, 1], n_samples) + 1j * rng.choice([-1, 1], n_samples)) / np.sqrt(2)
        a_int = steering_vector(n, config.element_spacing, angle)
        X += np.outer(s_int, a_int)
    
    # Noise
    noise_power = 10 ** (-config.snr_db / 10)
    noise = np.sqrt(noise_power/2) * (rng.standard_normal((n_samples, n)) + 1j * rng.standard_normal((n_samples, n)))
    X += noise
    
    return X, s_desired, channel


def compute_metrics(
    weights: np.ndarray,
    config: PUBAAConfig,
    mse_history: List[float],
) -> Dict[str, float]:
    """Compute all metrics from the PU-BAA paper."""
    n = config.n_elements
    
    # Radiation pattern
    angles = np.linspace(-90, 90, 361)
    pattern = np.zeros(len(angles), dtype=complex)
    for i, angle in enumerate(angles):
        sv = steering_vector(n, config.element_spacing, angle)
        pattern[i] = np.dot(weights.conj(), sv)
    pattern_db = 20 * np.log10(np.abs(pattern) + 1e-10)
    
    # Normalize to peak
    pattern_db = pattern_db - np.max(pattern_db)
    
    # Main lobe gain at desired angle
    desired_idx = np.argmin(np.abs(angles - config.desired_angle))
    main_gain = pattern_db[desired_idx]
    
    # Null depths at interferer angles
    null_depths = []
    for int_angle in config.interferer_angles:
        int_idx = np.argmin(np.abs(angles - int_angle))
        null_depths.append(pattern_db[int_idx])
    avg_null_depth = np.mean(null_depths)
    
    # Sidelobe level (peak outside main beam ±10°)
    main_mask = np.abs(angles - config.desired_angle) < 10
    sidelobe_pattern = pattern_db.copy()
    sidelobe_pattern[main_mask] = -100
    peak_sidelobe = np.max(sidelobe_pattern)
    
    # Convergence: iterations to reach within 3dB of final MSE
    final_mse = mse_history[-1] if mse_history else 1.0
    threshold = final_mse * 2  # Within 3dB
    convergence_iter = len(mse_history)
    for i, mse in enumerate(mse_history):
        if mse < threshold:
            convergence_iter = i
            break
    
    # Steady-state MSE in dB
    steady_mse_db = 10 * np.log10(final_mse + 1e-10)
    
    return {
        "convergence_iter": convergence_iter,
        "steady_mse_db": float(steady_mse_db),
        "null_depth_db": float(avg_null_depth),
        "sidelobe_db": float(peak_sidelobe),
        "main_gain_db": float(main_gain),
    }


# =============================================================================
# NLMS Variants (from PU-BAA paper)
# =============================================================================

def full_nlms(
    X: np.ndarray,
    d: np.ndarray,
    config: PUBAAConfig,
    mu: float = 0.5,
    delta: float = 1e-6,
) -> Tuple[np.ndarray, List[float], int]:
    """Full NLMS - updates all elements."""
    n = config.n_elements
    n_samples = len(d)
    
    weights = np.zeros(n, dtype=complex)
    mse_history = []
    
    for iteration in range(config.n_iterations):
        idx = iteration % n_samples
        x = X[idx]
        
        y = np.dot(weights.conj(), x)
        e = d[idx] - y
        
        # NLMS update: normalized by input power
        norm_factor = np.dot(x.conj(), x).real + delta
        weights = weights + mu * e.conj() * x / norm_factor
        
        # MSE over recent window
        if iteration % 10 == 0:
            y_all = X @ weights
            mse = np.mean(np.abs(d[:len(y_all)] - y_all) ** 2)
            mse_history.append(float(mse))
    
    return weights, mse_history, n


def mmax_nlms(
    X: np.ndarray,
    d: np.ndarray,
    config: PUBAAConfig,
    mu: float = 0.5,
    delta: float = 1e-6,
    m_fraction: float = 0.3,
) -> Tuple[np.ndarray, List[float], int]:
    """M-max PU-NLMS - updates M elements with largest gradients."""
    n = config.n_elements
    n_samples = len(d)
    m = max(1, int(n * m_fraction))
    
    weights = np.zeros(n, dtype=complex)
    mse_history = []
    
    for iteration in range(config.n_iterations):
        idx = iteration % n_samples
        x = X[idx]
        
        y = np.dot(weights.conj(), x)
        e = d[idx] - y
        
        # Compute gradient for all elements
        norm_factor = np.dot(x.conj(), x).real + delta
        grad = e.conj() * x / norm_factor
        
        # Select M elements with largest gradient magnitude
        top_m_idx = np.argsort(np.abs(grad))[-m:]
        
        # Update only selected elements
        weights[top_m_idx] = weights[top_m_idx] + mu * grad[top_m_idx]
        
        if iteration % 10 == 0:
            y_all = X @ weights
            mse = np.mean(np.abs(d[:len(y_all)] - y_all) ** 2)
            mse_history.append(float(mse))
    
    return weights, mse_history, m


def periodic_nlms(
    X: np.ndarray,
    d: np.ndarray,
    config: PUBAAConfig,
    mu: float = 0.5,
    delta: float = 1e-6,
    period: int = 3,
) -> Tuple[np.ndarray, List[float], int]:
    """Periodic PU-NLMS - updates every K-th element."""
    n = config.n_elements
    n_samples = len(d)
    
    weights = np.zeros(n, dtype=complex)
    mse_history = []
    m = len(range(0, n, period))
    
    for iteration in range(config.n_iterations):
        idx = iteration % n_samples
        x = X[idx]
        
        y = np.dot(weights.conj(), x)
        e = d[idx] - y
        
        norm_factor = np.dot(x.conj(), x).real + delta
        grad = e.conj() * x / norm_factor
        
        # Periodic: update elements offset, offset+period, offset+2*period, ...
        offset = iteration % period
        update_idx = np.arange(offset, n, period)
        
        weights[update_idx] = weights[update_idx] + mu * grad[update_idx]
        
        if iteration % 10 == 0:
            y_all = X @ weights
            mse = np.mean(np.abs(d[:len(y_all)] - y_all) ** 2)
            mse_history.append(float(mse))
    
    return weights, mse_history, m


def stochastic_nlms(
    X: np.ndarray,
    d: np.ndarray,
    config: PUBAAConfig,
    mu: float = 0.5,
    delta: float = 1e-6,
    m_fraction: float = 0.3,
    seed: int = 42,
) -> Tuple[np.ndarray, List[float], int]:
    """Stochastic PU-NLMS - randomly selects M elements."""
    rng = np.random.default_rng(seed)
    n = config.n_elements
    n_samples = len(d)
    m = max(1, int(n * m_fraction))
    
    weights = np.zeros(n, dtype=complex)
    mse_history = []
    
    for iteration in range(config.n_iterations):
        idx = iteration % n_samples
        x = X[idx]
        
        y = np.dot(weights.conj(), x)
        e = d[idx] - y
        
        norm_factor = np.dot(x.conj(), x).real + delta
        grad = e.conj() * x / norm_factor
        
        # Randomly select M elements
        update_idx = rng.choice(n, m, replace=False)
        
        weights[update_idx] = weights[update_idx] + mu * grad[update_idx]
        
        if iteration % 10 == 0:
            y_all = X @ weights
            mse = np.mean(np.abs(d[:len(y_all)] - y_all) ** 2)
            mse_history.append(float(mse))
    
    return weights, mse_history, m


# =============================================================================
# PSON for comparison
# =============================================================================

def pson_beamforming(
    X: np.ndarray,
    d: np.ndarray,
    config: PUBAAConfig,
    seed: int = 42,
) -> Tuple[np.ndarray, List[float], int]:
    """PSON algorithm adapted for complex beamforming weights."""
    rng = np.random.default_rng(seed)
    n = config.n_elements
    n_samples = len(d)
    
    # Initialize with steering vector toward desired
    weights = steering_vector(n, config.element_spacing, config.desired_angle)
    weights = weights / np.linalg.norm(weights)
    
    # Precision from element position (edge elements less certain)
    positions = np.arange(n) * config.element_spacing
    pos_center = np.mean(positions)
    irregularity = np.abs(positions - pos_center)
    precision = 1.0 / (1.0 + irregularity / np.max(irregularity + 1e-8))
    pson_weights = (1.0 - precision) / (np.sum(1.0 - precision) + 1e-8)
    
    mse_history = []
    best_weights = weights.copy()
    best_mse = np.inf
    avg_updates = 0
    
    for iteration in range(config.n_iterations):
        # Compute current MSE
        y_all = X @ weights
        mse_cur = np.mean(np.abs(d[:len(y_all)] - y_all) ** 2)
        
        if iteration % 10 == 0:
            mse_history.append(float(mse_cur))
        
        if mse_cur < best_mse:
            best_mse = mse_cur
            best_weights = weights.copy()
        
        # Energy for exploration
        energy = min(mse_cur / (best_mse + 1e-10), 2.0)
        
        # Non-local gradient (simplified for complex weights)
        idx = iteration % n_samples
        x = X[idx]
        y = np.dot(weights.conj(), x)
        e = d[idx] - y
        grad = e.conj() * x  # Direction toward reducing error
        
        # Deterministic proposal
        lr = 0.1
        proposal = weights + lr * grad
        proposal = proposal / (np.linalg.norm(proposal) + 1e-10)
        
        # PSON exploration
        change_probs = pson_weights * (0.2 + 0.8 * energy)
        change_probs = change_probs / change_probs.sum()
        
        n_changes = max(1, int(n * 0.2 * (0.3 + 0.7 * min(energy, 1.5))))
        avg_updates += n_changes
        
        change_idx = rng.choice(n, min(n_changes, n), replace=False, p=change_probs)
        
        # Add orthogonal noise to selected elements
        noise_scale = 0.1 * energy
        noise = noise_scale * (rng.standard_normal(len(change_idx)) + 1j * rng.standard_normal(len(change_idx)))
        
        candidate = proposal.copy()
        candidate[change_idx] = candidate[change_idx] + noise / np.sqrt(precision[change_idx] + 1e-8)
        candidate = candidate / (np.linalg.norm(candidate) + 1e-10)
        
        # Monotonic acceptance
        y_new = X @ candidate
        mse_new = np.mean(np.abs(d[:len(y_new)] - y_new) ** 2)
        
        if mse_new <= mse_cur:
            weights = candidate
    
    avg_updates = avg_updates / config.n_iterations
    return best_weights, mse_history, int(avg_updates)


# =============================================================================
# Main Test
# =============================================================================

def main():
    print("=" * 80)
    print("PSON vs PU-BAA (November 2025 Research)")
    print("Shubber, Jamel, Nahar - AIP Conference Proceedings")
    print("=" * 80)
    
    config = PUBAAConfig(
        n_elements=16,
        element_spacing=0.5,
        desired_angle=0.0,
        interferer_angles=[-30.0, 30.0],
        snr_db=15.0,
        n_multipath_taps=3,
        n_iterations=500,
    )
    
    print(f"\nConfiguration (matching PU-BAA paper):")
    print(f"  Elements: {config.n_elements}")
    print(f"  Spacing: {config.element_spacing} wavelengths")
    print(f"  Desired: {config.desired_angle}°")
    print(f"  Interferers: {config.interferer_angles}°")
    print(f"  SNR: {config.snr_db} dB")
    print(f"  Multipath taps: {config.n_multipath_taps}")
    print(f"  Iterations: {config.n_iterations}")
    
    seeds = [42, 123, 456]
    all_results = []
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed: {seed}")
        print(f"{'='*60}")
        
        # Generate signal
        X, d, channel = generate_received_signal(config, n_samples=1000, seed=seed)
        
        results = {}
        
        # Full NLMS
        weights, mse_hist, updates = full_nlms(X, d, config)
        metrics = compute_metrics(weights, config, mse_hist)
        metrics["updates_per_iter"] = updates
        metrics["complexity_reduction"] = 0.0
        results["Full-NLMS"] = metrics
        
        # M-max PU-NLMS
        weights, mse_hist, updates = mmax_nlms(X, d, config)
        metrics = compute_metrics(weights, config, mse_hist)
        metrics["updates_per_iter"] = updates
        metrics["complexity_reduction"] = 100 * (1 - updates / config.n_elements)
        results["M-max-NLMS"] = metrics
        
        # Periodic PU-NLMS
        weights, mse_hist, updates = periodic_nlms(X, d, config)
        metrics = compute_metrics(weights, config, mse_hist)
        metrics["updates_per_iter"] = updates
        metrics["complexity_reduction"] = 100 * (1 - updates / config.n_elements)
        results["Periodic-NLMS"] = metrics
        
        # Stochastic PU-NLMS
        weights, mse_hist, updates = stochastic_nlms(X, d, config, seed=seed)
        metrics = compute_metrics(weights, config, mse_hist)
        metrics["updates_per_iter"] = updates
        metrics["complexity_reduction"] = 100 * (1 - updates / config.n_elements)
        results["Stochastic-NLMS"] = metrics
        
        # PSON
        weights, mse_hist, updates = pson_beamforming(X, d, config, seed=seed)
        metrics = compute_metrics(weights, config, mse_hist)
        metrics["updates_per_iter"] = updates
        metrics["complexity_reduction"] = 100 * (1 - updates / config.n_elements)
        results["PSON"] = metrics
        
        # Find winners
        mse_winner = min(results.items(), key=lambda x: x[1]["steady_mse_db"])[0]
        null_winner = min(results.items(), key=lambda x: x[1]["null_depth_db"])[0]
        conv_winner = min(results.items(), key=lambda x: x[1]["convergence_iter"])[0]
        
        # Print results
        print(f"\n{'Method':<18} {'Conv':>6} {'MSE(dB)':>8} {'Null(dB)':>9} {'SL(dB)':>8} {'Upd':>4} {'Red%':>5}")
        print("-" * 70)
        for name, m in results.items():
            marker = "*" if name == mse_winner else " "
            print(f"{marker}{name:<17} {m['convergence_iter']:>6} {m['steady_mse_db']:>8.1f} {m['null_depth_db']:>9.1f} {m['sidelobe_db']:>8.1f} {m['updates_per_iter']:>4} {m['complexity_reduction']:>5.0f}%")
        
        print(f"\nWinners: MSE={mse_winner}, Null={null_winner}, Conv={conv_winner}")
        
        all_results.append({
            "seed": seed,
            "results": results,
            "winners": {"mse": mse_winner, "null": null_winner, "convergence": conv_winner}
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY ACROSS ALL SEEDS")
    print("=" * 80)
    
    method_wins = {m: {"mse": 0, "null": 0, "conv": 0} for m in ["Full-NLMS", "M-max-NLMS", "Periodic-NLMS", "Stochastic-NLMS", "PSON"]}
    
    for r in all_results:
        method_wins[r["winners"]["mse"]]["mse"] += 1
        method_wins[r["winners"]["null"]]["null"] += 1
        method_wins[r["winners"]["convergence"]]["conv"] += 1
    
    print(f"\n{'Method':<18} {'MSE Wins':>10} {'Null Wins':>10} {'Conv Wins':>10}")
    print("-" * 50)
    for name, wins in method_wins.items():
        print(f"{name:<18} {wins['mse']:>10} {wins['null']:>10} {wins['conv']:>10}")
    
    # Compare to PU-BAA paper's reported results
    print("\n" + "=" * 80)
    print("COMPARISON TO PU-BAA PAPER'S REPORTED RESULTS")
    print("=" * 80)
    print("\nPU-BAA Paper (Shubber et al. 2025):")
    print("| Method           | Conv   | MSE(dB) | Null(dB) |")
    print("|------------------|--------|---------|----------|")
    print("| Full NLMS        | 200-300| -25     | -40      |")
    print("| M-max PU-NLMS    | 100-150| -24     | -38      |")
    print("| Periodic PU-NLMS | 120-180| -25     | -39      |")
    print("| Stochastic PU    | 130-170| -24.5   | -39      |")
    
    # Save results
    output = {
        "config": {
            "n_elements": config.n_elements,
            "snr_db": config.snr_db,
            "interferers": config.interferer_angles,
            "multipath_taps": config.n_multipath_taps,
        },
        "results": all_results,
        "summary": method_wins,
    }
    
    with open("pson_vs_pu_baa_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved: pson_vs_pu_baa_results.json")


if __name__ == "__main__":
    main()

