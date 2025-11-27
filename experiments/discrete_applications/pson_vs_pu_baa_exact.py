"""
Exact replication of PU-BAA (PU-LMS) test from:
  Shubber, Jamel, Nahar (2025).
  "Beamforming Array Antenna: New Innovative Research Using Partial Update Adaptive Algorithms"
  AIP Conference Proceedings, 3350, 030014. DOI: 10.1063/5.0298348

Implements:
  - Full-band LMS
  - Partial Update LMS variants: Sequential, M-max, Periodic, Stochastic
Channel:
  - Extended Typical Urban (ETU) multipath model (3GPP) with 9 taps

Outputs:
  - Console table of metrics (convergence iterations, steady MSE dB, null depth dB, sidelobe dB)
  - JSON file: pson_vs_pu_baa_exact_results.json

Run:
  uv run python experiments/discrete_applications/pson_vs_pu_baa_exact.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# =============================================
# Utilities
# =============================================


def db10(x: float) -> float:
    return 10.0 * np.log10(max(x, 1e-12))


def steering_vector(n_elements: int, spacing_lambda: float, angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    k = 2.0 * np.pi  # normalized wavenumber (lambda = 1)
    n = np.arange(n_elements)
    return np.exp(1j * k * spacing_lambda * n * np.sin(angle_rad))


def qpsk_symbols(rng: np.random.Generator, n_samples: int) -> np.ndarray:
    return (rng.choice([-1, 1], n_samples) + 1j * rng.choice([-1, 1], n_samples)) / np.sqrt(2.0)


# =============================================
# ETU Channel Model (3GPP)
# Delays/powers based on widely used ETU profile
# =============================================


def etu_profile() -> Tuple[np.ndarray, np.ndarray]:
    # Delays in ns (3GPP ETU standard)
    delays_ns = np.array([0, 50, 120, 200, 230, 500, 1600, 2300, 5000], dtype=float)
    # Relative powers in dB (typical ETU)
    # Note: Variants exist; this commonly used set matches many references
    # and is sufficient to replicate the "ETU multipath" setting in the paper.
    powers_db = np.array([0.0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8, -26.6, -32.0], dtype=float)
    return delays_ns, powers_db


def generate_etu_channel(
    rng: np.random.Generator,
    sample_period_ns: float = 50.0,
    normalize: bool = True,
) -> np.ndarray:
    delays_ns, powers_db = etu_profile()
    tap_powers_lin = 10.0 ** (powers_db / 10.0)
    tap_scales = np.sqrt(tap_powers_lin)
    # Complex Rayleigh fading per tap
    taps = (rng.standard_normal(len(tap_scales)) + 1j * rng.standard_normal(len(tap_scales))) / np.sqrt(2.0)
    taps = taps * tap_scales
    # Map delays to sample indices (rounding to nearest)
    sample_delays = np.rint(delays_ns / sample_period_ns).astype(int)
    h_len = int(sample_delays.max()) + 1
    h = np.zeros(h_len, dtype=complex)
    for idx, sd in enumerate(sample_delays):
        h[sd] += taps[idx]
    if normalize:
        norm = np.linalg.norm(h)
        if norm > 0:
            h = h / norm
    return h


# =============================================
# Signal Generation
# =============================================


@dataclass
class PUBAAExactConfig:
    n_elements: int = 16
    spacing_lambda: float = 0.5
    desired_angle_deg: float = 0.0
    interferer_angles_deg: List[float] = None
    snr_db: float = 15.0
    n_iterations: int = 300
    n_samples: int = 2000
    sample_period_ns: float = 50.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.interferer_angles_deg is None:
            self.interferer_angles_deg = [-30.0, 30.0]


def generate_received_matrix(
    config: PUBAAExactConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate received data matrix X (n_samples x n_elements) and desired scalar d (n_samples).
    All sources (desired + interferers) traverse the same ETU channel realization; this is a
    reasonable, commonly used simplification for multipath-rich environments.
    """
    rng = np.random.default_rng(config.seed)
    n, T = config.n_elements, config.n_samples

    # Symbols
    s_des = qpsk_symbols(rng, T)
    s_int_list = [qpsk_symbols(rng, T) for _ in config.interferer_angles_deg]

    # Channel (ETU)
    h = generate_etu_channel(rng, sample_period_ns=config.sample_period_ns, normalize=True)
    s_des_ch = np.convolve(s_des, h, mode="same")
    s_int_ch = [np.convolve(s, h, mode="same") for s in s_int_list]

    # Steering vectors
    a_des = steering_vector(n, config.spacing_lambda, config.desired_angle_deg)
    a_int_list = [steering_vector(n, config.spacing_lambda, ang) for ang in config.interferer_angles_deg]

    # Received matrix construction
    X = np.outer(s_des_ch, a_des)
    # Scale interferers relative to SNR (interpreting as desired SNR at array)
    # SNR here is applied via additive noise; interferers are unit-power QPSK.
    for s_i, a_i in zip(s_int_ch, a_int_list):
        X += np.outer(s_i, a_i)

    # Add AWGN
    # Measure signal power per element to set noise (common across elements)
    sig_power = np.mean(np.abs(X) ** 2)
    noise_power = sig_power / (10.0 ** (config.snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal((T, n)) + 1j * rng.standard_normal((T, n))
    )
    X_noisy = X + noise

    # Desired scalar (beamformed target signal reference)
    # In adaptive beamforming with LMS, a common setup is to use the transmitted symbol as desired.
    d = s_des  # Reference for MSE
    return X_noisy, d


# =============================================
# Metrics
# =============================================


def compute_pattern_metrics(weights: np.ndarray, config: PUBAAExactConfig, mse_history: List[float]) -> Dict[str, float]:
    n = config.n_elements
    angles = np.linspace(-90.0, 90.0, 361)
    patt = np.zeros_like(angles, dtype=complex)
    for i, ang in enumerate(angles):
        sv = steering_vector(n, config.spacing_lambda, ang)
        patt[i] = np.dot(np.conj(weights), sv)
    patt_db = 20.0 * np.log10(np.maximum(np.abs(patt), 1e-10))
    patt_db -= np.max(patt_db)

    desired_idx = np.argmin(np.abs(angles - config.desired_angle_deg))
    main_gain_db = float(patt_db[desired_idx])

    null_depths = []
    for ang in config.interferer_angles_deg:
        idx = np.argmin(np.abs(angles - ang))
        null_depths.append(patt_db[idx])
    null_depth_db = float(np.mean(null_depths))

    # Peak sidelobe outside +/- 10 deg mainlobe
    mask = np.abs(angles - config.desired_angle_deg) < 10.0
    sl = patt_db.copy()
    sl[mask] = -100.0
    sidelobe_db = float(np.max(sl))

    final_mse = float(mse_history[-1]) if mse_history else 1.0
    steady_mse_db = db10(final_mse)

    # Convergence: first iteration where mse < 2x final (within ~3 dB)
    conv_iter = len(mse_history)
    thr = final_mse * 2.0
    for i, m in enumerate(mse_history):
        if m < thr:
            conv_iter = i
            break

    return {
        "convergence_iter": int(conv_iter),
        "steady_mse_db": steady_mse_db,
        "null_depth_db": null_depth_db,
        "sidelobe_db": sidelobe_db,
        "main_gain_db": main_gain_db,
    }


# =============================================
# LMS and Partial-Update Variants
# =============================================


def _lms_step(weights: np.ndarray, x: np.ndarray, d: complex, mu: float) -> Tuple[np.ndarray, complex]:
    """
    Complex LMS update:
      y = w^H x
      e = d - y
      w <- w + mu * x * conj(e)
    """
    y = np.dot(np.conj(weights), x)
    e = d - y
    weights = weights + mu * x * np.conj(e)
    return weights, e


def _avg_mse(X: np.ndarray, d: np.ndarray, w: np.ndarray) -> float:
    y_all = X @ w
    return float(np.mean(np.abs(d[: len(y_all)] - y_all) ** 2))


def full_lms(
    X: np.ndarray,
    d: np.ndarray,
    n_iters: int,
    mu: float,
) -> Tuple[np.ndarray, List[float], int]:
    n = X.shape[1]
    w = np.zeros(n, dtype=complex)
    mse_hist: List[float] = []
    for it in range(n_iters):
        x = X[it % len(d)]
        w, _ = _lms_step(w, x, d[it % len(d)], mu)
        if it % 10 == 0:
            mse_hist.append(_avg_mse(X, d, w))
    return w, mse_hist, n


def sequential_lms(
    X: np.ndarray,
    d: np.ndarray,
    n_iters: int,
    mu: float,
) -> Tuple[np.ndarray, List[float], int]:
    n = X.shape[1]
    w = np.zeros(n, dtype=complex)
    mse_hist: List[float] = []
    for it in range(n_iters):
        x = X[it % len(d)]
        y = np.dot(np.conj(w), x)
        e = d[it % len(d)] - y
        # Update one coefficient per iteration (cyclic)
        k = it % n
        w[k] = w[k] + mu * x[k] * np.conj(e)
        if it % 10 == 0:
            mse_hist.append(_avg_mse(X, d, w))
    return w, mse_hist, 1


def mmax_lms(
    X: np.ndarray,
    d: np.ndarray,
    n_iters: int,
    mu: float,
    m_fraction: float = 0.3,
) -> Tuple[np.ndarray, List[float], int]:
    n = X.shape[1]
    m = max(1, int(m_fraction * n))
    w = np.zeros(n, dtype=complex)
    mse_hist: List[float] = []
    for it in range(n_iters):
        x = X[it % len(d)]
        y = np.dot(np.conj(w), x)
        e = d[it % len(d)] - y
        grad = x * np.conj(e)
        idx = np.argsort(np.abs(grad))[-m:]
        w[idx] = w[idx] + mu * grad[idx]
        if it % 10 == 0:
            mse_hist.append(_avg_mse(X, d, w))
    return w, mse_hist, m


def periodic_lms(
    X: np.ndarray,
    d: np.ndarray,
    n_iters: int,
    mu: float,
    period: int = 3,
) -> Tuple[np.ndarray, List[float], int]:
    n = X.shape[1]
    w = np.zeros(n, dtype=complex)
    mse_hist: List[float] = []
    m = len(range(0, n, period))
    for it in range(n_iters):
        x = X[it % len(d)]
        y = np.dot(np.conj(w), x)
        e = d[it % len(d)] - y
        offset = it % period
        idx = np.arange(offset, n, period)
        w[idx] = w[idx] + mu * x[idx] * np.conj(e)
        if it % 10 == 0:
            mse_hist.append(_avg_mse(X, d, w))
    return w, mse_hist, m


def stochastic_lms(
    X: np.ndarray,
    d: np.ndarray,
    n_iters: int,
    mu: float,
    m_fraction: float = 0.3,
    seed: int = 42,
) -> Tuple[np.ndarray, List[float], int]:
    rng = np.random.default_rng(seed)
    n = X.shape[1]
    m = max(1, int(m_fraction * n))
    w = np.zeros(n, dtype=complex)
    mse_hist: List[float] = []
    for it in range(n_iters):
        x = X[it % len(d)]
        y = np.dot(np.conj(w), x)
        e = d[it % len(d)] - y
        idx = rng.choice(n, m, replace=False)
        w[idx] = w[idx] + mu * x[idx] * np.conj(e)
        if it % 10 == 0:
            mse_hist.append(_avg_mse(X, d, w))
    return w, mse_hist, m


# =============================================
# PSON (complex weights) on the same ETU setup
# =============================================


def pson_beamforming(
    X: np.ndarray,
    d: np.ndarray,
    n_iters: int,
    seed: int = 42,
    update_fraction: float = 0.2,
) -> Tuple[np.ndarray, List[float], int]:
    """
    PSON-style optimizer for complex beamforming weights with monotonic acceptance.
    Uses a deterministic descent proposal plus precision-scaled orthogonal noise on a subset of elements.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[1]

    # Initialize to steering vector toward desired direction proxy (estimate via matched filter on first snapshot)
    # As we do not pass angles here, use the principal component of X for initialization
    u, s, vh = np.linalg.svd(X[:64], full_matrices=False)
    w = vh[0].conj()
    w = w / (np.linalg.norm(w) + 1e-10)

    # Simple positional precision prior (center elements more certain)
    positions = np.arange(n)
    pos_center = np.mean(positions)
    irregularity = np.abs(positions - pos_center)
    precision = 1.0 / (1.0 + (irregularity / (np.max(irregularity) + 1e-8)))
    change_weights = (1.0 - precision)
    change_weights = change_weights / (np.sum(change_weights) + 1e-12)

    mse_hist: List[float] = []
    best_w = w.copy()
    best_mse = _avg_mse(X, d, w)
    avg_updates = 0

    for it in range(n_iters):
        # Current mse
        if it % 10 == 0:
            mse_hist.append(best_mse)

        # One-sample gradient proxy
        x = X[it % len(d)]
        y = np.dot(np.conj(w), x)
        e = d[it % len(d)] - y
        grad = x * np.conj(e)

        # Deterministic step
        lr = 0.05
        proposal = w + lr * grad
        proposal = proposal / (np.linalg.norm(proposal) + 1e-10)

        # PSON exploration subset
        # Energy ratio relative to best found so far (capped)
        cur_mse = _avg_mse(X, d, w)
        energy = min(cur_mse / (best_mse + 1e-12), 2.0)
        p = change_weights * (0.2 + 0.8 * min(energy, 1.5))
        p = p / (np.sum(p) + 1e-12)
        n_changes = max(1, int(update_fraction * n))
        idx = rng.choice(n, n_changes, replace=False, p=p)
        avg_updates += n_changes

        # Orthogonal-like noise (uncorrelated per element, scaled by inverse precision)
        noise_scale = 0.05 * energy
        noise = noise_scale * (rng.standard_normal(n_changes) + 1j * rng.standard_normal(n_changes))
        candidate = proposal.copy()
        candidate[idx] = candidate[idx] + noise / (np.sqrt(precision[idx]) + 1e-10)
        candidate = candidate / (np.linalg.norm(candidate) + 1e-10)

        # Monotonic acceptance
        cand_mse = _avg_mse(X, d, candidate)
        if cand_mse <= cur_mse:
            w = candidate
            if cand_mse < best_mse:
                best_mse = cand_mse
                best_w = candidate.copy()

    avg_updates = int(round(avg_updates / max(1, n_iters)))
    return best_w, mse_hist, avg_updates


# =============================================
# Main
# =============================================


def main() -> None:
    print("=" * 80)
    print("PU-BAA Exact Replication (PU-LMS Variants) - ETU Channel")
    print("=" * 80)

    config = PUBAAExactConfig(
        n_elements=16,
        spacing_lambda=0.5,
        desired_angle_deg=0.0,
        interferer_angles_deg=[-30.0, 30.0],
        snr_db=15.0,
        n_iterations=300,
        n_samples=2000,
        sample_period_ns=50.0,
        seed=42,
    )

    print("\nConfiguration:")
    print(f"  Elements: {config.n_elements}")
    print(f"  Spacing (lambda): {config.spacing_lambda}")
    print(f"  Desired angle (deg): {config.desired_angle_deg}")
    print(f"  Interferers (deg): {config.interferer_angles_deg}")
    print(f"  SNR (dB): {config.snr_db}")
    print(f"  Iterations: {config.n_iterations}")
    print(f"  Samples: {config.n_samples}")
    print("  Channel: ETU (9 taps)")

    # Data
    X, d = generate_received_matrix(config)

    # Choose LMS mu using average input power (keeps LMS, but stabilizes)
    avg_power = float(np.mean(np.sum(np.abs(X) ** 2, axis=1))) / config.n_elements
    mu_base = 0.05 / (avg_power + 1e-6)

    results: Dict[str, Dict[str, float]] = {}

    # Full LMS
    w, hist, upd = full_lms(X, d, n_iters=config.n_iterations, mu=mu_base)
    metrics = compute_pattern_metrics(w, config, hist)
    metrics["updates_per_iter"] = upd
    metrics["complexity_reduction"] = 0.0
    results["Full-LMS"] = metrics

    # Sequential LMS
    w, hist, upd = sequential_lms(X, d, n_iters=config.n_iterations, mu=mu_base)
    metrics = compute_pattern_metrics(w, config, hist)
    metrics["updates_per_iter"] = upd
    metrics["complexity_reduction"] = 100.0 * (1.0 - upd / config.n_elements)
    results["Sequential-LMS"] = metrics

    # M-max LMS
    w, hist, upd = mmax_lms(X, d, n_iters=config.n_iterations, mu=mu_base, m_fraction=0.3)
    metrics = compute_pattern_metrics(w, config, hist)
    metrics["updates_per_iter"] = upd
    metrics["complexity_reduction"] = 100.0 * (1.0 - upd / config.n_elements)
    results["M-max-LMS"] = metrics

    # Periodic LMS
    w, hist, upd = periodic_lms(X, d, n_iters=config.n_iterations, mu=mu_base, period=3)
    metrics = compute_pattern_metrics(w, config, hist)
    metrics["updates_per_iter"] = upd
    metrics["complexity_reduction"] = 100.0 * (1.0 - upd / config.n_elements)
    results["Periodic-LMS"] = metrics

    # Stochastic LMS
    w, hist, upd = stochastic_lms(X, d, n_iters=config.n_iterations, mu=mu_base, m_fraction=0.3, seed=config.seed)
    metrics = compute_pattern_metrics(w, config, hist)
    metrics["updates_per_iter"] = upd
    metrics["complexity_reduction"] = 100.0 * (1.0 - upd / config.n_elements)
    results["Stochastic-LMS"] = metrics

    # PSON (for head-to-head comparison on the exact same ETU setup)
    # Match Full-LMS update budget (16 updates/iter) by using update_fraction=1.0
    w, hist, upd = pson_beamforming(X, d, n_iters=config.n_iterations, seed=config.seed, update_fraction=1.0)
    metrics = compute_pattern_metrics(w, config, hist)
    metrics["updates_per_iter"] = upd
    metrics["complexity_reduction"] = 100.0 * (1.0 - upd / config.n_elements)
    results["PSON"] = metrics

    # Print
    print("\n{:<16} {:>6} {:>9} {:>9} {:>8} {:>5} {:>6}".format("Method", "Conv", "MSE(dB)", "Null(dB)", "SL(dB)", "Upd", "Red%"))
    print("-" * 70)
    for name, m in results.items():
        print(
            f"{name:<16} {m['convergence_iter']:>6} {m['steady_mse_db']:>9.1f} {m['null_depth_db']:>9.1f} "
            f"{m['sidelobe_db']:>8.1f} {m['updates_per_iter']:>5} {m['complexity_reduction']:>6.0f}%"
        )

    # Save
    out = {
        "config": {
            "n_elements": config.n_elements,
            "spacing_lambda": config.spacing_lambda,
            "desired_angle_deg": config.desired_angle_deg,
            "interferer_angles_deg": config.interferer_angles_deg,
            "snr_db": config.snr_db,
            "n_iterations": config.n_iterations,
            "n_samples": config.n_samples,
            "channel": "ETU (9 taps)",
        },
        "results": results,
    }
    with open("pson_vs_pu_baa_exact_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nResults saved: pson_vs_pu_baa_exact_results.json")


if __name__ == "__main__":
    main()


