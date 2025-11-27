"""
PSON vs LMS: Fair Comparison with Matched Initialization
=========================================================

This test provides a truly fair comparison between PSON and LMS variants:
- BOTH algorithms start with steering vector initialization (known target direction)
- Tests both static AND dynamic scenarios
- Reports wins/losses honestly

Scenarios:
1. Static beamforming (matched init) - LMS's home turf, but fair start
2. Moving target (matched init) - Operational realism
3. Moving jammer (matched init) - Adversarial conditions

Usage:
    uv run python experiments/discrete_applications/pson_vs_lms_fair_comparison.py
"""

import numpy as np
import json
from typing import Tuple, List, Dict
from dataclasses import dataclass, field
from pathlib import Path


# =============================================================================
# Array Model
# =============================================================================

@dataclass
class FairTestConfig:
    """Configuration for fair comparison test."""
    n_elements: int = 16
    n_phase_levels: int = 32  # 5-bit
    element_spacing: float = 0.5  # wavelengths
    target_angle: float = 0.0  # degrees
    interferer_angles: List[float] = field(default_factory=lambda: [-30.0, 30.0])
    snr_db: float = 15.0
    n_iterations: int = 300
    n_samples: int = 100


def steering_vector(n_elements: int, spacing: float, angle_deg: float) -> np.ndarray:
    """Compute steering vector for a ULA."""
    angle_rad = np.deg2rad(angle_deg)
    k = 2 * np.pi
    n = np.arange(n_elements)
    return np.exp(1j * k * spacing * n * np.sin(angle_rad))


def generate_signal(
    config: FairTestConfig,
    seed: int = 42,
    target_angle_override: float = None,
    jammer_angle_override: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate received signal matrix."""
    rng = np.random.default_rng(seed)
    n = config.n_elements
    n_samples = config.n_samples
    
    target_angle = target_angle_override if target_angle_override is not None else config.target_angle
    
    # Desired signal
    s_desired = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) / np.sqrt(2)
    a_desired = steering_vector(n, config.element_spacing, target_angle)
    
    # Interferer signals
    interferer_power = 10 ** (-config.snr_db / 10)
    X = np.outer(s_desired, a_desired)
    
    interferer_angles = list(config.interferer_angles)
    if jammer_angle_override is not None:
        interferer_angles[0] = jammer_angle_override  # Override first interferer as jammer
    
    for angle in interferer_angles:
        s_int = np.sqrt(interferer_power) * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) / np.sqrt(2)
        a_int = steering_vector(n, config.element_spacing, angle)
        X += np.outer(s_int, a_int)
    
    # Noise
    noise_power = 10 ** (-config.snr_db / 10)
    noise = np.sqrt(noise_power) * (rng.standard_normal((n_samples, n)) + 1j * rng.standard_normal((n_samples, n))) / np.sqrt(2)
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


def compute_sinr(weights: np.ndarray, config: FairTestConfig, jammer_angle: float = None) -> float:
    """Compute Signal-to-Interference-plus-Noise Ratio in dB."""
    # Signal power toward target
    a_target = steering_vector(config.n_elements, config.element_spacing, config.target_angle)
    signal_gain = np.abs(np.dot(weights.conj(), a_target)) ** 2
    
    # Interference power
    interference_power = 0
    jammer_angles = list(config.interferer_angles)
    if jammer_angle is not None:
        jammer_angles[0] = jammer_angle
    
    for angle in jammer_angles:
        a_int = steering_vector(config.n_elements, config.element_spacing, angle)
        interference_power += np.abs(np.dot(weights.conj(), a_int)) ** 2
    
    # Noise power (normalized)
    noise_power = 10 ** (-config.snr_db / 10)
    
    sinr = signal_gain / (interference_power + noise_power + 1e-10)
    return 10 * np.log10(sinr + 1e-10)


# =============================================================================
# Algorithms with MATCHED INITIALIZATION
# =============================================================================

def get_steering_init(config: FairTestConfig) -> np.ndarray:
    """Get steering vector initialization - SAME for all algorithms."""
    sv = steering_vector(config.n_elements, config.element_spacing, config.target_angle)
    return np.angle(sv)


def full_lms_matched(
    X: np.ndarray,
    d: np.ndarray,
    config: FairTestConfig,
    mu: float = 0.01,
) -> Tuple[np.ndarray, List[float]]:
    """Full LMS with steering vector initialization (MATCHED)."""
    n = config.n_elements
    n_levels = config.n_phase_levels
    n_samples = len(d)
    n_iterations = config.n_iterations
    
    # MATCHED: Start with steering vector (same as PSON)
    phases = get_steering_init(config)
    phases = quantize_phases(phases, n_levels)
    weights = np.exp(1j * phases)
    
    mse_curve = []
    
    for iteration in range(n_iterations):
        sample_idx = iteration % n_samples
        x = X[sample_idx]
        
        y = np.dot(weights.conj(), x)
        e = d[sample_idx] - y
        
        grad = e.conj() * x
        phases = phases + mu * np.real(grad * np.exp(-1j * phases))
        phases = quantize_phases(phases, n_levels)
        weights = np.exp(1j * phases)
        
        mse = compute_mse(weights, X, d)
        mse_curve.append(mse)
    
    return weights, mse_curve


def mmax_lms_matched(
    X: np.ndarray,
    d: np.ndarray,
    config: FairTestConfig,
    mu: float = 0.01,
    m_fraction: float = 0.3,
) -> Tuple[np.ndarray, List[float]]:
    """M-max LMS with steering vector initialization (MATCHED)."""
    n = config.n_elements
    n_levels = config.n_phase_levels
    n_samples = len(d)
    n_iterations = config.n_iterations
    m = max(1, int(n * m_fraction))
    
    # MATCHED: Start with steering vector
    phases = get_steering_init(config)
    phases = quantize_phases(phases, n_levels)
    weights = np.exp(1j * phases)
    
    mse_curve = []
    
    for iteration in range(n_iterations):
        sample_idx = iteration % n_samples
        x = X[sample_idx]
        
        y = np.dot(weights.conj(), x)
        e = d[sample_idx] - y
        
        grad = e.conj() * x
        grad_phase = np.real(grad * np.exp(-1j * phases))
        
        top_m_idx = np.argsort(np.abs(grad_phase))[-m:]
        phases[top_m_idx] = phases[top_m_idx] + mu * grad_phase[top_m_idx]
        
        phases = quantize_phases(phases, n_levels)
        weights = np.exp(1j * phases)
        
        mse = compute_mse(weights, X, d)
        mse_curve.append(mse)
    
    return weights, mse_curve


def stochastic_lms_matched(
    X: np.ndarray,
    d: np.ndarray,
    config: FairTestConfig,
    mu: float = 0.01,
    m_fraction: float = 0.3,
    seed: int = 42,
) -> Tuple[np.ndarray, List[float]]:
    """Stochastic LMS with steering vector initialization (MATCHED)."""
    rng = np.random.default_rng(seed)
    n = config.n_elements
    n_levels = config.n_phase_levels
    n_samples = len(d)
    n_iterations = config.n_iterations
    m = max(1, int(n * m_fraction))
    
    # MATCHED: Start with steering vector
    phases = get_steering_init(config)
    phases = quantize_phases(phases, n_levels)
    weights = np.exp(1j * phases)
    
    mse_curve = []
    
    for iteration in range(n_iterations):
        sample_idx = iteration % n_samples
        x = X[sample_idx]
        
        y = np.dot(weights.conj(), x)
        e = d[sample_idx] - y
        
        grad = e.conj() * x
        grad_phase = np.real(grad * np.exp(-1j * phases))
        
        update_idx = rng.choice(n, m, replace=False)
        phases[update_idx] = phases[update_idx] + mu * grad_phase[update_idx]
        
        phases = quantize_phases(phases, n_levels)
        weights = np.exp(1j * phases)
        
        mse = compute_mse(weights, X, d)
        mse_curve.append(mse)
    
    return weights, mse_curve


def pson_matched(
    X: np.ndarray,
    d: np.ndarray,
    config: FairTestConfig,
    seed: int = 42,
) -> Tuple[np.ndarray, List[float]]:
    """PSON with steering vector initialization (MATCHED - same as before)."""
    rng = np.random.default_rng(seed)
    n = config.n_elements
    n_levels = config.n_phase_levels
    phase_levels = np.array([i * 2 * np.pi / n_levels for i in range(n_levels)])
    n_iterations = config.n_iterations
    
    # MATCHED: Start with steering vector (this was always PSON's init)
    phases = get_steering_init(config)
    phases = quantize_phases(phases, n_levels)
    weights = np.exp(1j * phases)
    
    # Precision from element position
    positions = np.arange(n) * config.element_spacing
    pos_center = np.mean(positions)
    irregularity = np.abs(positions - pos_center)
    precision = 1.0 / (1.0 + irregularity / np.max(irregularity))
    weights_pson = (1.0 - precision) / (np.sum(1.0 - precision) + 1e-8)
    
    mse_curve = []
    best_mse = compute_mse(weights, X, d)
    best_phases = phases.copy()
    
    for iteration in range(n_iterations):
        mse_cur = compute_mse(weights, X, d)
        mse_curve.append(mse_cur)
        
        if mse_cur < best_mse:
            best_mse = mse_cur
            best_phases = phases.copy()
        
        energy = min(mse_cur / (best_mse + 1e-10), 2.0)
        
        change_probs = weights_pson * (0.2 + 0.8 * (energy - 1.0 + 1.0))
        change_probs = np.clip(change_probs, 0.01, None)
        change_probs = change_probs / change_probs.sum()
        
        n_changes = max(1, int(n * 0.2 * (0.3 + 0.7 * min(energy, 1.5))))
        change_idx = rng.choice(n, min(n_changes, n), replace=False, p=change_probs)
        
        candidate_phases = phases.copy()
        for idx in change_idx:
            current_level = int(round(phases[idx] * n_levels / (2 * np.pi))) % n_levels
            step = 1 if precision[idx] > 0.5 else rng.choice([1, 2])
            delta = rng.choice([-step, step])
            new_level = (current_level + delta) % n_levels
            candidate_phases[idx] = phase_levels[new_level]
        
        candidate_weights = np.exp(1j * candidate_phases)
        mse_new = compute_mse(candidate_weights, X, d)
        
        # Monotonic descent
        if mse_new <= mse_cur:
            phases = candidate_phases
            weights = candidate_weights
    
    return np.exp(1j * best_phases), mse_curve


def pson_subspace_matched(
    X: np.ndarray,
    d: np.ndarray,
    config: FairTestConfig,
    seed: int = 42,
    k_subspace: int = 8,  # Project to k-dimensional subspace
) -> Tuple[np.ndarray, List[float]]:
    """
    PSON-Subspace with steering vector initialization (MATCHED).
    
    Projects to signal subspace via SVD, optimizes there, then lifts back.
    This is more effective at higher dimensions where standard PSON struggles.
    """
    rng = np.random.default_rng(seed)
    n = config.n_elements
    n_iterations = config.n_iterations
    
    # Find signal subspace via SVD
    U, S, Vh = np.linalg.svd(X[:min(64, len(X))], full_matrices=False)
    k = min(k_subspace, n, len(S))
    V_k = Vh[:k].T  # (n, k) - projection matrix to subspace
    
    # MATCHED: Start with steering vector, project to subspace
    init_phases = get_steering_init(config)
    w_full = np.exp(1j * init_phases)
    w_sub = V_k.T.conj() @ w_full  # Project to subspace (k,)
    
    # Precision in subspace (uniform since we don't have position info)
    precision_sub = np.ones(k) * 0.5
    weights_pson_sub = np.ones(k) / k
    
    mse_curve = []
    best_mse = compute_mse(V_k @ w_sub, X, d)
    best_w_sub = w_sub.copy()
    
    for iteration in range(n_iterations):
        w_full_cur = V_k @ w_sub
        w_full_cur = w_full_cur / (np.linalg.norm(w_full_cur) + 1e-10)
        mse_cur = compute_mse(w_full_cur, X, d)
        mse_curve.append(mse_cur)
        
        if mse_cur < best_mse:
            best_mse = mse_cur
            best_w_sub = w_sub.copy()
        
        energy = min(mse_cur / (best_mse + 1e-10), 2.0)
        
        # Gradient in subspace
        sample_idx = iteration % len(d)
        x_sub = V_k.T.conj() @ X[sample_idx]  # Project input to subspace
        y = np.dot(w_sub.conj(), x_sub)
        e = d[sample_idx] - y
        grad_sub = e.conj() * x_sub
        
        # Deterministic step in subspace
        lr = 0.05
        proposal_sub = w_sub + lr * grad_sub
        
        # PSON exploration in subspace (much lower dimensional!)
        noise_scale = 0.1 * energy
        n_changes = max(1, int(k * 0.3))
        change_idx = rng.choice(k, n_changes, replace=False)
        
        noise = noise_scale * (rng.standard_normal(n_changes) + 1j * rng.standard_normal(n_changes))
        candidate_sub = proposal_sub.copy()
        candidate_sub[change_idx] = candidate_sub[change_idx] + noise / (np.sqrt(precision_sub[change_idx]) + 1e-10)
        
        # Lift back to full space and evaluate
        candidate_full = V_k @ candidate_sub
        candidate_full = candidate_full / (np.linalg.norm(candidate_full) + 1e-10)
        mse_new = compute_mse(candidate_full, X, d)
        
        # Monotonic descent
        if mse_new <= mse_cur:
            w_sub = candidate_sub
    
    # Return best found, lifted to full space
    w_best_full = V_k @ best_w_sub
    w_best_full = w_best_full / (np.linalg.norm(w_best_full) + 1e-10)
    return w_best_full, mse_curve


# =============================================================================
# Test Scenarios
# =============================================================================

def run_static_test(config: FairTestConfig, seed: int) -> Dict:
    """Static scenario with matched initialization."""
    X, d = generate_signal(config, seed=seed)
    
    results = {}
    
    # All algorithms with MATCHED initialization
    w, mse = full_lms_matched(X, d, config)
    results["Full-LMS"] = {"final_mse": mse[-1], "sinr": compute_sinr(w, config)}
    
    w, mse = mmax_lms_matched(X, d, config)
    results["M-max-LMS"] = {"final_mse": mse[-1], "sinr": compute_sinr(w, config)}
    
    w, mse = stochastic_lms_matched(X, d, config, seed=seed)
    results["Stochastic-LMS"] = {"final_mse": mse[-1], "sinr": compute_sinr(w, config)}
    
    w, mse = pson_matched(X, d, config, seed=seed)
    results["PSON"] = {"final_mse": mse[-1], "sinr": compute_sinr(w, config)}
    
    w, mse = pson_subspace_matched(X, d, config, seed=seed)
    results["PSON-Sub"] = {"final_mse": mse[-1], "sinr": compute_sinr(w, config)}
    
    return results


def run_moving_target_test(config: FairTestConfig, seed: int, target_velocity: float = 0.5) -> Dict:
    """Moving target scenario - target moves during optimization."""
    rng = np.random.default_rng(seed)
    n = config.n_elements
    n_levels = config.n_phase_levels
    n_iterations = config.n_iterations
    
    results = {alg: {"tracking_errors": []} for alg in ["Full-LMS", "M-max-LMS", "Stochastic-LMS", "PSON", "PSON-Sub"]}
    
    # Initialize all with steering vector toward INITIAL target
    init_phases = get_steering_init(config)
    
    # For PSON-Sub, we need signal subspace - use initial data
    X_init, d_init = generate_signal(config, seed=seed)
    U, S, Vh = np.linalg.svd(X_init[:min(64, len(X_init))], full_matrices=False)
    k_sub = min(8, n)
    V_k = Vh[:k_sub].T
    w_init_full = np.exp(1j * init_phases)
    w_sub_init = V_k.T.conj() @ w_init_full
    
    algorithms = {
        "Full-LMS": {"phases": quantize_phases(init_phases.copy(), n_levels), "mu": 0.01},
        "M-max-LMS": {"phases": quantize_phases(init_phases.copy(), n_levels), "mu": 0.01},
        "Stochastic-LMS": {"phases": quantize_phases(init_phases.copy(), n_levels), "mu": 0.01},
        "PSON": {"phases": quantize_phases(init_phases.copy(), n_levels), "best_mse": float('inf')},
        "PSON-Sub": {"w_sub": w_sub_init.copy(), "V_k": V_k, "best_mse": float('inf')},
    }
    
    # PSON precision setup
    positions = np.arange(n) * config.element_spacing
    pos_center = np.mean(positions)
    irregularity = np.abs(positions - pos_center)
    precision = 1.0 / (1.0 + irregularity / np.max(irregularity))
    weights_pson = (1.0 - precision) / (np.sum(1.0 - precision) + 1e-8)
    phase_levels = np.array([i * 2 * np.pi / n_levels for i in range(n_levels)])
    
    current_target = config.target_angle
    
    for iteration in range(n_iterations):
        # Target moves
        current_target += target_velocity
        
        # Generate signal with current target position
        X, d = generate_signal(config, seed=seed + iteration, target_angle_override=current_target)
        
        for alg_name, alg_state in algorithms.items():
            # Handle PSON-Sub separately (uses w_sub not phases)
            if alg_name == "PSON-Sub":
                w_sub = alg_state["w_sub"]
                V_k_local = alg_state["V_k"]
                k_local = len(w_sub)
                
                w_full = V_k_local @ w_sub
                w_full = w_full / (np.linalg.norm(w_full) + 1e-10)
                mse_cur = compute_mse(w_full, X, d)
                
                if mse_cur < alg_state["best_mse"]:
                    alg_state["best_mse"] = mse_cur
                
                energy = min(mse_cur / (alg_state["best_mse"] + 1e-10), 2.0)
                
                sample_idx = iteration % len(d)
                x_sub = V_k_local.T.conj() @ X[sample_idx]
                y = np.dot(w_sub.conj(), x_sub)
                e = d[sample_idx] - y
                grad_sub = e.conj() * x_sub
                
                lr = 0.05
                proposal_sub = w_sub + lr * grad_sub
                
                noise_scale = 0.1 * energy
                n_changes_sub = max(1, int(k_local * 0.3))
                change_idx_sub = rng.choice(k_local, n_changes_sub, replace=False)
                noise = noise_scale * (rng.standard_normal(n_changes_sub) + 1j * rng.standard_normal(n_changes_sub))
                candidate_sub = proposal_sub.copy()
                candidate_sub[change_idx_sub] = candidate_sub[change_idx_sub] + noise
                
                candidate_full = V_k_local @ candidate_sub
                candidate_full = candidate_full / (np.linalg.norm(candidate_full) + 1e-10)
                mse_new = compute_mse(candidate_full, X, d)
                
                if mse_new <= mse_cur:
                    alg_state["w_sub"] = candidate_sub
                
                weights = V_k_local @ alg_state["w_sub"]
                weights = weights / (np.linalg.norm(weights) + 1e-10)
                
                # Compute tracking error for PSON-Sub
                test_angles = np.linspace(-90, 90, 361)
                pattern = np.array([np.abs(np.dot(weights.conj(), steering_vector(n, config.element_spacing, ang)))**2 for ang in test_angles])
                beam_direction = test_angles[np.argmax(pattern)]
                tracking_error = beam_direction - current_target
                results[alg_name]["tracking_errors"].append(tracking_error)
                continue
            
            phases = alg_state["phases"]
            weights = np.exp(1j * phases)
            
            if alg_name == "Full-LMS":
                sample_idx = iteration % len(d)
                x = X[sample_idx]
                y = np.dot(weights.conj(), x)
                e = d[sample_idx] - y
                grad = e.conj() * x
                phases = phases + alg_state["mu"] * np.real(grad * np.exp(-1j * phases))
                
            elif alg_name == "M-max-LMS":
                sample_idx = iteration % len(d)
                x = X[sample_idx]
                y = np.dot(weights.conj(), x)
                e = d[sample_idx] - y
                grad = e.conj() * x
                grad_phase = np.real(grad * np.exp(-1j * phases))
                m = max(1, int(n * 0.3))
                top_m_idx = np.argsort(np.abs(grad_phase))[-m:]
                phases[top_m_idx] = phases[top_m_idx] + alg_state["mu"] * grad_phase[top_m_idx]
                
            elif alg_name == "Stochastic-LMS":
                sample_idx = iteration % len(d)
                x = X[sample_idx]
                y = np.dot(weights.conj(), x)
                e = d[sample_idx] - y
                grad = e.conj() * x
                grad_phase = np.real(grad * np.exp(-1j * phases))
                m = max(1, int(n * 0.3))
                update_idx = rng.choice(n, m, replace=False)
                phases[update_idx] = phases[update_idx] + alg_state["mu"] * grad_phase[update_idx]
                
            elif alg_name == "PSON":
                mse_cur = compute_mse(weights, X, d)
                if mse_cur < alg_state["best_mse"]:
                    alg_state["best_mse"] = mse_cur
                
                energy = min(mse_cur / (alg_state["best_mse"] + 1e-10), 2.0)
                change_probs = weights_pson * (0.2 + 0.8 * min(energy, 1.5))
                change_probs = change_probs / change_probs.sum()
                n_changes = max(1, int(n * 0.2))
                change_idx = rng.choice(n, n_changes, replace=False, p=change_probs)
                
                candidate_phases = phases.copy()
                for idx in change_idx:
                    current_level = int(round(phases[idx] * n_levels / (2 * np.pi))) % n_levels
                    delta = rng.choice([-1, 1])
                    new_level = (current_level + delta) % n_levels
                    candidate_phases[idx] = phase_levels[new_level]
                
                candidate_weights = np.exp(1j * candidate_phases)
                mse_new = compute_mse(candidate_weights, X, d)
                
                if mse_new <= mse_cur:
                    phases = candidate_phases
                
                phases = quantize_phases(phases, n_levels)
                alg_state["phases"] = phases
                weights = np.exp(1j * phases)
            
            elif alg_name == "PSON-Sub":
                # PSON-Subspace operates in reduced dimensional space
                w_sub = alg_state["w_sub"]
                V_k_local = alg_state["V_k"]
                k_local = len(w_sub)
                
                w_full = V_k_local @ w_sub
                w_full = w_full / (np.linalg.norm(w_full) + 1e-10)
                mse_cur = compute_mse(w_full, X, d)
                
                if mse_cur < alg_state["best_mse"]:
                    alg_state["best_mse"] = mse_cur
                
                energy = min(mse_cur / (alg_state["best_mse"] + 1e-10), 2.0)
                
                # Gradient in subspace
                sample_idx = iteration % len(d)
                x_sub = V_k_local.T.conj() @ X[sample_idx]
                y = np.dot(w_sub.conj(), x_sub)
                e = d[sample_idx] - y
                grad_sub = e.conj() * x_sub
                
                lr = 0.05
                proposal_sub = w_sub + lr * grad_sub
                
                # PSON exploration in subspace
                noise_scale = 0.1 * energy
                n_changes = max(1, int(k_local * 0.3))
                change_idx = rng.choice(k_local, n_changes, replace=False)
                noise = noise_scale * (rng.standard_normal(n_changes) + 1j * rng.standard_normal(n_changes))
                candidate_sub = proposal_sub.copy()
                candidate_sub[change_idx] = candidate_sub[change_idx] + noise
                
                candidate_full = V_k_local @ candidate_sub
                candidate_full = candidate_full / (np.linalg.norm(candidate_full) + 1e-10)
                mse_new = compute_mse(candidate_full, X, d)
                
                if mse_new <= mse_cur:
                    alg_state["w_sub"] = candidate_sub
                
                weights = V_k_local @ alg_state["w_sub"]
                weights = weights / (np.linalg.norm(weights) + 1e-10)
            
            else:
                phases = quantize_phases(phases, n_levels)
                alg_state["phases"] = phases
                weights = np.exp(1j * phases)
            
            # Compute tracking error (beam pointing vs actual target)
            # Find peak of beam pattern
            test_angles = np.linspace(-90, 90, 361)
            pattern = np.array([np.abs(np.dot(weights.conj(), steering_vector(n, config.element_spacing, ang)))**2 for ang in test_angles])
            beam_direction = test_angles[np.argmax(pattern)]
            tracking_error = beam_direction - current_target
            results[alg_name]["tracking_errors"].append(tracking_error)
    
    # Compute mean absolute tracking error
    for alg_name in results:
        errors = np.array(results[alg_name]["tracking_errors"])
        results[alg_name]["mean_abs_error"] = float(np.mean(np.abs(errors)))
        results[alg_name]["final_error"] = float(errors[-1])
    
    return results


def run_moving_jammer_test(config: FairTestConfig, seed: int, jammer_move_interval: int = 50) -> Dict:
    """Moving jammer scenario - jammer repositions periodically."""
    rng = np.random.default_rng(seed)
    n = config.n_elements
    n_levels = config.n_phase_levels
    n_iterations = config.n_iterations
    
    results = {alg: {"sinr_history": []} for alg in ["Full-LMS", "M-max-LMS", "Stochastic-LMS", "PSON", "PSON-Sub"]}
    
    # Initialize all with steering vector
    init_phases = get_steering_init(config)
    
    # For PSON-Sub
    X_init, d_init = generate_signal(config, seed=seed)
    U, S, Vh = np.linalg.svd(X_init[:min(64, len(X_init))], full_matrices=False)
    k_sub = min(8, n)
    V_k = Vh[:k_sub].T
    w_init_full = np.exp(1j * init_phases)
    w_sub_init = V_k.T.conj() @ w_init_full
    
    algorithms = {
        "Full-LMS": {"phases": quantize_phases(init_phases.copy(), n_levels), "mu": 0.01},
        "M-max-LMS": {"phases": quantize_phases(init_phases.copy(), n_levels), "mu": 0.01},
        "Stochastic-LMS": {"phases": quantize_phases(init_phases.copy(), n_levels), "mu": 0.01},
        "PSON": {"phases": quantize_phases(init_phases.copy(), n_levels), "best_mse": float('inf')},
        "PSON-Sub": {"w_sub": w_sub_init.copy(), "V_k": V_k, "best_mse": float('inf')},
    }
    
    # PSON setup
    positions = np.arange(n) * config.element_spacing
    pos_center = np.mean(positions)
    irregularity = np.abs(positions - pos_center)
    precision = 1.0 / (1.0 + irregularity / np.max(irregularity))
    weights_pson = (1.0 - precision) / (np.sum(1.0 - precision) + 1e-8)
    phase_levels = np.array([i * 2 * np.pi / n_levels for i in range(n_levels)])
    
    jammer_angle = config.interferer_angles[0]
    
    for iteration in range(n_iterations):
        # Jammer moves periodically
        if iteration > 0 and iteration % jammer_move_interval == 0:
            jammer_angle = rng.uniform(-60, 60)
        
        X, d = generate_signal(config, seed=seed + iteration, jammer_angle_override=jammer_angle)
        
        for alg_name, alg_state in algorithms.items():
            # Handle PSON-Sub separately
            if alg_name == "PSON-Sub":
                w_sub = alg_state["w_sub"]
                V_k_local = alg_state["V_k"]
                k_local = len(w_sub)
                
                w_full = V_k_local @ w_sub
                w_full = w_full / (np.linalg.norm(w_full) + 1e-10)
                mse_cur = compute_mse(w_full, X, d)
                
                if mse_cur < alg_state["best_mse"]:
                    alg_state["best_mse"] = mse_cur
                
                energy = min(mse_cur / (alg_state["best_mse"] + 1e-10), 2.0)
                
                sample_idx = iteration % len(d)
                x_sub = V_k_local.T.conj() @ X[sample_idx]
                y = np.dot(w_sub.conj(), x_sub)
                e = d[sample_idx] - y
                grad_sub = e.conj() * x_sub
                
                lr = 0.05
                proposal_sub = w_sub + lr * grad_sub
                
                noise_scale = 0.1 * energy
                n_changes_sub = max(1, int(k_local * 0.3))
                change_idx_sub = rng.choice(k_local, n_changes_sub, replace=False)
                noise = noise_scale * (rng.standard_normal(n_changes_sub) + 1j * rng.standard_normal(n_changes_sub))
                candidate_sub = proposal_sub.copy()
                candidate_sub[change_idx_sub] = candidate_sub[change_idx_sub] + noise
                
                candidate_full = V_k_local @ candidate_sub
                candidate_full = candidate_full / (np.linalg.norm(candidate_full) + 1e-10)
                mse_new = compute_mse(candidate_full, X, d)
                
                if mse_new <= mse_cur:
                    alg_state["w_sub"] = candidate_sub
                
                weights = V_k_local @ alg_state["w_sub"]
                weights = weights / (np.linalg.norm(weights) + 1e-10)
                
                # Compute SINR for PSON-Sub
                sinr = compute_sinr(weights, config, jammer_angle=jammer_angle)
                results[alg_name]["sinr_history"].append(sinr)
                continue
            
            phases = alg_state["phases"]
            weights = np.exp(1j * phases)
            
            if alg_name == "Full-LMS":
                sample_idx = iteration % len(d)
                x = X[sample_idx]
                y = np.dot(weights.conj(), x)
                e = d[sample_idx] - y
                grad = e.conj() * x
                phases = phases + alg_state["mu"] * np.real(grad * np.exp(-1j * phases))
                
            elif alg_name == "M-max-LMS":
                sample_idx = iteration % len(d)
                x = X[sample_idx]
                y = np.dot(weights.conj(), x)
                e = d[sample_idx] - y
                grad = e.conj() * x
                grad_phase = np.real(grad * np.exp(-1j * phases))
                m = max(1, int(n * 0.3))
                top_m_idx = np.argsort(np.abs(grad_phase))[-m:]
                phases[top_m_idx] = phases[top_m_idx] + alg_state["mu"] * grad_phase[top_m_idx]
                
            elif alg_name == "Stochastic-LMS":
                sample_idx = iteration % len(d)
                x = X[sample_idx]
                y = np.dot(weights.conj(), x)
                e = d[sample_idx] - y
                grad = e.conj() * x
                grad_phase = np.real(grad * np.exp(-1j * phases))
                m = max(1, int(n * 0.3))
                update_idx = rng.choice(n, m, replace=False)
                phases[update_idx] = phases[update_idx] + alg_state["mu"] * grad_phase[update_idx]
                
            elif alg_name == "PSON":
                mse_cur = compute_mse(weights, X, d)
                if mse_cur < alg_state["best_mse"]:
                    alg_state["best_mse"] = mse_cur
                
                energy = min(mse_cur / (alg_state["best_mse"] + 1e-10), 2.0)
                change_probs = weights_pson * (0.2 + 0.8 * min(energy, 1.5))
                change_probs = change_probs / change_probs.sum()
                n_changes = max(1, int(n * 0.2))
                change_idx = rng.choice(n, n_changes, replace=False, p=change_probs)
                
                candidate_phases = phases.copy()
                for idx in change_idx:
                    current_level = int(round(phases[idx] * n_levels / (2 * np.pi))) % n_levels
                    delta = rng.choice([-1, 1])
                    new_level = (current_level + delta) % n_levels
                    candidate_phases[idx] = phase_levels[new_level]
                
                candidate_weights = np.exp(1j * candidate_phases)
                mse_new = compute_mse(candidate_weights, X, d)
                
                if mse_new <= mse_cur:
                    phases = candidate_phases
                
                phases = quantize_phases(phases, n_levels)
                alg_state["phases"] = phases
                weights = np.exp(1j * phases)
            
            elif alg_name == "PSON-Sub":
                # PSON-Subspace
                w_sub = alg_state["w_sub"]
                V_k_local = alg_state["V_k"]
                k_local = len(w_sub)
                
                w_full = V_k_local @ w_sub
                w_full = w_full / (np.linalg.norm(w_full) + 1e-10)
                mse_cur = compute_mse(w_full, X, d)
                
                if mse_cur < alg_state["best_mse"]:
                    alg_state["best_mse"] = mse_cur
                
                energy = min(mse_cur / (alg_state["best_mse"] + 1e-10), 2.0)
                
                sample_idx = iteration % len(d)
                x_sub = V_k_local.T.conj() @ X[sample_idx]
                y = np.dot(w_sub.conj(), x_sub)
                e = d[sample_idx] - y
                grad_sub = e.conj() * x_sub
                
                lr = 0.05
                proposal_sub = w_sub + lr * grad_sub
                
                noise_scale = 0.1 * energy
                n_changes = max(1, int(k_local * 0.3))
                change_idx = rng.choice(k_local, n_changes, replace=False)
                noise = noise_scale * (rng.standard_normal(n_changes) + 1j * rng.standard_normal(n_changes))
                candidate_sub = proposal_sub.copy()
                candidate_sub[change_idx] = candidate_sub[change_idx] + noise
                
                candidate_full = V_k_local @ candidate_sub
                candidate_full = candidate_full / (np.linalg.norm(candidate_full) + 1e-10)
                mse_new = compute_mse(candidate_full, X, d)
                
                if mse_new <= mse_cur:
                    alg_state["w_sub"] = candidate_sub
                
                weights = V_k_local @ alg_state["w_sub"]
                weights = weights / (np.linalg.norm(weights) + 1e-10)
            
            else:
                phases = quantize_phases(phases, n_levels)
                alg_state["phases"] = phases
                weights = np.exp(1j * phases)
            
            # Compute SINR
            sinr = compute_sinr(weights, config, jammer_angle=jammer_angle)
            results[alg_name]["sinr_history"].append(sinr)
    
    # Compute statistics
    for alg_name in results:
        sinr_hist = np.array(results[alg_name]["sinr_history"])
        results[alg_name]["mean_sinr"] = float(np.mean(sinr_hist))
        results[alg_name]["min_sinr"] = float(np.min(sinr_hist))
        results[alg_name]["final_sinr"] = float(sinr_hist[-1])
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("PSON vs LMS: FAIR COMPARISON (Matched Initialization)")
    print("=" * 80)
    print("\nBoth PSON and LMS variants start with steering vector initialization.")
    print("This is an apples-to-apples comparison.\n")
    
    config = FairTestConfig()
    seeds = [42, 123, 456]
    
    all_results = {
        "static": [],
        "moving_target": [],
        "moving_jammer": [],
    }
    
    # =========================================================================
    # Test 1: Static Scenario (Matched Init)
    # =========================================================================
    print("=" * 70)
    print("TEST 1: STATIC BEAMFORMING (Matched Initialization)")
    print("=" * 70)
    
    static_wins = {"Full-LMS": 0, "M-max-LMS": 0, "Stochastic-LMS": 0, "PSON": 0, "PSON-Sub": 0}
    
    for seed in seeds:
        results = run_static_test(config, seed)
        all_results["static"].append({"seed": seed, "results": results})
        
        # Find MSE winner
        mse_winner = min(results.items(), key=lambda x: x[1]["final_mse"])[0]
        static_wins[mse_winner] += 1
        
        print(f"\nSeed {seed}:")
        print(f"  {'Algorithm':<18} {'MSE':>10} {'SINR (dB)':>12}")
        print(f"  {'-'*42}")
        for name, res in sorted(results.items(), key=lambda x: x[1]["final_mse"]):
            marker = "* " if name == mse_winner else "  "
            print(f"{marker}{name:<18} {res['final_mse']:>10.4f} {res['sinr']:>12.1f}")
    
    print(f"\nSTATIC RESULTS (Matched Init):")
    for name in sorted(static_wins.keys(), key=lambda x: -static_wins[x]):
        print(f"  {name:<18}: {static_wins[name]}/{len(seeds)} wins")
    
    # =========================================================================
    # Test 2: Moving Target (Matched Init)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: MOVING TARGET (Matched Initialization)")
    print("Target moves 0.5 deg/iteration during optimization")
    print("=" * 70)
    
    target_wins = {"Full-LMS": 0, "M-max-LMS": 0, "Stochastic-LMS": 0, "PSON": 0, "PSON-Sub": 0}
    
    for seed in seeds:
        results = run_moving_target_test(config, seed, target_velocity=0.5)
        all_results["moving_target"].append({"seed": seed, "results": results})
        
        # Winner = lowest mean tracking error
        winner = min(results.items(), key=lambda x: x[1]["mean_abs_error"])[0]
        target_wins[winner] += 1
        
        print(f"\nSeed {seed}:")
        print(f"  {'Algorithm':<18} {'Mean Error':>12} {'Final Error':>12}")
        print(f"  {'-'*44}")
        for name, res in sorted(results.items(), key=lambda x: x[1]["mean_abs_error"]):
            marker = "* " if name == winner else "  "
            print(f"{marker}{name:<18} {res['mean_abs_error']:>12.2f}° {res['final_error']:>12.2f}°")
    
    print(f"\nMOVING TARGET RESULTS (Matched Init):")
    for name in sorted(target_wins.keys(), key=lambda x: -target_wins[x]):
        print(f"  {name:<18}: {target_wins[name]}/{len(seeds)} wins")
    
    # =========================================================================
    # Test 3: Moving Jammer (Matched Init)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: MOVING JAMMER (Matched Initialization)")
    print("Strong jammer (+20dB) repositions every 50 iterations")
    print("=" * 70)
    
    jammer_wins = {"Full-LMS": 0, "M-max-LMS": 0, "Stochastic-LMS": 0, "PSON": 0, "PSON-Sub": 0}
    
    for seed in seeds:
        results = run_moving_jammer_test(config, seed, jammer_move_interval=50)
        all_results["moving_jammer"].append({"seed": seed, "results": results})
        
        # Winner = highest mean SINR
        winner = max(results.items(), key=lambda x: x[1]["mean_sinr"])[0]
        jammer_wins[winner] += 1
        
        print(f"\nSeed {seed}:")
        print(f"  {'Algorithm':<18} {'Mean SINR':>12} {'Min SINR':>12} {'Final SINR':>12}")
        print(f"  {'-'*56}")
        for name, res in sorted(results.items(), key=lambda x: -x[1]["mean_sinr"]):
            marker = "* " if name == winner else "  "
            print(f"{marker}{name:<18} {res['mean_sinr']:>12.1f} {res['min_sinr']:>12.1f} {res['final_sinr']:>12.1f}")
    
    print(f"\nMOVING JAMMER RESULTS (Matched Init):")
    for name in sorted(jammer_wins.keys(), key=lambda x: -jammer_wins[x]):
        print(f"  {name:<18}: {jammer_wins[name]}/{len(seeds)} wins")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY (Fair Comparison - Matched Initialization)")
    print("=" * 80)
    
    total_wins = {alg: 0 for alg in ["Full-LMS", "M-max-LMS", "Stochastic-LMS", "PSON", "PSON-Sub"]}
    for alg in total_wins:
        total_wins[alg] = static_wins[alg] + target_wins[alg] + jammer_wins[alg]
    
    print(f"\n{'Scenario':<25} {'Full-LMS':>10} {'M-max':>10} {'Stochastic':>10} {'PSON':>10}")
    print("-" * 70)
    print(f"{'Static (MSE)':<25} {static_wins['Full-LMS']:>10} {static_wins['M-max-LMS']:>10} {static_wins['Stochastic-LMS']:>10} {static_wins['PSON']:>10}")
    print(f"{'Moving Target':<25} {target_wins['Full-LMS']:>10} {target_wins['M-max-LMS']:>10} {target_wins['Stochastic-LMS']:>10} {target_wins['PSON']:>10}")
    print(f"{'Moving Jammer':<25} {jammer_wins['Full-LMS']:>10} {jammer_wins['M-max-LMS']:>10} {jammer_wins['Stochastic-LMS']:>10} {jammer_wins['PSON']:>10}")
    print("-" * 70)
    print(f"{'TOTAL':<25} {total_wins['Full-LMS']:>10} {total_wins['M-max-LMS']:>10} {total_wins['Stochastic-LMS']:>10} {total_wins['PSON']:>10}")
    
    total_tests = len(seeds) * 3
    print(f"\nWin Rates:")
    for alg in sorted(total_wins.keys(), key=lambda x: -total_wins[x]):
        pct = 100 * total_wins[alg] / total_tests
        bar = "*" * int(pct / 5)
        print(f"  {alg:<18}: {total_wins[alg]}/{total_tests} ({pct:5.1f}%) {bar}")
    
    # Save results
    output_dir = Path("results/fair_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "description": "Fair comparison with matched initialization (steering vector for all)",
        "config": {
            "n_elements": config.n_elements,
            "n_phase_levels": config.n_phase_levels,
            "target_angle": config.target_angle,
            "interferer_angles": config.interferer_angles,
            "snr_db": config.snr_db,
            "n_iterations": config.n_iterations,
        },
        "seeds": seeds,
        "wins": {
            "static": static_wins,
            "moving_target": target_wins,
            "moving_jammer": jammer_wins,
            "total": total_wins,
        },
        "detailed_results": all_results,
    }
    
    with open(output_dir / "pson_vs_lms_fair_comparison_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    print(f"\nResults saved: {output_dir / 'pson_vs_lms_fair_comparison_results.json'}")


if __name__ == "__main__":
    main()

