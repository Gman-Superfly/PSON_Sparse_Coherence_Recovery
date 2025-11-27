"""
PSON vs PU-LMS: Dynamic Scenarios Where Speed Matters
======================================================

Tests where PSON's design should provide advantage:
1. Rapidly Moving Targets (radar tracking) - target angle changes during optimization
2. Massive MIMO (1000+ elements) - high-dimensional adaptation
3. Real-time Adaptive Nulling (jammers) - interferer positions change unpredictably

Usage:
    uv run python experiments/discrete_applications/pson_dynamic_scenarios_test.py
"""

import numpy as np
import json
from typing import Tuple, List, Dict
from dataclasses import dataclass, field


# =============================================================================
# Scenario 1: Rapidly Moving Target (Radar Tracking)
# =============================================================================

@dataclass
class MovingTargetConfig:
    """Target moves during optimization - tests tracking ability."""
    n_elements: int = 16
    spacing_lambda: float = 0.5
    initial_target_deg: float = 0.0
    target_velocity_deg_per_iter: float = 0.5  # Target moves 0.5 deg/iteration
    snr_db: float = 15.0
    n_iterations: int = 200
    seed: int = 42


def steering_vector(n: int, spacing: float, angle_deg: float) -> np.ndarray:
    """ULA steering vector."""
    angle_rad = np.deg2rad(angle_deg)
    k = 2 * np.pi
    idx = np.arange(n)
    return np.exp(1j * k * spacing * idx * np.sin(angle_rad))


def run_moving_target_test(config: MovingTargetConfig) -> Dict:
    """Test tracking a moving target."""
    rng = np.random.default_rng(config.seed)
    n = config.n_elements
    
    # Initialize weights
    w_lms = steering_vector(n, config.spacing_lambda, config.initial_target_deg)
    w_lms = w_lms / np.linalg.norm(w_lms)
    
    w_mmax = w_lms.copy()
    w_pson = w_lms.copy()
    
    # Track cumulative error (lower is better tracking)
    lms_errors = []
    mmax_errors = []
    pson_errors = []
    
    mu = 0.1
    best_pson_w = w_pson.copy()
    
    for it in range(config.n_iterations):
        # Target moves!
        current_target = config.initial_target_deg + it * config.target_velocity_deg_per_iter
        
        # Generate signal from current target position
        a_target = steering_vector(n, config.spacing_lambda, current_target)
        s = (rng.choice([-1, 1]) + 1j * rng.choice([-1, 1])) / np.sqrt(2)
        
        # Add noise
        noise_power = 10 ** (-config.snr_db / 10)
        noise = np.sqrt(noise_power / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        x = s * a_target + noise
        d = s  # Desired is the original signal
        
        # --- Full LMS ---
        y_lms = np.dot(w_lms.conj(), x)
        e_lms = d - y_lms
        w_lms = w_lms + mu * e_lms.conj() * x
        w_lms = w_lms / (np.linalg.norm(w_lms) + 1e-10)
        
        # --- M-max LMS (update top 4 elements) ---
        y_mmax = np.dot(w_mmax.conj(), x)
        e_mmax = d - y_mmax
        grad = e_mmax.conj() * x
        top_idx = np.argsort(np.abs(grad))[-4:]
        w_mmax[top_idx] = w_mmax[top_idx] + mu * grad[top_idx]
        w_mmax = w_mmax / (np.linalg.norm(w_mmax) + 1e-10)
        
        # --- PSON ---
        y_pson = np.dot(w_pson.conj(), x)
        e_pson = d - y_pson
        grad_pson = e_pson.conj() * x
        
        # Deterministic step
        proposal = w_pson + mu * grad_pson
        proposal = proposal / (np.linalg.norm(proposal) + 1e-10)
        
        # PSON exploration (all elements since we're matching budget)
        noise_scale = 0.05
        pson_noise = noise_scale * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        candidate = proposal + pson_noise
        candidate = candidate / (np.linalg.norm(candidate) + 1e-10)
        
        # Monotonic acceptance based on instantaneous error
        if np.abs(d - np.dot(candidate.conj(), x)) <= np.abs(e_pson):
            w_pson = candidate
        
        # Track beam gain toward CURRENT target position
        gain_lms = np.abs(np.dot(w_lms.conj(), a_target)) ** 2
        gain_mmax = np.abs(np.dot(w_mmax.conj(), a_target)) ** 2
        gain_pson = np.abs(np.dot(w_pson.conj(), a_target)) ** 2
        
        lms_errors.append(1.0 - gain_lms)
        mmax_errors.append(1.0 - gain_mmax)
        pson_errors.append(1.0 - gain_pson)
    
    return {
        "scenario": "Moving Target (Radar)",
        "target_velocity_deg_per_iter": config.target_velocity_deg_per_iter,
        "total_target_movement_deg": config.n_iterations * config.target_velocity_deg_per_iter,
        "Full-LMS": {
            "mean_tracking_error": float(np.mean(lms_errors)),
            "final_tracking_error": float(lms_errors[-1]),
        },
        "M-max-LMS": {
            "mean_tracking_error": float(np.mean(mmax_errors)),
            "final_tracking_error": float(mmax_errors[-1]),
        },
        "PSON": {
            "mean_tracking_error": float(np.mean(pson_errors)),
            "final_tracking_error": float(pson_errors[-1]),
        },
        "winner_mean": min(["Full-LMS", "M-max-LMS", "PSON"], 
                          key=lambda m: {"Full-LMS": np.mean(lms_errors), 
                                        "M-max-LMS": np.mean(mmax_errors),
                                        "PSON": np.mean(pson_errors)}[m]),
    }


# =============================================================================
# Scenario 2: Massive MIMO (1000+ elements)
# =============================================================================

@dataclass
class MassiveMIMOConfig:
    """High-dimensional array - tests scalability."""
    n_elements: int = 1024
    spacing_lambda: float = 0.5
    desired_angle_deg: float = 15.0
    interferer_angles_deg: List[float] = field(default_factory=lambda: [-20.0, 35.0, -45.0])
    snr_db: float = 10.0
    n_iterations: int = 500
    seed: int = 42


def run_massive_mimo_test(config: MassiveMIMOConfig) -> Dict:
    """Test on large element arrays (256 to 8192+)."""
    rng = np.random.default_rng(config.seed)
    n = config.n_elements
    
    print(f"  Running Massive MIMO ({n} elements)...")
    
    # Generate received signal matrix
    # Scale samples with array size for proper subspace estimation
    n_samples = max(500, min(n // 2, 2000))
    a_desired = steering_vector(n, config.spacing_lambda, config.desired_angle_deg)
    
    # Desired signal
    s_desired = (rng.choice([-1, 1], n_samples) + 1j * rng.choice([-1, 1], n_samples)) / np.sqrt(2)
    X = np.outer(s_desired, a_desired)
    
    # Interferers
    for angle in config.interferer_angles_deg:
        s_int = (rng.choice([-1, 1], n_samples) + 1j * rng.choice([-1, 1], n_samples)) / np.sqrt(2)
        a_int = steering_vector(n, config.spacing_lambda, angle)
        X += 0.5 * np.outer(s_int, a_int)
    
    # Noise
    noise_power = 10 ** (-config.snr_db / 10)
    X += np.sqrt(noise_power / 2) * (rng.standard_normal((n_samples, n)) + 1j * rng.standard_normal((n_samples, n)))
    d = s_desired
    
    # Initialize - use SVD-based initialization for better start
    w_init = steering_vector(n, config.spacing_lambda, config.desired_angle_deg)
    w_init = w_init / np.linalg.norm(w_init)
    
    w_lms = w_init.copy()
    w_mmax = w_init.copy()
    w_pson = w_init.copy()
    w_pson_batch = w_init.copy()
    
    mu = 0.01  # Smaller step for stability in high-dim
    
    import time
    
    # --- Full LMS ---
    t0 = time.perf_counter()
    for it in range(config.n_iterations):
        idx = it % n_samples
        y = np.dot(w_lms.conj(), X[idx])
        e = d[idx] - y
        w_lms = w_lms + mu * e.conj() * X[idx]
        # Clip to prevent overflow
        norm = np.linalg.norm(w_lms)
        if norm > 100:
            w_lms = w_lms / norm
    w_lms = w_lms / (np.linalg.norm(w_lms) + 1e-10)
    time_lms = time.perf_counter() - t0
    mse_lms = float(np.mean(np.abs(d - X @ w_lms) ** 2))
    if np.isnan(mse_lms):
        mse_lms = 999.0
    
    # --- M-max LMS (update top 10%) ---
    m = max(1, n // 10)
    t0 = time.perf_counter()
    for it in range(config.n_iterations):
        idx = it % n_samples
        y = np.dot(w_mmax.conj(), X[idx])
        e = d[idx] - y
        grad = e.conj() * X[idx]
        top_idx = np.argsort(np.abs(grad))[-m:]
        w_mmax[top_idx] = w_mmax[top_idx] + mu * grad[top_idx]
        norm = np.linalg.norm(w_mmax)
        if norm > 100:
            w_mmax = w_mmax / norm
    w_mmax = w_mmax / (np.linalg.norm(w_mmax) + 1e-10)
    time_mmax = time.perf_counter() - t0
    mse_mmax = float(np.mean(np.abs(d - X @ w_mmax) ** 2))
    
    # --- PSON single-sample (original) ---
    t0 = time.perf_counter()
    best_mse_single = np.inf
    for it in range(config.n_iterations):
        idx = it % n_samples
        y = np.dot(w_pson.conj(), X[idx])
        e = d[idx] - y
        grad = e.conj() * X[idx]
        
        proposal = w_pson + mu * grad
        proposal = proposal / (np.linalg.norm(proposal) + 1e-10)
        
        # Sparse PSON exploration (10% of elements)
        n_update = max(1, n // 10)
        update_idx = rng.choice(n, n_update, replace=False)
        noise_scale = 0.02
        noise = noise_scale * (rng.standard_normal(n_update) + 1j * rng.standard_normal(n_update))
        candidate = proposal.copy()
        candidate[update_idx] = candidate[update_idx] + noise
        candidate = candidate / (np.linalg.norm(candidate) + 1e-10)
        
        # Monotonic
        cur_mse = np.abs(e) ** 2
        cand_mse = np.abs(d[idx] - np.dot(candidate.conj(), X[idx])) ** 2
        if cand_mse <= cur_mse:
            w_pson = candidate
    
    w_pson = w_pson / (np.linalg.norm(w_pson) + 1e-10)
    time_pson = time.perf_counter() - t0
    mse_pson = float(np.mean(np.abs(d - X @ w_pson) ** 2))
    
    # --- PSON-Subspace (project to lower-dim, optimize there, lift back) ---
    # Key insight: in massive MIMO, the signal subspace is low-rank
    # Use SVD to find dominant directions and optimize PSON in that subspace
    t0 = time.perf_counter()
    
    # Find signal subspace via SVD (rank-k approximation)
    # Scale subspace dimension with array size for better coverage
    if n <= 512:
        k_target = min(32, n // 4)
    elif n <= 2048:
        k_target = 64
    else:
        k_target = 128  # For 4096+ element arrays
    
    # Use truncated SVD for efficiency on large arrays
    # k_subspace cannot exceed min(n_samples, n)
    n_svd_samples = min(200, n_samples)  # Use more samples for better subspace
    U, S, Vh = np.linalg.svd(X[:n_svd_samples], full_matrices=False)
    k_subspace = min(k_target, len(S))  # Can't exceed available singular values
    V_k = Vh[:k_subspace].T  # (n, k) - top k right singular vectors
    
    # Project initialization to subspace
    w_sub = V_k.T.conj() @ w_pson_batch  # (k,)
    w_sub = w_sub / (np.linalg.norm(w_sub) + 1e-10)
    
    # Project data to subspace
    X_sub = X @ V_k  # (n_samples, k)
    
    best_mse_sub = float(np.mean(np.abs(d - X_sub @ w_sub) ** 2))
    
    for it in range(config.n_iterations):
        idx = it % n_samples
        y = np.dot(w_sub.conj(), X_sub[idx])
        e = d[idx] - y
        grad = e.conj() * X_sub[idx]
        
        # Deterministic step
        proposal = w_sub + mu * 2 * grad
        proposal = proposal / (np.linalg.norm(proposal) + 1e-10)
        
        # PSON exploration in subspace (all k dimensions)
        noise_scale = 0.05
        pson_noise = noise_scale * (rng.standard_normal(k_subspace) + 1j * rng.standard_normal(k_subspace))
        candidate = proposal + pson_noise
        candidate = candidate / (np.linalg.norm(candidate) + 1e-10)
        
        # Monotonic
        cur_mse = np.abs(e) ** 2
        cand_mse = np.abs(d[idx] - np.dot(candidate.conj(), X_sub[idx])) ** 2
        if cand_mse <= cur_mse:
            w_sub = candidate
    
    # Lift back to full space
    w_pson_batch = V_k @ w_sub
    w_pson_batch = w_pson_batch / (np.linalg.norm(w_pson_batch) + 1e-10)
    
    time_pson_batch = time.perf_counter() - t0
    mse_pson_batch = float(np.mean(np.abs(d - X @ w_pson_batch) ** 2))
    
    # Determine winners
    all_mse = {
        "Full-LMS": mse_lms, 
        "M-max-LMS": mse_mmax, 
        "PSON": mse_pson, 
        "PSON-Subspace": mse_pson_batch
    }
    all_time = {
        "Full-LMS": time_lms, 
        "M-max-LMS": time_mmax, 
        "PSON": time_pson, 
        "PSON-Subspace": time_pson_batch
    }
    
    return {
        "scenario": f"Massive MIMO ({n} elements)",
        "n_elements": n,
        "Full-LMS": {"mse": mse_lms, "time_sec": time_lms},
        "M-max-LMS": {"mse": mse_mmax, "time_sec": time_mmax, "updates_pct": 10},
        "PSON": {"mse": mse_pson, "time_sec": time_pson, "updates_pct": 10},
        "PSON-Subspace": {"mse": mse_pson_batch, "time_sec": time_pson_batch, "subspace_dim": min(32, n // 4)},
        "winner_mse": min(all_mse.keys(), key=lambda m: all_mse[m]),
        "winner_speed": min(all_time.keys(), key=lambda m: all_time[m]),
    }


#


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("PSON vs PU-LMS: Dynamic Scenarios Where Speed/Adaptivity Matter")
    print("=" * 80)
    
    all_results = {}
    
    # Scenario 1: Moving Target
    print("\n[1/3] Moving Target (Radar Tracking)...")
    for velocity in [0.2, 0.5, 1.0]:
        config = MovingTargetConfig(target_velocity_deg_per_iter=velocity)
        result = run_moving_target_test(config)
        key = f"moving_target_v{velocity}"
        all_results[key] = result
        print(f"  v={velocity} deg/iter: Winner = {result['winner_mean']}")
        print(f"    Full-LMS: {result['Full-LMS']['mean_tracking_error']:.4f}")
        print(f"    M-max:    {result['M-max-LMS']['mean_tracking_error']:.4f}")
        print(f"    PSON:     {result['PSON']['mean_tracking_error']:.4f}")
    
    # Scenario 2: Massive MIMO (scaling to large-scale arrays)
    print("\n[2/3] Massive MIMO (256 to 8192 elements)...")
    mimo_sizes = [256, 512, 1024, 2048, 4096, 8192]
    for n_elem in mimo_sizes:
        config = MassiveMIMOConfig(n_elements=n_elem)
        result = run_massive_mimo_test(config)
        key = f"massive_mimo_{n_elem}"
        all_results[key] = result
        print(f"  {n_elem} elements:")
        print(f"    MSE Winner: {result['winner_mse']}")
        print(f"    Full-LMS:    MSE={result['Full-LMS']['mse']:.6f}, Time={result['Full-LMS']['time_sec']:.3f}s")
        print(f"    M-max:       MSE={result['M-max-LMS']['mse']:.6f}, Time={result['M-max-LMS']['time_sec']:.3f}s")
        print(f"    PSON:        MSE={result['PSON']['mse']:.6f}, Time={result['PSON']['time_sec']:.3f}s")
        print(f"    PSON-Sub:    MSE={result['PSON-Subspace']['mse']:.6f}, Time={result['PSON-Subspace']['time_sec']:.3f}s")
    
    # Jammer nulling comparison moved to fair comparison script:
    # experiments/discrete_applications/pson_vs_lms_fair_comparison.py
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    pson_wins = sum(1 for r in all_results.values() 
                   if r.get("winner_mean") == "PSON" or 
                      r.get("winner_mse") in ["PSON", "PSON-Subspace"] or 
                      r.get("winner_mean_sinr") == "PSON")
    total = len(all_results)
    
    print(f"\nPSON (any variant) Wins: {pson_wins}/{total} scenarios")
    
    # Save
    with open("pson_dynamic_scenarios_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved: pson_dynamic_scenarios_results.json")


if __name__ == "__main__":
    main()

