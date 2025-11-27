"""
Find optimal step count for PSON path integral.
Too few steps = underfit, too many = noise in tails.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List


def generate_primes(n: int) -> List[int]:
    primes = []
    candidate = 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1 if candidate == 2 else 2
    return primes


def build_sparse_gaps(count: int) -> np.ndarray:
    return np.array([p * 10.0 for p in generate_primes(count)])


def build_dense_gaps(sparse_gaps: np.ndarray, num_dense: int) -> np.ndarray:
    return np.linspace(sparse_gaps.min(), sparse_gaps.max(), num_dense)


_X_SCREEN = np.linspace(-0.005, 0.005, 500)
_THETA = _X_SCREEN / 1.0


def simulate_intensity(gaps: np.ndarray, phases: np.ndarray) -> np.ndarray:
    k = 2 * np.pi / (633.0 * 1e-9)
    intensities = []
    for gap, phase in zip(gaps, phases):
        d = gap * 1e-6
        phi = k * d * np.sin(_THETA) + phase
        I = np.abs(0.5 + 0.5 * np.exp(1j * phi)) ** 2
        intensities.append(I)
    return np.mean(intensities, axis=0)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def compute_precision(gaps: np.ndarray) -> np.ndarray:
    gap_mean = np.mean(gaps)
    gap_var = np.var(gaps) + 1e-8
    irregularity = (gaps - gap_mean) ** 2 / gap_var
    return np.clip(1.0 / (1.0 + irregularity), 1e-4, 1.0)


def compute_weights(gaps: np.ndarray) -> np.ndarray:
    gap_mean = np.mean(gaps)
    gap_var = np.var(gaps) + 1e-8
    irregularity = (gaps - gap_mean) ** 2 / gap_var
    weights = irregularity.copy()
    if weights.sum() < 1e-8:
        weights = np.ones_like(weights)
    return weights / weights.sum()


def project_orthogonal(grad: np.ndarray, precision: np.ndarray, rng) -> np.ndarray:
    z = rng.normal(0, 1, size=grad.shape[0])
    Mz = precision * z
    Mg = precision * grad
    denom = np.dot(grad, Mg) + 1e-12
    if abs(denom) < 1e-18:
        return z
    alpha = np.dot(grad, Mz) / denom
    return z - alpha * grad


def run_pson_with_tracking(
    sparse_gaps: np.ndarray,
    I_target: np.ndarray,
    max_steps: int = 500,
    seed: int = 42,
) -> tuple:
    """Run PSON and track MSE at each step."""
    rng = np.random.default_rng(seed)
    n = len(sparse_gaps)
    
    phases = np.zeros(n)
    weights = compute_weights(sparse_gaps)
    precision = compute_precision(sparse_gaps)
    
    w, lr, noise_scale = 0.2, 0.1, 0.02
    
    mse_curve = []
    best_mse = float('inf')
    best_step = 0
    best_phases = phases.copy()
    
    for step in range(max_steps):
        I_cur = simulate_intensity(sparse_gaps, phases)
        E_cur = mse(I_cur, I_target)
        mse_curve.append(E_cur)
        
        if E_cur < best_mse:
            best_mse = E_cur
            best_step = step
            best_phases = phases.copy()
        
        grad = -w * E_cur * weights
        proposal = phases - lr * grad
        
        delta = project_orthogonal(grad, precision, rng)
        noise = (delta / (np.sqrt(precision) + 1e-12)) * noise_scale
        candidate = proposal + noise
        
        I_new = simulate_intensity(sparse_gaps, candidate)
        E_new = mse(I_new, I_target)
        
        if E_new <= E_cur:
            phases = candidate
        else:
            I_det = simulate_intensity(sparse_gaps, proposal)
            E_det = mse(I_det, I_target)
            if E_det <= E_cur:
                phases = proposal
    
    return mse_curve, best_step, best_mse, best_phases


def main():
    n_sparse, n_dense = 25, 200
    
    sparse_gaps = build_sparse_gaps(n_sparse)
    dense_gaps = build_dense_gaps(sparse_gaps, n_dense)
    I_target = simulate_intensity(dense_gaps, np.zeros(n_dense))
    
    print("=" * 60)
    print("FINDING OPTIMAL STEP COUNT")
    print("=" * 60)
    
    # Run with tracking
    mse_curve, best_step, best_mse, best_phases = run_pson_with_tracking(
        sparse_gaps, I_target, max_steps=500, seed=42
    )
    
    print(f"\nBest MSE: {best_mse:.6f} at step {best_step}")
    
    # Find where MSE stops improving significantly
    window = 20
    smoothed = np.convolve(mse_curve, np.ones(window)/window, mode='valid')
    
    # Find elbow - where improvement slows
    improvements = -np.diff(smoothed)
    threshold = 0.1 * improvements.max()  # 10% of max improvement
    
    for i, imp in enumerate(improvements):
        if imp < threshold:
            elbow_step = i + window // 2
            break
    else:
        elbow_step = len(mse_curve) - 1
    
    print(f"Elbow (diminishing returns): step {elbow_step}")
    print(f"MSE at elbow: {mse_curve[elbow_step]:.6f}")
    
    # Plot MSE curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MSE over steps
    ax1 = axes[0]
    ax1.plot(mse_curve, label='MSE')
    ax1.axvline(best_step, color='g', linestyle='--', label=f'Best: step {best_step}')
    ax1.axvline(elbow_step, color='r', linestyle='--', label=f'Elbow: step {elbow_step}')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('MSE')
    ax1.set_title('MSE vs Steps')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Intensity at best step
    ax2 = axes[1]
    I_init = simulate_intensity(sparse_gaps, np.zeros(n_sparse))
    I_best = simulate_intensity(sparse_gaps, best_phases)
    
    ax2.plot(I_target, label='Dense Target', linewidth=2, alpha=0.8)
    ax2.plot(I_init, label='Sparse Init', linestyle='--', alpha=0.6)
    ax2.plot(I_best, label=f'Sparse @ step {best_step}', linewidth=2)
    ax2.set_xlabel('Screen Position')
    ax2.set_ylabel('Intensity')
    ax2.set_title(f'Intensity at Best Step (MSE={best_mse:.6f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimal_steps_analysis.png', dpi=150)
    plt.close()
    
    print("\nPlot saved: optimal_steps_analysis.png")
    
    # Test specific step counts
    print("\n" + "-" * 60)
    print("MSE AT SPECIFIC STEP COUNTS:")
    print("-" * 60)
    test_steps = [50, 80, 100, 120, 150, 200, 300]
    for s in test_steps:
        if s < len(mse_curve):
            print(f"  Step {s:3d}: MSE = {mse_curve[s]:.6f}")


if __name__ == "__main__":
    main()

