import argparse
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Homeostat Vector Test: PSON vs No-PSON on Multi-Phase Optical Model
# =============================================================================
# - Multi-dimensional gate: per-gap phase vector (one phase per gap)
# - Energy: F = (1 - Visibility)^2 (minimize to maximize coherence)
# - Wormhole gradient: dF/d eta_i = -w * Delta_benefit_i (independent of current eta_i)
# - PSON: precision-scaled, metric-orthogonal noise; down-only acceptance
# - Outputs: metrics (final V, ΔF90, acceptance rate) and plot
# =============================================================================


def simulate_interference_vector(gaps_um: List[float], phases: np.ndarray) -> np.ndarray:
    """
    Compute I(x) by averaging interference patterns over gaps, each with its own phase.
    Args:
        gaps_um: list of gap sizes in micrometers (μm)
        phases: np.ndarray of shape (len(gaps_um),), phase shift per gap (radians)
    Returns:
        I(x): np.ndarray of intensity across the screen
    """
    assert len(gaps_um) == phases.shape[0], "phases must align with gaps"
    # Physical constants
    lambda_nm = 633.0
    k = 2 * np.pi / (lambda_nm * 1e-9)
    L = 1.0
    x_screen = np.linspace(-0.005, 0.005, 500)
    theta = x_screen / L
    amp_per_slit = 0.5

    layer_intensities = []
    for i, g_um in enumerate(gaps_um):
        d = g_um * 1e-6
        phi = k * d * np.sin(theta) + phases[i]
        field1 = amp_per_slit * np.exp(1j * 0.0)
        field2 = amp_per_slit * np.exp(1j * phi)
        I = np.abs(field1 + field2) ** 2
        layer_intensities.append(I)

    return np.mean(layer_intensities, axis=0)


def calculate_visibility(I: np.ndarray) -> float:
    """Fringe visibility (contrast)."""
    I_max = float(np.max(I))
    I_min = float(np.min(I))
    return (I_max - I_min) / (I_max + I_min + 1e-8)


def energy_from_visibility(V: float) -> float:
    """Energy to minimize: lower when visibility is higher."""
    return (1.0 - V) ** 2


def compute_precision_and_weights(gaps_um: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Derive per-parameter precision (Λ) and wormhole weights from gap irregularity.
    - precision_i in (0, 1]: lower for more irregular gaps (more exploration permitted)
    - weights_i >= 0, sum to 1: more irregular gaps receive larger non-local credit
    """
    gaps = np.asarray(gaps_um, dtype=float)
    g_mean = float(np.mean(gaps))
    g_var = float(np.var(gaps)) + 1e-8
    irregularity = ((gaps - g_mean) ** 2) / g_var  # dimensionless

    # Precision in (0, 1], small for large irregularity
    precision = 1.0 / (1.0 + irregularity)
    precision = np.clip(precision, 1e-4, 1.0)

    # Wormhole weights proportional to irregularity (default uniform if all ~0)
    weights = irregularity.copy()
    if float(np.sum(weights)) <= 1e-8:
        weights = np.ones_like(weights)
    weights = weights / float(np.sum(weights))
    return precision, weights


def project_noise_metric_orthogonal(
    grad: np.ndarray, precision: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """
    Metric-orthogonal projection of standard normal noise relative to grad with metric M = diag(precision).
    Returns a vector δ such that g^T M δ = 0.
    """
    assert grad.ndim == 1 and precision.ndim == 1 and grad.shape == precision.shape
    d = grad.shape[0]
    z = rng.normal(0.0, 1.0, size=d)
    # Metric inner products: <a,b>_M = a^T M b
    Mz = precision * z
    Mg = precision * grad
    denom = float(np.dot(grad, Mg)) + 1e-12
    if abs(denom) < 1e-18:
        # Gradient is (near) zero; return any noise (scaled later)
        return z
    alpha = float(np.dot(grad, Mz)) / denom
    delta_perp = z - alpha * grad
    return delta_perp


@dataclass
class RunResult:
    energies: List[float]
    visibilities: List[float]
    phases: List[np.ndarray]
    accepted: int
    attempts: int

    @property
    def acceptance_rate(self) -> float:
        return 0.0 if self.attempts == 0 else self.accepted / self.attempts


def run_homeostat_vector(
    gaps_um: List[float],
    steps: int = 200,
    w: float = 0.2,
    lr: float = 0.1,
    noise_scale: float = 0.02,
    use_pson: bool = True,
    seed: int = 42,
) -> RunResult:
    """
    Multi-parameter homeostat with down-only acceptance and optional PSON.
    - Gradient: g_i = -w * Delta_benefit_i, with Delta_benefit_i ∝ weights_i * current_energy
    - PSON: δ = Λ^{-1/2} proj_{⊥_M} N(0, I), candidate accepted only if energy decreases
    """
    rng = np.random.default_rng(seed)

    d = len(gaps_um)
    phases = np.zeros(d, dtype=float)
    precision, weights = compute_precision_and_weights(gaps_um)

    energies: List[float] = []
    visibilities: List[float] = []
    phase_hist: List[np.ndarray] = []
    accepted = 0
    attempts = 0

    # Initial energy
    I0 = simulate_interference_vector(gaps_um, phases)
    V0 = calculate_visibility(I0)
    E0 = energy_from_visibility(V0)
    energies.append(E0)
    visibilities.append(V0)
    phase_hist.append(phases.copy())

    for _ in range(steps):
        # Global measurement
        I_cur = simulate_interference_vector(gaps_um, phases)
        V_cur = calculate_visibility(I_cur)
        E_cur = energy_from_visibility(V_cur)

        # Wormhole gradient vector (independent of current eta_i, scaled by benefit)
        benefit = E_cur  # proportional to current defect
        grad = -w * benefit * weights  # shape (d,)

        # Deterministic proposal
        proposal = phases - lr * grad

        # PSON proposal
        if use_pson:
            delta_perp = project_noise_metric_orthogonal(grad=grad, precision=precision, rng=rng)
            # Precision-scaled noise (std ∝ 1/sqrt(precision)), scaled by noise_scale
            noise = (delta_perp / (np.sqrt(precision) + 1e-12)) * noise_scale
            candidate = proposal + noise
        else:
            candidate = proposal

        # Down-only acceptance with one backoff to deterministic
        attempts += 1
        I_new = simulate_interference_vector(gaps_um, candidate)
        V_new = calculate_visibility(I_new)
        E_new = energy_from_visibility(V_new)

        if E_new <= E_cur:
            phases = candidate
            accepted += 1
            energies.append(E_new)
            visibilities.append(V_new)
            phase_hist.append(phases.copy())
            continue

        # Try deterministic only (no noise) if PSON was used
        if use_pson:
            attempts += 1
            I_det = simulate_interference_vector(gaps_um, proposal)
            V_det = calculate_visibility(I_det)
            E_det = energy_from_visibility(V_det)
            if E_det <= E_cur:
                phases = proposal
                accepted += 1
                energies.append(E_det)
                visibilities.append(V_det)
                phase_hist.append(phases.copy())
                continue

        # Reject step
        energies.append(E_cur)
        visibilities.append(V_cur)
        phase_hist.append(phases.copy())

    return RunResult(energies=energies, visibilities=visibilities, phases=phase_hist, accepted=accepted, attempts=attempts)


def delta_f90_steps(energies: List[float]) -> int:
    """
    Steps needed to achieve 90% of total energy drop.
    Returns -1 if not achieved.
    """
    E0 = energies[0]
    Ef = energies[-1]
    target = Ef + 0.1 * (E0 - Ef)  # E <= target means 90% of drop achieved
    for i, E in enumerate(energies):
        if E <= target:
            return i
    return -1


def first_25_primes() -> List[int]:
    return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]


def build_gaps(config: str) -> List[float]:
    primes = first_25_primes()
    if config == "uniform":
        return [100.0] * len(primes)
    if config == "primes":
        return [float(p * 10) for p in primes]
    raise ValueError(f"Unknown config: {config}")


def run_ablation(steps: int, w: float, lr: float, noise_scale: float, seed: int) -> Dict[str, Dict[str, float]]:
    configs = ["uniform", "primes"]
    results_summary: Dict[str, Dict[str, float]] = {}
    plt.figure(figsize=(10, 6))

    for cfg in configs:
        gaps = build_gaps(cfg)
        res_n = run_homeostat_vector(gaps, steps=steps, w=w, lr=lr, noise_scale=noise_scale, use_pson=False, seed=seed)
        res_p = run_homeostat_vector(gaps, steps=steps, w=w, lr=lr, noise_scale=noise_scale, use_pson=True, seed=seed)

        df90_n = delta_f90_steps(res_n.energies)
        df90_p = delta_f90_steps(res_p.energies)

        results_summary[cfg] = {
            "final_V_no_pson": float(res_n.visibilities[-1]),
            "final_V_pson": float(res_p.visibilities[-1]),
            "accept_rate_no_pson": float(res_n.acceptance_rate),
            "accept_rate_pson": float(res_p.acceptance_rate),
            "deltaF90_steps_no_pson": int(df90_n),
            "deltaF90_steps_pson": int(df90_p),
        }

        # Plot energies
        plt.plot(res_n.energies, linestyle="--", alpha=0.8, label=f"{cfg} (no PSON)")
        plt.plot(res_p.energies, linewidth=2, alpha=0.9, label=f"{cfg} (PSON)")

    plt.xlabel("Step")
    plt.ylabel("Energy (1 - V)^2")
    plt.title("Homeostat Vector Ablation: PSON vs No-PSON")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("homeostat_vector_ablation.png")

    return results_summary


def main():
    parser = argparse.ArgumentParser(description="Vector Homeostat Ablation (PSON vs No-PSON)")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--w", type=float, default=0.2, help="Wormhole gain")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for wormhole step")
    parser.add_argument("--noise", type=float, default=0.02, help="PSON noise scale")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = run_ablation(steps=args.steps, w=args.w, lr=args.lr, noise_scale=args.noise, seed=args.seed)
    print("\n=== Ablation Summary ===")
    print(json.dumps(summary, indent=2))
    print("Plot saved: homeostat_vector_ablation.png")


if __name__ == "__main__":
    main()


