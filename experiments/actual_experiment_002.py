import argparse
import json
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

# Try to import mpmath for true zeta; fallback to synthetic zeros if missing
try:
    import mpmath as mp
    HAS_MPMATH = True
except Exception:
    HAS_MPMATH = False


# =============================================================================
# ACTUAL EXPERIMENT 002
# Zeta-coupled optical phases + vector Homeostat (Wormhole + PSON)
# =============================================================================
# - We compute a zeta-driven phase component per gap: phi_zeta_i ∝ Re(ζ(1/2 + i t_i))
# - We evolve per-gap phase gates with a vector Homeostat:
#     * Wormhole gradient: non-local, gate-independent update (Eq. 3)
#     * PSON: precision-scaled metric-orthogonal noise with down-only acceptance
# - We compare primes vs uniform gaps, report final visibility, ΔF90, acceptance
# - Artifacts: actual_experiment_002_energy.png + printed JSON summary
# =============================================================================


def first_25_primes() -> List[int]:
    return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]


def build_gaps(config: str) -> List[float]:
    primes = first_25_primes()
    if config == "uniform":
        return [100.0] * len(primes)
    if config == "primes":
        return [float(p * 10) for p in primes]
    raise ValueError(f"Unknown config: {config}")


def compute_zeta_re_for_gaps(gaps_um: List[float], t_scale: float = 10.0) -> np.ndarray:
    """
    Compute Re(zeta(1/2 + i t_i)) per gap with t_i scaled by gap/max_gap * t_scale.
    Fallback: synthetic zeta-like sum of sines over a few known zeros if mpmath is missing.
    """
    gaps = np.asarray(gaps_um, dtype=float)
    max_gap = float(np.max(gaps)) + 1e-8
    t = (gaps / max_gap) * t_scale
    if HAS_MPMATH:
        vals = [float(mp.re(mp.zeta(0.5 + 1j * float(tt)))) for tt in t]
        return np.asarray(vals, dtype=float)
    # Fallback: truncated zeros mixture
    zeros = np.array([14.134725, 21.022040, 25.010857, 30.424876, 32.935062], dtype=float)
    # Normalize t to avoid extreme phases
    zsig = np.sum(np.sin(2 * np.pi * np.outer(t, zeros)), axis=1)
    return zsig.astype(float)


def simulate_interference_zeta(
    gaps_um: List[float],
    phases: np.ndarray,
    zeta_re: np.ndarray,
    zeta_gain: float,
) -> np.ndarray:
    """
    Compute intensity I(x) by averaging interference patterns over gaps.
    Per-gap phase: k*d*sin(theta) + zeta_gain * zeta_re[i] + phases[i]
    """
    assert len(gaps_um) == phases.shape[0] == zeta_re.shape[0], "phases/zeta must match gaps"
    lambda_nm = 633.0
    k = 2 * np.pi / (lambda_nm * 1e-9)
    L = 1.0
    x_screen = np.linspace(-0.005, 0.005, 500)
    theta = x_screen / L
    amp_per_slit = 0.5

    intensities = []
    for i, g_um in enumerate(gaps_um):
        d = g_um * 1e-6
        phi = k * d * np.sin(theta) + zeta_gain * zeta_re[i] + phases[i]
        field1 = amp_per_slit * np.exp(1j * 0.0)
        field2 = amp_per_slit * np.exp(1j * phi)
        I = np.abs(field1 + field2) ** 2
        intensities.append(I)
    return np.mean(intensities, axis=0)


def calculate_visibility(I: np.ndarray) -> float:
    I_max = float(np.max(I))
    I_min = float(np.min(I))
    return (I_max - I_min) / (I_max + I_min + 1e-8)


def energy_from_visibility(V: float) -> float:
    return (1.0 - V) ** 2


def compute_precision_and_weights(gaps_um: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precision Λ and wormhole weights from gap irregularity.
    """
    gaps = np.asarray(gaps_um, dtype=float)
    g_mean = float(np.mean(gaps))
    g_var = float(np.var(gaps)) + 1e-8
    irregularity = ((gaps - g_mean) ** 2) / g_var
    precision = 1.0 / (1.0 + irregularity)
    precision = np.clip(precision, 1e-4, 1.0)
    weights = irregularity.copy()
    if float(np.sum(weights)) <= 1e-8:
        weights = np.ones_like(weights)
    weights = weights / float(np.sum(weights))
    return precision, weights


def project_noise_metric_orthogonal(
    grad: np.ndarray, precision: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """
    Metric-orthogonal projection of N(0,I) relative to grad with metric M=diag(precision).
    Ensures grad^T M delta = 0.
    """
    z = rng.normal(0.0, 1.0, size=grad.shape[0])
    Mz = precision * z
    Mg = precision * grad
    denom = float(np.dot(grad, Mg)) + 1e-12
    if abs(denom) < 1e-18:
        return z
    alpha = float(np.dot(grad, Mz)) / denom
    return z - alpha * grad


def run_homeostat_vector_zeta(
    gaps_um: List[float],
    zeta_re: np.ndarray,
    zeta_gain: float,
    steps: int = 200,
    w: float = 0.2,
    lr: float = 0.1,
    noise_scale: float = 0.02,
    use_pson: bool = True,
    seed: int = 42,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    d = len(gaps_um)
    phases = np.zeros(d, dtype=float)
    precision, weights = compute_precision_and_weights(gaps_um)

    energies: List[float] = []
    visibilities: List[float] = []
    phase_hist: List[np.ndarray] = []
    accepted = 0
    attempts = 0

    I0 = simulate_interference_zeta(gaps_um, phases, zeta_re, zeta_gain)
    V0 = calculate_visibility(I0)
    E0 = energy_from_visibility(V0)
    energies.append(E0)
    visibilities.append(V0)
    phase_hist.append(phases.copy())

    for _ in range(steps):
        I_cur = simulate_interference_zeta(gaps_um, phases, zeta_re, zeta_gain)
        V_cur = calculate_visibility(I_cur)
        E_cur = energy_from_visibility(V_cur)

        benefit = E_cur
        grad = -w * benefit * weights  # wormhole gradient vector

        proposal = phases - lr * grad

        if use_pson:
            delta_perp = project_noise_metric_orthogonal(grad=grad, precision=precision, rng=rng)
            noise = (delta_perp / (np.sqrt(precision) + 1e-12)) * noise_scale
            candidate = proposal + noise
        else:
            candidate = proposal

        attempts += 1
        I_new = simulate_interference_zeta(gaps_um, candidate, zeta_re, zeta_gain)
        V_new = calculate_visibility(I_new)
        E_new = energy_from_visibility(V_new)

        if E_new <= E_cur:
            phases = candidate
            accepted += 1
            energies.append(E_new)
            visibilities.append(V_new)
            phase_hist.append(phases.copy())
            continue

        if use_pson:
            attempts += 1
            I_det = simulate_interference_zeta(gaps_um, proposal, zeta_re, zeta_gain)
            V_det = calculate_visibility(I_det)
            E_det = energy_from_visibility(V_det)
            if E_det <= E_cur:
                phases = proposal
                accepted += 1
                energies.append(E_det)
                visibilities.append(V_det)
                phase_hist.append(phases.copy())
                continue

        energies.append(E_cur)
        visibilities.append(V_cur)
        phase_hist.append(phases.copy())

    return {
        "energies": energies,
        "visibilities": visibilities,
        "phases": phase_hist,
        "accepted": accepted,
        "attempts": attempts,
        "accept_rate": 0.0 if attempts == 0 else accepted / attempts,
        "final_V": float(visibilities[-1]),
    }


def delta_f90_steps(energies: List[float]) -> int:
    E0 = energies[0]
    Ef = energies[-1]
    target = Ef + 0.1 * (E0 - Ef)
    for i, E in enumerate(energies):
        if E <= target:
            return i
    return -1


def run_experiment(steps: int, w: float, lr: float, noise: float, zeta_gain: float, seed: int) -> Dict[str, Dict[str, float]]:
    configs = ["uniform", "primes"]
    summary: Dict[str, Dict[str, float]] = {}
    plt.figure(figsize=(10, 6))

    for cfg in configs:
        gaps = build_gaps(cfg)
        zeta_re = compute_zeta_re_for_gaps(gaps_um=gaps, t_scale=10.0)

        # Baseline (no updates, phases=0)
        I_base = simulate_interference_zeta(gaps, np.zeros(len(gaps)), zeta_re, zeta_gain)
        V_base = calculate_visibility(I_base)

        # Vector Homeostat runs
        res_no = run_homeostat_vector_zeta(gaps, zeta_re, zeta_gain, steps=steps, w=w, lr=lr, noise_scale=noise, use_pson=False, seed=seed)
        res_ps = run_homeostat_vector_zeta(gaps, zeta_re, zeta_gain, steps=steps, w=w, lr=lr, noise_scale=noise, use_pson=True, seed=seed)

        df90_no = delta_f90_steps(res_no["energies"])
        df90_ps = delta_f90_steps(res_ps["energies"])

        summary[cfg] = {
            "baseline_V": float(V_base),
            "final_V_no_pson": float(res_no["final_V"]),
            "final_V_pson": float(res_ps["final_V"]),
            "accept_rate_no_pson": float(res_no["accept_rate"]),
            "accept_rate_pson": float(res_ps["accept_rate"]),
            "deltaF90_steps_no_pson": int(df90_no),
            "deltaF90_steps_pson": int(df90_ps),
        }

        # Plot energies
        plt.plot(res_no["energies"], linestyle="--", alpha=0.85, label=f"{cfg} (no PSON)")
        plt.plot(res_ps["energies"], linewidth=2, alpha=0.9, label=f"{cfg} (PSON)")

    plt.xlabel("Step")
    plt.ylabel("Energy (1 - V)^2")
    plt.title("Experiment 002: Zeta-Coupled Vector Homeostat (PSON vs No-PSON)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("actual_experiment_002_energy.png")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Experiment 002: Zeta-coupled vector homeostat")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--w", type=float, default=0.2, help="Wormhole gain")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--noise", type=float, default=0.02, help="PSON noise scale")
    parser.add_argument("--zeta_gain", type=float, default=0.2, help="Gain for zeta phase contribution")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    summary = run_experiment(steps=args.steps, w=args.w, lr=args.lr, noise=args.noise, zeta_gain=args.zeta_gain, seed=args.seed)
    print("\n=== Experiment 002 Summary ===")
    print(json.dumps(summary, indent=2))
    print("Plot saved: actual_experiment_002_energy.png")
    if not HAS_MPMATH:
        print("Note: mpmath not available, used synthetic zeta-like signal fallback.")


if __name__ == "__main__":
    main()


