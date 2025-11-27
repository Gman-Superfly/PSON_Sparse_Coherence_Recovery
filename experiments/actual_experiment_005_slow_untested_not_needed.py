import argparse
import csv
import json
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import mpmath as mp
    HAS_MPMATH = True
except Exception:
    HAS_MPMATH = False


# =============================================================================
# ACTUAL EXPERIMENT 005
# Amplified σ sensitivity:
# - Zeta coupling varies across screen position x: φ_i(x) += zeta_gain * Re(ζ(σ + i t_x))
# - σ sweep: {0.50, 0.55, 0.60, 0.65}
# - zeta_gain sweep: {0.3, 0.5, 0.7}
# - Robustness: seeds {41,42,43}; small lr ∈ {0.10,0.12,0.15}; noise ∈ {0.02,0.03}
# - Vector Homeostat with Wormhole + PSON, down-only acceptance
# Artifacts:
#  - actual_experiment_005_results.csv
#  - actual_experiment_005_summary.json
#  - actual_experiment_005_finalV_vs_sigma.png
#  - actual_experiment_005_energy_mean.png
# =============================================================================


def first_25_primes() -> List[int]:
    return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]


def build_gaps_primes() -> List[float]:
    return [float(p * 10) for p in first_25_primes()]


def screen_and_theta() -> Tuple[np.ndarray, np.ndarray]:
    L = 1.0
    x_screen = np.linspace(-0.005, 0.005, 500)
    theta = x_screen / L
    return x_screen, theta


def compute_zeta_x_sigma(sigma: float, t_scale: float = 10.0) -> np.ndarray:
    """
    Compute Z(x) = Re(ζ(sigma + i t_x)) over the screen.
    t_x maps linearly from x∈[-X,X] to [0, t_scale].
    """
    x_screen, _ = screen_and_theta()
    # Map x to [0, t_scale]
    t_x = (x_screen - x_screen.min()) / (x_screen.max() - x_screen.min() + 1e-12) * t_scale
    if HAS_MPMATH:
        vals = [float(mp.re(mp.zeta(sigma + 1j * float(tt)))) for tt in t_x]
        return np.asarray(vals, dtype=float)
    # Fallback: zeta-like mixture of sines with damping dependent on sigma
    zeros = np.array([14.134725, 21.022040, 25.010857, 30.424876, 32.935062], dtype=float)
    zsig = np.sum(np.sin(2 * np.pi * np.outer(t_x, zeros)), axis=1)
    damp = 0.5 / max(1e-6, sigma)
    return (damp * zsig).astype(float)


def simulate_interference_zeta_x(
    gaps_um: List[float],
    phases: np.ndarray,
    sigma: float,
    zeta_gain: float,
) -> np.ndarray:
    """
    Intensity with x-dependent zeta modulation:
    φ_i(x) = k d_i sin(theta) + zeta_gain * Z(x; sigma) + phases[i]
    """
    assert len(gaps_um) == phases.shape[0], "phases must match gaps"
    lambda_nm = 633.0
    k = 2 * np.pi / (lambda_nm * 1e-9)
    _, theta = screen_and_theta()
    amp_per_slit = 0.5

    Zx = compute_zeta_x_sigma(sigma=sigma, t_scale=10.0)  # shape (len(x),)

    intensities = []
    for i, g_um in enumerate(gaps_um):
        d = g_um * 1e-6
        phi = k * d * np.sin(theta) + zeta_gain * Zx + phases[i]
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
    z = rng.normal(0.0, 1.0, size=grad.shape[0])
    Mz = precision * z
    Mg = precision * grad
    denom = float(np.dot(grad, Mg)) + 1e-12
    if abs(denom) < 1e-18:
        return z
    alpha = float(np.dot(grad, Mz)) / denom
    return z - alpha * grad


def run_homeostat_vector_zeta_x(
    gaps_um: List[float],
    sigma: float,
    zeta_gain: float,
    steps: int,
    w: float,
    lr: float,
    noise_scale: float,
    use_pson: bool,
    seed: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    d = len(gaps_um)
    phases = np.zeros(d, dtype=float)
    precision, weights = compute_precision_and_weights(gaps_um)

    energies: List[float] = []
    visibilities: List[float] = []
    accepted = 0
    attempts = 0

    I0 = simulate_interference_zeta_x(gaps_um, phases, sigma, zeta_gain)
    V0 = calculate_visibility(I0)
    E0 = energy_from_visibility(V0)
    energies.append(E0)
    visibilities.append(V0)

    for _ in range(steps):
        I_cur = simulate_interference_zeta_x(gaps_um, phases, sigma, zeta_gain)
        V_cur = calculate_visibility(I_cur)
        E_cur = energy_from_visibility(V_cur)
        benefit = E_cur
        grad = -w * benefit * weights
        proposal = phases - lr * grad
        if use_pson:
            delta_perp = project_noise_metric_orthogonal(grad=grad, precision=precision, rng=rng)
            noise = (delta_perp / (np.sqrt(precision) + 1e-12)) * noise_scale
            candidate = proposal + noise
        else:
            candidate = proposal
        attempts += 1
        I_new = simulate_interference_zeta_x(gaps_um, candidate, sigma, zeta_gain)
        V_new = calculate_visibility(I_new)
        E_new = energy_from_visibility(V_new)
        if E_new <= E_cur:
            phases = candidate
            accepted += 1
            energies.append(E_new)
            visibilities.append(V_new)
            continue
        if use_pson:
            attempts += 1
            I_det = simulate_interference_zeta_x(gaps_um, proposal, sigma, zeta_gain)
            V_det = calculate_visibility(I_det)
            E_det = energy_from_visibility(V_det)
            if E_det <= E_cur:
                phases = proposal
                accepted += 1
                energies.append(E_det)
                visibilities.append(V_det)
                continue
        energies.append(E_cur)
        visibilities.append(V_cur)

    return {
        "energies": energies,
        "final_V": float(visibilities[-1]),
        "accept_rate": 0.0 if attempts == 0 else accepted / attempts,
    }


def delta_f90_steps(energies: List[float]) -> int:
    E0 = energies[0]
    Ef = energies[-1]
    target = Ef + 0.1 * (E0 - Ef)
    for i, E in enumerate(energies):
        if E <= target:
            return i
    return -1


def run_amplified_sigma_sweep(
    steps: int,
    w: float,
    zeta_gains: List[float],
    lrs: List[float],
    noises: List[float],
    seeds: List[int],
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], List[Dict[str, float]], Dict[str, Dict[str, List[float]]]]:
    gaps = build_gaps_primes()
    sigmas = [0.50, 0.55, 0.60, 0.65]

    rows: List[Dict[str, float]] = []
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}  # sigma -> zeta_gain -> metrics
    energy_means: Dict[str, Dict[str, List[float]]] = {}  # sigma -> zeta_gain(mid) -> mean energy

    mid_zg = str(zeta_gains[len(zeta_gains) // 2])
    energy_means[mid_zg] = {}

    for sigma in sigmas:
        sigma_key = f"{sigma:.2f}"
        summary[sigma_key] = {}

        # For energy mean curves (only for mid zeta_gain across seeds with base lr/noise)
        energy_means[mid_zg][sigma_key] = []
        base_lr = lrs[1]
        base_noise = noises[0]

        for zg in zeta_gains:
            final_Vs: List[float] = []
            df90s: List[int] = []
            accs: List[float] = []

            for seed in seeds:
                for lr in lrs:
                    for nz in noises:
                        res = run_homeostat_vector_zeta_x(
                            gaps_um=gaps,
                            sigma=sigma,
                            zeta_gain=zg,
                            steps=steps,
                            w=w,
                            lr=lr,
                            noise_scale=nz,
                            use_pson=True,
                            seed=seed,
                        )
                        df90 = delta_f90_steps(res["energies"])
                        rows.append({
                            "sigma": sigma,
                            "zeta_gain": zg,
                            "seed": seed,
                            "lr": lr,
                            "noise": nz,
                            "final_V": res["final_V"],
                            "deltaF90": df90,
                            "accept_rate": res["accept_rate"],
                        })
                        final_Vs.append(res["final_V"])
                        df90s.append(df90)
                        accs.append(res["accept_rate"])

                        if abs(zg - float(mid_zg)) < 1e-12 and abs(lr - base_lr) < 1e-12 and abs(nz - base_noise) < 1e-12:
                            energy_means[mid_zg][sigma_key].append(res["energies"])

            # Aggregate per (sigma, zeta_gain)
            fv = np.array(final_Vs, dtype=float)
            df = np.array([d if d >= 0 else steps for d in df90s], dtype=float)
            ac = np.array(accs, dtype=float)
            summary[sigma_key][f"{zg:.2f}"] = {
                "mean_final_V": float(fv.mean()),
                "std_final_V": float(fv.std(ddof=1)) if fv.size > 1 else 0.0,
                "mean_deltaF90": float(df.mean()),
                "std_deltaF90": float(df.std(ddof=1)) if df.size > 1 else 0.0,
                "mean_accept_rate": float(ac.mean()),
            }

        # Reduce to mean energy curve for mid zg across seeds at base lr/noise
        runs = energy_means[mid_zg][sigma_key]
        if runs:
            min_len = min(len(e) for e in runs)
            aligned = np.array([e[:min_len] for e in runs], dtype=float)
            energy_means[mid_zg][sigma_key] = list(aligned.mean(axis=0))
        else:
            energy_means[mid_zg][sigma_key] = []

    return summary, rows, energy_means


def save_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def plot_finalV_vs_sigma(summary: Dict[str, Dict[str, Dict[str, float]]], zeta_gains: List[float]) -> None:
    plt.figure(figsize=(9, 6))
    sigmas = sorted(summary.keys(), key=lambda s: float(s))
    for zg in zeta_gains:
        means = [summary[s][f"{zg:.2f}"]["mean_final_V"] for s in sigmas]
        stds = [summary[s][f"{zg:.2f}"]["std_final_V"] for s in sigmas]
        plt.errorbar([float(s) for s in sigmas], means, yerr=stds, marker="o", capsize=4, label=f"zeta_gain={zg:.2f}")
    plt.xlabel("σ")
    plt.ylabel("Final Visibility (mean ± sd)")
    plt.title("Experiment 005: Final V vs σ for different zeta_gain")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("actual_experiment_005_finalV_vs_sigma.png")


def plot_energy_means(energy_means: Dict[str, Dict[str, List[float]]], mid_zg: float) -> None:
    plt.figure(figsize=(10, 6))
    for sigma_key, curve in energy_means[f"{mid_zg}"].items():
        if curve:
            plt.plot(curve, label=f"σ={sigma_key}")
    plt.xlabel("Step")
    plt.ylabel("Energy (1 - V)^2")
    plt.title(f"Experiment 005: Mean Energy (zeta_gain={mid_zg}) across seeds")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("actual_experiment_005_energy_mean.png")


def main():
    parser = argparse.ArgumentParser(description="Experiment 005: Amplified σ sensitivity with x-dependent zeta")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--w", type=float, default=0.2)
    parser.add_argument("--zeta_gains", type=str, default="0.3,0.5,0.7")
    parser.add_argument("--lrs", type=str, default="0.10,0.12,0.15")
    parser.add_argument("--noises", type=str, default="0.02,0.03")
    parser.add_argument("--seeds", type=str, default="41,42,43")
    args = parser.parse_args()

    zeta_gains = [float(x) for x in args.zeta_gains.split(",") if x.strip()]
    lrs = [float(x) for x in args.lrs.split(",") if x.strip()]
    noises = [float(x) for x in args.noises.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    summary, rows, energy_means = run_amplified_sigma_sweep(
        steps=args.steps,
        w=args.w,
        zeta_gains=zeta_gains,
        lrs=lrs,
        noises=noises,
        seeds=seeds,
    )

    save_csv("actual_experiment_005_results.csv", rows)
    with open("actual_experiment_005_summary.json", "w") as f:
        json.dump({"summary": summary, "params": vars(args), "mpmath": HAS_MPMATH}, f, indent=2)
    plot_finalV_vs_sigma(summary, zeta_gains)
    # Use mid zeta_gain for energy plot
    mid_zg = zeta_gains[len(zeta_gains) // 2]
    plot_energy_means(energy_means, mid_zg=mid_zg)

    print("\n=== Experiment 005 Summary ===")
    print(json.dumps({"summary": summary, "params": vars(args), "mpmath": HAS_MPMATH}, indent=2))
    print("Artifacts: actual_experiment_005_results.csv, actual_experiment_005_summary.json, actual_experiment_005_finalV_vs_sigma.png, actual_experiment_005_energy_mean.png")
    if not HAS_MPMATH:
        print("Note: mpmath not available, used synthetic zeta-like signal fallback.")


if __name__ == "__main__":
    main()


