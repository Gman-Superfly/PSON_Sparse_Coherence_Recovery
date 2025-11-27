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
# ACTUAL EXPERIMENT 004
# RH Sweep: On-line (sigma=0.5) vs Off-line (sigma=0.6) zeta-coupled vector Homeostat
# - Uses best config from Exp 003 with small perturbations and 3 seeds for robustness
# - Reports aggregated metrics and plots mean energy for the base config across seeds
# Artifacts:
#  - actual_experiment_004_results.csv
#  - actual_experiment_004_summary.json
#  - actual_experiment_004_finalV_bar.png
#  - actual_experiment_004_energy_mean.png
# =============================================================================


def first_25_primes() -> List[int]:
    return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]


def build_gaps_primes() -> List[float]:
    return [float(p * 10) for p in first_25_primes()]


def compute_zeta_re_for_gaps_sigma(gaps_um: List[float], sigma: float = 0.5, t_scale: float = 10.0) -> np.ndarray:
    """
    Compute Re(zeta(sigma + i t_i)) per gap with t_i scaled by gap/max_gap * t_scale.
    Fallback: synthetic zeta-like sum of sines; use sigma to damp amplitude.
    """
    gaps = np.asarray(gaps_um, dtype=float)
    max_gap = float(np.max(gaps)) + 1e-8
    t = (gaps / max_gap) * t_scale
    if HAS_MPMATH:
        vals = [float(mp.re(mp.zeta(sigma + 1j * float(tt)))) for tt in t]
        return np.asarray(vals, dtype=float)
    zeros = np.array([14.134725, 21.022040, 25.010857, 30.424876, 32.935062], dtype=float)
    zsig = np.sum(np.sin(2 * np.pi * np.outer(t, zeros)), axis=1)
    # Off-line damping heuristic: multiply by (0.5 / sigma)
    damp = 0.5 / max(1e-6, sigma)
    return (damp * zsig).astype(float)


def simulate_interference_zeta(
    gaps_um: List[float],
    phases: np.ndarray,
    zeta_re: np.ndarray,
    zeta_gain: float,
) -> np.ndarray:
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


def run_homeostat_vector_zeta(
    gaps_um: List[float],
    zeta_re: np.ndarray,
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

    I0 = simulate_interference_zeta(gaps_um, phases, zeta_re, zeta_gain)
    V0 = calculate_visibility(I0)
    E0 = energy_from_visibility(V0)
    energies.append(E0)
    visibilities.append(V0)

    for _ in range(steps):
        I_cur = simulate_interference_zeta(gaps_um, phases, zeta_re, zeta_gain)
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
        I_new = simulate_interference_zeta(gaps_um, candidate, zeta_re, zeta_gain)
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
            I_det = simulate_interference_zeta(gaps_um, proposal, zeta_re, zeta_gain)
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


def run_robust_sweep(
    steps: int,
    w: float,
    base_zeta_gain: float,
    base_lr: float,
    base_noise: float,
    seeds: List[int],
) -> Tuple[Dict[str, float], List[Dict[str, float]], Dict[str, List[float]]]:
    gaps = build_gaps_primes()
    sigmas = [0.5, 0.6]  # on-line vs off-line

    zg_vals = [base_zeta_gain - 0.05, base_zeta_gain, base_zeta_gain + 0.05]
    lr_vals = [max(1e-4, base_lr - 0.02), base_lr, base_lr + 0.02]
    nz_vals = [max(1e-4, base_noise - 0.01), base_noise, base_noise + 0.01]

    rows: List[Dict[str, float]] = []
    summary: Dict[str, Dict[str, float]] = {}
    energy_means_base: Dict[str, List[float]] = {}

    # Track base-config energies across seeds to plot mean
    base_energy_by_sigma: Dict[float, List[List[float]]] = {0.5: [], 0.6: []}

    for sigma in sigmas:
        zeta_re = compute_zeta_re_for_gaps_sigma(gaps_um=gaps, sigma=sigma, t_scale=10.0)

        final_Vs: List[float] = []
        df90s: List[int] = []
        accs: List[float] = []

        for seed in seeds:
            for zg in zg_vals:
                for lr in lr_vals:
                    for nz in nz_vals:
                        res = run_homeostat_vector_zeta(
                            gaps_um=gaps,
                            zeta_re=zeta_re,
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
                            "seed": seed,
                            "zeta_gain": zg,
                            "lr": lr,
                            "noise": nz,
                            "final_V": res["final_V"],
                            "deltaF90": df90,
                            "accept_rate": res["accept_rate"],
                        })

                        final_Vs.append(res["final_V"])
                        df90s.append(df90)
                        accs.append(res["accept_rate"])

                        # Collect base config energies (for plot)
                        if abs(zg - base_zeta_gain) < 1e-12 and abs(lr - base_lr) < 1e-12 and abs(nz - base_noise) < 1e-12:
                            base_energy_by_sigma[sigma].append(res["energies"])

        # Aggregate per sigma
        final_Vs_arr = np.array(final_Vs, dtype=float)
        df90s_arr = np.array([d if d >= 0 else steps for d in df90s], dtype=float)
        accs_arr = np.array(accs, dtype=float)

        summary[str(sigma)] = {
            "mean_final_V": float(final_Vs_arr.mean()),
            "std_final_V": float(final_Vs_arr.std(ddof=1)) if final_Vs_arr.size > 1 else 0.0,
            "mean_deltaF90": float(df90s_arr.mean()),
            "std_deltaF90": float(df90s_arr.std(ddof=1)) if df90s_arr.size > 1 else 0.0,
            "mean_accept_rate": float(accs_arr.mean()),
        }

    # Mean energy curves for base config across seeds (per sigma)
    energy_means_base = {}
    for sigma in sigmas:
        run_list = base_energy_by_sigma[sigma]
        if not run_list:
            energy_means_base[str(sigma)] = []
            continue
        min_len = min(len(e) for e in run_list)
        aligned = np.array([e[:min_len] for e in run_list], dtype=float)
        energy_means_base[str(sigma)] = list(aligned.mean(axis=0))

    return summary, rows, energy_means_base


def save_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def plot_finalV_bar(summary: Dict[str, Dict[str, float]]) -> None:
    labels = ["σ=0.5 (on-line)", "σ=0.6 (off-line)"]
    means = [summary["0.5"]["mean_final_V"], summary["0.6"]["mean_final_V"]]
    stds = [summary["0.5"]["std_final_V"], summary["0.6"]["std_final_V"]]
    plt.figure(figsize=(7, 5))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=stds, capsize=6, color=["#4C78A8", "#F58518"])
    plt.xticks(x, labels)
    plt.ylabel("Final Visibility (mean ± sd)")
    plt.title("Experiment 004: Final V — On-line vs Off-line")
    plt.tight_layout()
    plt.savefig("actual_experiment_004_finalV_bar.png")


def plot_energy_means(energy_means_base: Dict[str, List[float]]) -> None:
    plt.figure(figsize=(10, 6))
    if energy_means_base.get("0.5"):
        plt.plot(energy_means_base["0.5"], label="σ=0.5 (base config mean)", linewidth=2)
    if energy_means_base.get("0.6"):
        plt.plot(energy_means_base["0.6"], label="σ=0.6 (base config mean)", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Energy (1 - V)^2")
    plt.title("Experiment 004: Mean Energy (base config across seeds)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("actual_experiment_004_energy_mean.png")


def main():
    parser = argparse.ArgumentParser(description="Experiment 004: RH on/off-line sweep with robustness")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--w", type=float, default=0.2)
    parser.add_argument("--base_zeta_gain", type=float, default=0.3)
    parser.add_argument("--base_lr", type=float, default=0.1)
    parser.add_argument("--base_noise", type=float, default=0.03)
    parser.add_argument("--seeds", type=str, default="41,42,43")
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    summary, rows, energy_means_base = run_robust_sweep(
        steps=args.steps,
        w=args.w,
        base_zeta_gain=args.base_zeta_gain,
        base_lr=args.base_lr,
        base_noise=args.base_noise,
        seeds=seeds,
    )

    save_csv("actual_experiment_004_results.csv", rows)
    plot_finalV_bar(summary)
    plot_energy_means(energy_means_base)

    out = {
        "summary": summary,
        "grid_size_per_sigma": len(rows) // 2,  # symmetric grid across two sigmas
        "seeds": seeds,
        "params": {
            "steps": args.steps,
            "w": args.w,
            "base_zeta_gain": args.base_zeta_gain,
            "base_lr": args.base_lr,
            "base_noise": args.base_noise,
        },
        "mpmath": HAS_MPMATH,
    }
    with open("actual_experiment_004_summary.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Experiment 004 Summary ===")
    print(json.dumps(out, indent=2))
    print("Artifacts: actual_experiment_004_results.csv, actual_experiment_004_finalV_bar.png, actual_experiment_004_energy_mean.png, actual_experiment_004_summary.json")
    if not HAS_MPMATH:
        print("Note: mpmath not available, used synthetic zeta-like signal fallback.")


if __name__ == "__main__":
    main()


