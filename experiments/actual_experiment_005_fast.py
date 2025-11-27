import argparse
import csv
import json
from functools import lru_cache
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import mpmath as mp
    HAS_MPMATH = True
except Exception:
    HAS_MPMATH = False


# =============================================================================
# ACTUAL EXPERIMENT 005 (FAST)
# Speedups:
# - Precompute screen geometry (theta) once
# - Cache Z(x; sigma) across steps via lru_cache
# - Compute Z(x; sigma) once per sigma and reuse inside the step loop
# - Optional --no_mpmath to force fast fallback (no uninstall needed)
# - Lightweight progress logging
# Artifacts:
#  - actual_experiment_005_fast_results.csv
#  - actual_experiment_005_fast_summary.json
#  - actual_experiment_005_fast_finalV_vs_sigma.png
#  - actual_experiment_005_fast_energy_mean.png
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


# Precompute screen geometry once
_X_SCREEN, _THETA = screen_and_theta()


@lru_cache(maxsize=None)
def _compute_zeta_x_sigma_cached(sigma: float, t_scale: float, use_mpmath: bool) -> Tuple[float, ...]:
    """
    Cached computation of Z(x; sigma) = Re(ζ(sigma + i t_x)) over the screen.
    Returns a tuple for cacheability; caller converts back to np.ndarray.
    """
    # Map x to [0, t_scale]
    x_min = float(_X_SCREEN.min())
    x_max = float(_X_SCREEN.max())
    t_x = (_X_SCREEN - x_min) / (x_max - x_min + 1e-12) * t_scale

    if use_mpmath:
        vals = [float(mp.re(mp.zeta(sigma + 1j * float(tt)))) for tt in t_x]
        return tuple(vals)

    # Fallback: zeta-like mixture of sines with damping dependent on sigma
    zeros = np.array([14.134725, 21.022040, 25.010857, 30.424876, 32.935062], dtype=float)
    zsig = np.sum(np.sin(2 * np.pi * np.outer(t_x, zeros)), axis=1)
    damp = 0.5 / max(1e-6, sigma)
    out = (damp * zsig).astype(float)
    return tuple(float(v) for v in out)


def compute_zeta_x_sigma(sigma: float, t_scale: float, use_mpmath: bool) -> np.ndarray:
    """Public wrapper that returns a NumPy array (immutable use)."""
    return np.asarray(_compute_zeta_x_sigma_cached(sigma, t_scale, use_mpmath), dtype=float)


def simulate_interference_with_Zx(
    gaps_um: List[float],
    phases: np.ndarray,
    zeta_gain: float,
    Zx: np.ndarray,
) -> np.ndarray:
    """
    Intensity with x-dependent zeta modulation (precomputed Zx):
    φ_i(x) = k d_i sin(theta) + zeta_gain * Zx + phases[i]
    """
    assert len(gaps_um) == phases.shape[0], "phases must match gaps"
    lambda_nm = 633.0
    k = 2 * np.pi / (lambda_nm * 1e-9)
    theta = _THETA
    amp_per_slit = 0.5

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
    Zx: np.ndarray,
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

    I0 = simulate_interference_with_Zx(gaps_um, phases, zeta_gain, Zx)
    V0 = calculate_visibility(I0)
    E0 = energy_from_visibility(V0)
    energies.append(E0)
    visibilities.append(V0)

    for _ in range(steps):
        I_cur = simulate_interference_with_Zx(gaps_um, phases, zeta_gain, Zx)
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
        I_new = simulate_interference_with_Zx(gaps_um, candidate, zeta_gain, Zx)
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
            I_det = simulate_interference_with_Zx(gaps_um, proposal, zeta_gain, Zx)
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


def run_amplified_sigma_sweep_fast(
    steps: int,
    w: float,
    zeta_gains: List[float],
    lrs: List[float],
    noises: List[float],
    seeds: List[int],
    use_mpmath: bool,
    progress_every: int = 20,
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], List[Dict[str, float]], Dict[str, Dict[str, List[float]]]]:
    gaps = build_gaps_primes()
    sigmas = [0.50, 0.55, 0.60, 0.65]

    rows: List[Dict[str, float]] = []
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}  # sigma -> zeta_gain -> metrics
    energy_means: Dict[str, Dict[str, List[float]]] = {}  # zeta_gain(mid) -> sigma_key -> mean energy

    mid_zg_val = zeta_gains[len(zeta_gains) // 2]
    mid_zg = f"{mid_zg_val}"
    energy_means[mid_zg] = {}

    # Precompute Zx per sigma once; reuse across all steps/runs
    Zx_map: Dict[float, np.ndarray] = {}
    for sigma in sigmas:
        Zx_map[sigma] = compute_zeta_x_sigma(sigma=sigma, t_scale=10.0, use_mpmath=use_mpmath)

    base_lr = lrs[1]
    base_noise = noises[0]

    total = len(sigmas) * len(zeta_gains) * len(lrs) * len(noises) * len(seeds)
    done = 0

    for sigma in sigmas:
        sigma_key = f"{sigma:.2f}"
        summary[sigma_key] = {}
        energy_means[mid_zg][sigma_key] = []

        final_Vs_per_gain: Dict[float, List[float]] = {zg: [] for zg in zeta_gains}
        df90s_per_gain: Dict[float, List[int]] = {zg: [] for zg in zeta_gains}
        accs_per_gain: Dict[float, List[float]] = {zg: [] for zg in zeta_gains}

        for zg in zeta_gains:
            for seed in seeds:
                for lr in lrs:
                    for nz in noises:
                        res = run_homeostat_vector_zeta_x(
                            gaps_um=gaps,
                            Zx=Zx_map[sigma],
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
                        final_Vs_per_gain[zg].append(res["final_V"])
                        df90s_per_gain[zg].append(df90)
                        accs_per_gain[zg].append(res["accept_rate"])

                        if abs(zg - mid_zg_val) < 1e-12 and abs(lr - base_lr) < 1e-12 and abs(nz - base_noise) < 1e-12:
                            energy_means[mid_zg][sigma_key].append(res["energies"])

                        done += 1
                        if progress_every > 0 and (done % progress_every == 0 or done == total):
                            print(f"[fast-005] progress: {done}/{total} runs")

            # Aggregate per (sigma, zeta_gain)
            fv = np.array(final_Vs_per_gain[zg], dtype=float)
            df = np.array([d if d >= 0 else steps for d in df90s_per_gain[zg]], dtype=float)
            ac = np.array(accs_per_gain[zg], dtype=float)
            summary[sigma_key][f"{zg:.2f}"] = {
                "mean_final_V": float(fv.mean()) if fv.size else 0.0,
                "std_final_V": float(fv.std(ddof=1)) if fv.size > 1 else 0.0,
                "mean_deltaF90": float(df.mean()) if df.size else float(steps),
                "std_deltaF90": float(df.std(ddof=1)) if df.size > 1 else 0.0,
                "mean_accept_rate": float(ac.mean()) if ac.size else 0.0,
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


def plot_finalV_vs_sigma(summary: Dict[str, Dict[str, Dict[str, float]]], zeta_gains: List[float], out_path: str) -> None:
    plt.figure(figsize=(9, 6))
    sigmas = sorted(summary.keys(), key=lambda s: float(s))
    for zg in zeta_gains:
        means = [summary[s][f"{zg:.2f}"]["mean_final_V"] for s in sigmas]
        stds = [summary[s][f"{zg:.2f}"]["std_final_V"] for s in sigmas]
        plt.errorbar([float(s) for s in sigmas], means, yerr=stds, marker="o", capsize=4, label=f"zeta_gain={zg:.2f}")
    plt.xlabel("σ")
    plt.ylabel("Final Visibility (mean ± sd)")
    plt.title("Experiment 005 (FAST): Final V vs σ for different zeta_gain")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)


def plot_energy_means(energy_means: Dict[str, Dict[str, List[float]]], mid_zg: float, out_path: str) -> None:
    plt.figure(figsize=(10, 6))
    for sigma_key, curve in energy_means[f"{mid_zg}"].items():
        if curve:
            plt.plot(curve, label=f"σ={sigma_key}")
    plt.xlabel("Step")
    plt.ylabel("Energy (1 - V)^2")
    plt.title(f"Experiment 005 (FAST): Mean Energy (zeta_gain={mid_zg}) across seeds")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)


def main():
    parser = argparse.ArgumentParser(description="Experiment 005 (FAST): x-dependent zeta with caching and progress logging")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--w", type=float, default=0.2)
    parser.add_argument("--zeta_gains", type=str, default="0.3,0.5,0.7")
    parser.add_argument("--lrs", type=str, default="0.10,0.12,0.15")
    parser.add_argument("--noises", type=str, default="0.02,0.03")
    parser.add_argument("--seeds", type=str, default="41,42,43")
    parser.add_argument("--no_mpmath", action="store_true", help="Force fast fallback even if mpmath is available")
    parser.add_argument("--progress_every", type=int, default=20, help="Print progress every N runs (0 to disable)")
    args = parser.parse_args()

    zeta_gains = [float(x) for x in args.zeta_gains.split(",") if x.strip()]
    lrs = [float(x) for x in args.lrs.split(",") if x.strip()]
    noises = [float(x) for x in args.noises.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    use_mpmath = HAS_MPMATH and (not args.no_mpmath)
    if not use_mpmath and HAS_MPMATH:
        print("[fast-005] mpmath detected but disabled via --no_mpmath (using fast fallback).")
    elif use_mpmath:
        print("[fast-005] Using true ζ via mpmath (cached Zx per σ).")

    summary, rows, energy_means = run_amplified_sigma_sweep_fast(
        steps=args.steps,
        w=args.w,
        zeta_gains=zeta_gains,
        lrs=lrs,
        noises=noises,
        seeds=seeds,
        use_mpmath=use_mpmath,
        progress_every=args.progress_every,
    )

    save_csv("actual_experiment_005_fast_results.csv", rows)
    with open("actual_experiment_005_fast_summary.json", "w") as f:
        json.dump({"summary": summary, "params": vars(args), "mpmath": use_mpmath}, f, indent=2)

    # Plots
    plot_finalV_vs_sigma(summary, zeta_gains, out_path="actual_experiment_005_fast_finalV_vs_sigma.png")
    mid_zg = zeta_gains[len(zeta_gains) // 2]
    plot_energy_means(energy_means, mid_zg=mid_zg, out_path="actual_experiment_005_fast_energy_mean.png")

    print("\n=== Experiment 005 (FAST) Summary ===")
    print(json.dumps({"summary": summary, "params": vars(args), "mpmath": use_mpmath}, indent=2))
    print("Artifacts: actual_experiment_005_fast_results.csv, actual_experiment_005_fast_summary.json, actual_experiment_005_fast_finalV_vs_sigma.png, actual_experiment_005_fast_energy_mean.png")
    if HAS_MPMATH and not use_mpmath:
        print("Note: mpmath available but fast fallback was forced (--no_mpmath).")
    if not HAS_MPMATH:
        print("Note: mpmath not available, used synthetic zeta-like signal fallback.")


if __name__ == "__main__":
    main()