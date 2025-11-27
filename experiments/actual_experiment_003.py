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
# ACTUAL EXPERIMENT 003
# Hyperparameter tuning on primes: zeta_gain, lr, noise
# Vector Homeostat (Wormhole + PSON, down-only acceptance)
# Outputs:
#  - JSON summary for best config
#  - CSV of all runs
#  - Plot of energy curves for best config (PSON vs no-PSON) + baseline marker
# =============================================================================


def first_25_primes() -> List[int]:
    return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]


def build_gaps_primes() -> List[float]:
    return [float(p * 10) for p in first_25_primes()]


def compute_zeta_re_for_gaps(gaps_um: List[float], t_scale: float = 10.0) -> np.ndarray:
    gaps = np.asarray(gaps_um, dtype=float)
    max_gap = float(np.max(gaps)) + 1e-8
    t = (gaps / max_gap) * t_scale
    if HAS_MPMATH:
        vals = [float(mp.re(mp.zeta(0.5 + 1j * float(tt)))) for tt in t]
        return np.asarray(vals, dtype=float)
    zeros = np.array([14.134725, 21.022040, 25.010857, 30.424876, 32.935062], dtype=float)
    zsig = np.sum(np.sin(2 * np.pi * np.outer(t, zeros)), axis=1)
    return zsig.astype(float)


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


def run_grid(
    steps: int,
    w: float,
    zeta_gains: List[float],
    lrs: List[float],
    noises: List[float],
    seed: int,
) -> Tuple[Dict[str, float], List[Dict[str, float]], Dict[str, List[float]]]:
    gaps = build_gaps_primes()
    zeta_re = compute_zeta_re_for_gaps(gaps, t_scale=10.0)

    # Baseline
    I_base = simulate_interference_zeta(gaps, np.zeros(len(gaps)), zeta_re, zeta_gain=0.2)
    V_base = calculate_visibility(I_base)

    records: List[Dict[str, float]] = []
    best_score = -1e9
    best = None
    best_energies_no = None
    best_energies_ps = None

    for zg in zeta_gains:
        for lr in lrs:
            for nz in noises:
                res_no = run_homeostat_vector_zeta(
                    gaps, zeta_re, zg, steps, w, lr, nz, use_pson=False, seed=seed
                )
                res_ps = run_homeostat_vector_zeta(
                    gaps, zeta_re, zg, steps, w, lr, nz, use_pson=True, seed=seed
                )
                df90_no = delta_f90_steps(res_no["energies"])
                df90_ps = delta_f90_steps(res_ps["energies"])
                rec = {
                    "zeta_gain": zg,
                    "lr": lr,
                    "noise": nz,
                    "baseline_V": V_base,
                    "final_V_no_pson": res_no["final_V"],
                    "final_V_pson": res_ps["final_V"],
                    "deltaF90_no_pson": df90_no,
                    "deltaF90_pson": df90_ps,
                    "accept_rate_no_pson": res_no["accept_rate"],
                    "accept_rate_pson": res_ps["accept_rate"],
                }
                records.append(rec)

                # Scoring: prioritize high final_V_pson, then lower ΔF90_pson
                score = rec["final_V_pson"] - 0.2 * (
                    (rec["deltaF90_pson"] if rec["deltaF90_pson"] >= 0 else steps) / max(1, steps)
                )
                if score > best_score:
                    best_score = score
                    best = rec
                    best_energies_no = res_no["energies"]
                    best_energies_ps = res_ps["energies"]

    detail = {
        "energies_no_pson": best_energies_no if best_energies_no is not None else [],
        "energies_pson": best_energies_ps if best_energies_ps is not None else [],
    }
    return best, records, detail


def save_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def plot_best(energies_no: List[float], energies_ps: List[float]) -> None:
    plt.figure(figsize=(10, 6))
    if energies_no:
        plt.plot(energies_no, linestyle="--", alpha=0.85, label="no PSON")
    if energies_ps:
        plt.plot(energies_ps, linewidth=2, alpha=0.9, label="PSON")
    plt.xlabel("Step")
    plt.ylabel("Energy (1 - V)^2")
    plt.title("Experiment 003: Best config energy (PSON vs no-PSON)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("actual_experiment_003_best_energy.png")


def main():
    parser = argparse.ArgumentParser(description="Experiment 003: Tune zeta_gain, lr, noise on primes")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--w", type=float, default=0.2)
    parser.add_argument("--zeta_gains", type=str, default="0.1,0.2,0.3")
    parser.add_argument("--lrs", type=str, default="0.05,0.1,0.15")
    parser.add_argument("--noises", type=str, default="0.01,0.02,0.03")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    zeta_gains = [float(x) for x in args.zeta_gains.split(",") if x.strip()]
    lrs = [float(x) for x in args.lrs.split(",") if x.strip()]
    noises = [float(x) for x in args.noises.split(",") if x.strip()]

    best, rows, detail = run_grid(
        steps=args.steps,
        w=args.w,
        zeta_gains=zeta_gains,
        lrs=lrs,
        noises=noises,
        seed=args.seed,
    )

    save_csv("actual_experiment_003_results.csv", rows)
    plot_best(detail["energies_no_pson"], detail["energies_pson"])

    out = {
        "best": best,
        "grid_size": len(rows),
        "note": "Score = final_V_pson - 0.2 * (ΔF90_pson/steps); higher is better",
        "mpmath": HAS_MPMATH,
    }
    with open("actual_experiment_003_summary.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Experiment 003 Best Config ===")
    print(json.dumps(out, indent=2))
    print("Artifacts: actual_experiment_003_results.csv, actual_experiment_003_best_energy.png, actual_experiment_003_summary.json")
    if not HAS_MPMATH:
        print("Note: mpmath not available, used synthetic zeta-like signal fallback.")


if __name__ == "__main__":
    main()


