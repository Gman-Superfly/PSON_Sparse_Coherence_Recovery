import argparse
import csv
import json
from typing import Dict, List, Tuple, Callable, Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    import mpmath as mp
    HAS_MPMATH = True
except Exception:
    HAS_MPMATH = False


# =============================================================================
# AIR-TIGHT VALIDATION EXPERIMENT 001
# Goal:
# - Validate mechanism invariance by swapping ζ with alternative structured signals
# - Ablate couplings: phase vs amplitude, per-gap vs per-screen dependencies
# - Expectation: Wormhole + PSON + acceptance improves visibility across families
# Artifacts:
#  - airtight_experiments_001_results.csv
#  - airtight_experiments_001_summary.json
#  - airtight_experiments_001_pson_gain_bar.png
# =============================================================================


# -------------------------
# Shared optics primitives
# -------------------------
def first_25_primes() -> List[int]:
    return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]


def build_gaps_primes() -> List[float]:
    return [float(p * 10) for p in first_25_primes()]


def screen_and_theta() -> Tuple[np.ndarray, np.ndarray]:
    L = 1.0
    x_screen = np.linspace(-0.005, 0.005, 500)
    theta = x_screen / L
    return x_screen, theta


# Precompute screen geometry
_X_SCREEN, _THETA = screen_and_theta()


# -------------------------
# Structured signal sources
# -------------------------
def signal_zeta_per_screen(sigma: float, t_scale: float, use_mpmath: bool) -> np.ndarray:
    x = _X_SCREEN
    t_x = (x - float(x.min())) / (float(x.max()) - float(x.min()) + 1e-12) * t_scale
    if use_mpmath and HAS_MPMATH:
        vals = [float(mp.re(mp.zeta(sigma + 1j * float(tt)))) for tt in t_x]
        return np.asarray(vals, dtype=float)
    zeros = np.array([14.134725, 21.022040, 25.010857, 30.424876, 32.935062], dtype=float)
    zsig = np.sum(np.sin(2 * np.pi * np.outer(t_x, zeros)), axis=1)
    damp = 0.5 / max(1e-6, sigma)
    return (damp * zsig).astype(float)


def signal_zeta_per_gap(gaps_um: List[float], sigma: float, t_scale: float, use_mpmath: bool) -> np.ndarray:
    gaps = np.asarray(gaps_um, dtype=float)
    max_gap = float(np.max(gaps)) + 1e-8
    t = (gaps / max_gap) * t_scale
    if use_mpmath and HAS_MPMATH:
        vals = [float(mp.re(mp.zeta(sigma + 1j * float(tt)))) for tt in t]
        return np.asarray(vals, dtype=float)
    zeros = np.array([14.134725, 21.022040, 25.010857, 30.424876, 32.935062], dtype=float)
    zsig = np.sum(np.sin(2 * np.pi * np.outer(t, zeros)), axis=1)
    damp = 0.5 / max(1e-6, sigma)
    return (damp * zsig).astype(float)


def signal_sinmix(length: int, freqs: List[float], phases: Optional[List[float]] = None) -> np.ndarray:
    t = np.linspace(0.0, 1.0, length, endpoint=True)
    if phases is None:
        phases = [0.0] * len(freqs)
    acc = np.zeros_like(t)
    for f, ph in zip(freqs, phases):
        acc += np.sin(2 * np.pi * f * t + ph)
    return acc.astype(float)


def signal_one_over_f(length: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    # Generate real 1/f^beta noise via spectral shaping
    # Reference approach: random phases, magnitude ~ 1 / f^(beta/2), mirror symmetry for real signal
    n = length
    freqs = np.fft.rfftfreq(n)
    mag = np.ones_like(freqs)
    mag[1:] = 1.0 / (freqs[1:] ** (beta / 2.0))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=freqs.shape)
    spectrum = mag * np.exp(1j * phases)
    # Hermitian symmetry
    time = np.fft.irfft(spectrum, n=n)
    # Normalize
    time = (time - np.mean(time)) / (np.std(time) + 1e-12)
    return time.astype(float)


def signal_chirp(length: int, f0: float, f1: float, phase0: float = 0.0) -> np.ndarray:
    t = np.linspace(0.0, 1.0, length, endpoint=True)
    # Linear chirp phase integral: 2π ∫ (f0 + (f1 - f0) t) dt = 2π (f0 t + 0.5 (f1-f0) t^2)
    phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t * t) + phase0
    return np.sin(phase).astype(float)


def signal_turbulence_like(length: int, rng: np.random.Generator) -> np.ndarray:
    # Kolmogorov-esque spectrum ~ 1/f^(5/3)
    return signal_one_over_f(length=length, beta=5.0 / 3.0, rng=rng)


# -------------------------
# Precision, weights, energy
# -------------------------
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


def calculate_visibility(I: np.ndarray) -> float:
    I_max = float(np.max(I))
    I_min = float(np.min(I))
    return (I_max - I_min) / (I_max + I_min + 1e-8)


def energy_from_visibility(V: float) -> float:
    return (1.0 - V) ** 2


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


# -------------------------
# Simulation with ablations
# -------------------------
def simulate_intensity(
    gaps_um: List[float],
    phases: np.ndarray,
    coupling: str,           # "phase" | "amplitude"
    dependency: str,         # "per_gap" | "per_screen"
    signal_per_gap: Optional[np.ndarray],
    signal_per_screen: Optional[np.ndarray],
    phase_gain: float,
    amp_gain: float,
) -> np.ndarray:
    """
    Compute I(x) under ablated coupling/dependency using a shared two-slit model per gap.
    - phase coupling: phi += phase_gain * S
    - amplitude coupling: field2 amplitude *= (1 + amp_gain * S) [clipped >= 0]
    """
    assert len(gaps_um) == phases.shape[0], "phases must match gaps"
    lambda_nm = 633.0
    k = 2 * np.pi / (lambda_nm * 1e-9)
    theta = _THETA
    amp_per_slit = 0.5

    intensities = []
    for i, g_um in enumerate(gaps_um):
        d = g_um * 1e-6
        base_phi = k * d * np.sin(theta) + phases[i]

        if coupling == "phase":
            if dependency == "per_gap":
                assert signal_per_gap is not None, "signal_per_gap required for per_gap"
                phi = base_phi + phase_gain * signal_per_gap[i]
                field2_amp = amp_per_slit
                field2 = field2_amp * np.exp(1j * phi)
                field1 = amp_per_slit * np.exp(1j * 0.0)
            elif dependency == "per_screen":
                assert signal_per_screen is not None, "signal_per_screen required for per_screen"
                phi = base_phi + phase_gain * signal_per_screen
                field2_amp = amp_per_slit
                field2 = field2_amp * np.exp(1j * phi)
                field1 = amp_per_slit * np.exp(1j * 0.0)
            else:
                raise ValueError(f"Unknown dependency: {dependency}")
        elif coupling == "amplitude":
            if dependency == "per_gap":
                assert signal_per_gap is not None, "signal_per_gap required for per_gap"
                phi = base_phi
                amp2 = amp_per_slit * (1.0 + amp_gain * signal_per_gap[i])
                amp2 = np.clip(amp2, 0.0, None)
                field2 = amp2 * np.exp(1j * phi)
                field1 = amp_per_slit * np.exp(1j * 0.0)
            elif dependency == "per_screen":
                assert signal_per_screen is not None, "signal_per_screen required for per_screen"
                phi = base_phi
                amp2 = amp_per_slit * (1.0 + amp_gain * signal_per_screen)
                amp2 = np.clip(amp2, 0.0, None)
                field2 = amp2 * np.exp(1j * phi)
                field1 = amp_per_slit * np.exp(1j * 0.0)
            else:
                raise ValueError(f"Unknown dependency: {dependency}")
        else:
            raise ValueError(f"Unknown coupling: {coupling}")

        I = np.abs(field1 + field2) ** 2
        intensities.append(I)

    return np.mean(intensities, axis=0)


def run_homeostat_vector(
    gaps_um: List[float],
    steps: int,
    w: float,
    lr: float,
    noise_scale: float,
    use_pson: bool,
    seed: int,
    simulate_fn: Callable[[np.ndarray], np.ndarray],
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    d = len(gaps_um)
    phases = np.zeros(d, dtype=float)
    precision, weights = compute_precision_and_weights(gaps_um)

    energies: List[float] = []
    visibilities: List[float] = []
    accepted = 0
    attempts = 0

    I0 = simulate_fn(phases)
    V0 = calculate_visibility(I0)
    E0 = energy_from_visibility(V0)
    energies.append(E0)
    visibilities.append(V0)

    for _ in range(steps):
        I_cur = simulate_fn(phases)
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
        I_new = simulate_fn(candidate)
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
            I_det = simulate_fn(proposal)
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


# -------------------------
# Scenario runner
# -------------------------
def run_invariance_grid(
    steps: int,
    w: float,
    lr: float,
    noise: float,
    seeds: List[int],
    signals: List[str],
    couplings: List[str],
    dependencies: List[str],
    phase_gain: float,
    amp_gain: float,
    use_mpmath: bool,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, float]]]:
    gaps = build_gaps_primes()
    rng = np.random.default_rng(12345)

    rows: List[Dict[str, object]] = []
    summary: Dict[str, Dict[str, float]] = {}

    for signal in signals:
        for coupling in couplings:
            for dependency in dependencies:
                key = f"{signal}|{coupling}|{dependency}"
                final_no_pson: List[float] = []
                final_pson: List[float] = []
                df90_no_pson: List[int] = []
                df90_pson: List[int] = []
                acc_no_pson: List[float] = []
                acc_pson: List[float] = []

                # Prepare signal sources
                if dependency == "per_screen":
                    if signal == "zeta":
                        Sx = signal_zeta_per_screen(sigma=0.55, t_scale=10.0, use_mpmath=use_mpmath)
                    elif signal == "sinmix":
                        freqs = [np.sqrt(2.0), np.pi, np.e, 14.134725, 21.022040]
                        Sx = signal_sinmix(length=_X_SCREEN.shape[0], freqs=freqs, phases=None)
                    elif signal == "one_over_f":
                        Sx = signal_one_over_f(length=_X_SCREEN.shape[0], beta=1.0, rng=rng)
                    elif signal == "chirp":
                        Sx = signal_chirp(length=_X_SCREEN.shape[0], f0=3.0, f1=50.0, phase0=0.0)
                    elif signal == "turbulence":
                        Sx = signal_turbulence_like(length=_X_SCREEN.shape[0], rng=rng)
                    else:
                        raise ValueError(f"Unknown signal: {signal}")
                    Si = None
                else:
                    if signal == "zeta":
                        Si = signal_zeta_per_gap(gaps_um=gaps, sigma=0.55, t_scale=10.0, use_mpmath=use_mpmath)
                    elif signal == "sinmix":
                        freqs = [np.sqrt(3.0), np.sqrt(5.0), 14.134725, 21.022040]
                        Si = signal_sinmix(length=len(gaps), freqs=freqs, phases=None)
                    elif signal == "one_over_f":
                        Si = signal_one_over_f(length=len(gaps), beta=1.0, rng=rng)
                    elif signal == "chirp":
                        Si = signal_chirp(length=len(gaps), f0=2.0, f1=20.0, phase0=0.0)
                    elif signal == "turbulence":
                        Si = signal_turbulence_like(length=len(gaps), rng=rng)
                    else:
                        raise ValueError(f"Unknown signal: {signal}")
                    Sx = None

                # Normalize signals to unit std for stable gains
                if Sx is not None:
                    Sx = (Sx - float(np.mean(Sx))) / (float(np.std(Sx) + 1e-12))
                if Si is not None:
                    Si = (Si - float(np.mean(Si))) / (float(np.std(Si) + 1e-12))

                def make_sim_fn(dep: str, coup: str, Sx_arr: Optional[np.ndarray], Si_arr: Optional[np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
                    def _inner(phases: np.ndarray) -> np.ndarray:
                        return simulate_intensity(
                            gaps_um=gaps,
                            phases=phases,
                            coupling=coup,
                            dependency=dep,
                            signal_per_gap=Si_arr,
                            signal_per_screen=Sx_arr,
                            phase_gain=phase_gain,
                            amp_gain=amp_gain,
                        )
                    return _inner

                sim_fn = make_sim_fn(dependency, coupling, Sx, Si)

                for seed in seeds:
                    res_n = run_homeostat_vector(
                        gaps_um=gaps,
                        steps=steps,
                        w=w,
                        lr=lr,
                        noise_scale=noise,
                        use_pson=False,
                        seed=seed,
                        simulate_fn=sim_fn,
                    )
                    res_p = run_homeostat_vector(
                        gaps_um=gaps,
                        steps=steps,
                        w=w,
                        lr=lr,
                        noise_scale=noise,
                        use_pson=True,
                        seed=seed,
                        simulate_fn=sim_fn,
                    )

                    df90_n = delta_f90_steps(res_n["energies"])
                    df90_p = delta_f90_steps(res_p["energies"])

                    rows.append({
                        "signal": signal,
                        "coupling": coupling,
                        "dependency": dependency,
                        "seed": seed,
                        "final_V_no_pson": res_n["final_V"],
                        "final_V_pson": res_p["final_V"],
                        "pson_gain": res_p["final_V"] - res_n["final_V"],
                        "deltaF90_no_pson": df90_n,
                        "deltaF90_pson": df90_p,
                        "accept_rate_no_pson": res_n["accept_rate"],
                        "accept_rate_pson": res_p["accept_rate"],
                    })

                    final_no_pson.append(res_n["final_V"])
                    final_pson.append(res_p["final_V"])
                    df90_no_pson.append(df90_n)
                    df90_pson.append(df90_p)
                    acc_no_pson.append(res_n["accept_rate"])
                    acc_pson.append(res_p["accept_rate"])

                f_n = np.array(final_no_pson, dtype=float)
                f_p = np.array(final_pson, dtype=float)
                d_n = np.array([d if d >= 0 else steps for d in df90_no_pson], dtype=float)
                d_p = np.array([d if d >= 0 else steps for d in df90_pson], dtype=float)
                a_n = np.array(acc_no_pson, dtype=float)
                a_p = np.array(acc_pson, dtype=float)

                summary[key] = {
                    "mean_final_V_no_pson": float(f_n.mean()) if f_n.size else 0.0,
                    "mean_final_V_pson": float(f_p.mean()) if f_p.size else 0.0,
                    "mean_pson_gain": float(f_p.mean() - f_n.mean()) if f_p.size and f_n.size else 0.0,
                    "mean_deltaF90_no_pson": float(d_n.mean()) if d_n.size else float(steps),
                    "mean_deltaF90_pson": float(d_p.mean()) if d_p.size else float(steps),
                    "mean_accept_no_pson": float(a_n.mean()) if a_n.size else 0.0,
                    "mean_accept_pson": float(a_p.mean()) if a_p.size else 0.0,
                }

    return rows, summary


def save_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def plot_pson_gain_bar(summary: Dict[str, Dict[str, float]], out_path: str) -> None:
    keys = sorted(summary.keys())
    gains = [summary[k]["mean_pson_gain"] for k in keys]
    plt.figure(figsize=(12, 6))
    x = np.arange(len(keys))
    plt.bar(x, gains, color="#4C78A8")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(x, keys, rotation=45, ha="right")
    plt.ylabel("PSON Gain: mean(final_V_pson - final_V_no_pson)")
    plt.title("Airtight 001: PSON Gain Across Signal/Coupling/Dependency")
    plt.tight_layout()
    plt.savefig(out_path)


def main():
    parser = argparse.ArgumentParser(description="Airtight Experiment 001: Invariance validations across structured signals and couplings")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--w", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--signals", type=str, default="zeta,sinmix,one_over_f,chirp,turbulence")
    parser.add_argument("--couplings", type=str, default="phase,amplitude")
    parser.add_argument("--dependencies", type=str, default="per_gap,per_screen")
    parser.add_argument("--phase_gain", type=float, default=0.5)
    parser.add_argument("--amp_gain", type=float, default=0.2)
    parser.add_argument("--no_mpmath", action="store_true", help="Force synthetic ζ-like fallback instead of true ζ for zeta signal")
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    signals = [s.strip() for s in args.signals.split(",") if s.strip()]
    couplings = [s.strip() for s in args.couplings.split(",") if s.strip()]
    dependencies = [s.strip() for s in args.dependencies.split(",") if s.strip()]

    use_mpmath = HAS_MPMATH and (not args.no_mpmath)
    if "zeta" in signals:
        if use_mpmath:
            print("[airtight-001] Using true ζ via mpmath for zeta signal.")
        else:
            print("[airtight-001] ζ will use synthetic fallback (no mpmath or disabled).")

    rows, summary = run_invariance_grid(
        steps=args.steps,
        w=args.w,
        lr=args.lr,
        noise=args.noise,
        seeds=seeds,
        signals=signals,
        couplings=couplings,
        dependencies=dependencies,
        phase_gain=args.phase_gain,
        amp_gain=args.amp_gain,
        use_mpmath=use_mpmath,
    )

    save_csv("airtight_experiments_001_results.csv", rows)
    out = {
        "summary": summary,
        "params": {
            "steps": args.steps,
            "w": args.w,
            "lr": args.lr,
            "noise": args.noise,
            "seeds": seeds,
            "signals": signals,
            "couplings": couplings,
            "dependencies": dependencies,
            "phase_gain": args.phase_gain,
            "amp_gain": args.amp_gain,
            "mpmath": use_mpmath,
        },
    }
    with open("airtight_experiments_001_summary.json", "w") as f:
        json.dump(out, f, indent=2)

    plot_pson_gain_bar(summary, out_path="airtight_experiments_001_pson_gain_bar.png")

    print("\n=== Airtight Experiment 001 Summary ===")
    print(json.dumps(out, indent=2))
    print("Artifacts: airtight_experiments_001_results.csv, airtight_experiments_001_summary.json, airtight_experiments_001_pson_gain_bar.png")


if __name__ == "__main__":
    main()


