"""
Multi-Seed Validation Experiment 001
====================================
Run the main PSON ablation across multiple random seeds to establish
statistical significance with confidence intervals.

This addresses the reviewer concern: "Results shown for seed=42 only"

Artifacts:
  - multi_seed_validation_001_results.csv
  - multi_seed_validation_001_summary.json
  - multi_seed_validation_001_box.png (box plots)
  - multi_seed_validation_001_gain_dist.png (gain distribution)
"""

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
# Shared Optics Primitives
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


_X_SCREEN, _THETA = screen_and_theta()


# =============================================================================
# Signal Generators
# =============================================================================

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


def signal_sinmix(length: int, freqs: List[float]) -> np.ndarray:
    t = np.linspace(0.0, 1.0, length, endpoint=True)
    acc = np.zeros_like(t)
    for f in freqs:
        acc += np.sin(2 * np.pi * f * t)
    return acc.astype(float)


def signal_one_over_f(length: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    n = length
    freqs = np.fft.rfftfreq(n)
    mag = np.ones_like(freqs)
    mag[1:] = 1.0 / (freqs[1:] ** (beta / 2.0))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=freqs.shape)
    spectrum = mag * np.exp(1j * phases)
    time = np.fft.irfft(spectrum, n=n)
    time = (time - np.mean(time)) / (np.std(time) + 1e-12)
    return time.astype(float)


def signal_chirp(length: int, f0: float, f1: float) -> np.ndarray:
    t = np.linspace(0.0, 1.0, length, endpoint=True)
    phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t * t)
    return np.sin(phase).astype(float)


def signal_turbulence(length: int, rng: np.random.Generator) -> np.ndarray:
    return signal_one_over_f(length=length, beta=5.0 / 3.0, rng=rng)


# =============================================================================
# Core Functions
# =============================================================================

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


def simulate_intensity(
    gaps_um: List[float],
    phases: np.ndarray,
    signal_per_gap: Optional[np.ndarray],
    phase_gain: float,
) -> np.ndarray:
    """Simplified intensity simulation with phase coupling per-gap."""
    assert len(gaps_um) == phases.shape[0]
    lambda_nm = 633.0
    k = 2 * np.pi / (lambda_nm * 1e-9)
    theta = _THETA
    amp_per_slit = 0.5

    intensities = []
    for i, g_um in enumerate(gaps_um):
        d = g_um * 1e-6
        base_phi = k * d * np.sin(theta) + phases[i]
        if signal_per_gap is not None:
            phi = base_phi + phase_gain * signal_per_gap[i]
        else:
            phi = base_phi
        field1 = amp_per_slit * np.exp(1j * 0.0)
        field2 = amp_per_slit * np.exp(1j * phi)
        I = np.abs(field1 + field2) ** 2
        intensities.append(I)

    return np.mean(intensities, axis=0)


def run_homeostat(
    gaps_um: List[float],
    steps: int,
    w: float,
    lr: float,
    noise_scale: float,
    use_pson: bool,
    seed: int,
    signal_per_gap: Optional[np.ndarray],
    phase_gain: float,
) -> Dict[str, float]:
    """Run homeostat and return final metrics."""
    rng = np.random.default_rng(seed)
    d = len(gaps_um)
    phases = np.zeros(d, dtype=float)
    precision, weights = compute_precision_and_weights(gaps_um)

    accepted = 0
    attempts = 0

    # Initial
    I0 = simulate_intensity(gaps_um, phases, signal_per_gap, phase_gain)
    V0 = calculate_visibility(I0)

    for _ in range(steps):
        I_cur = simulate_intensity(gaps_um, phases, signal_per_gap, phase_gain)
        V_cur = calculate_visibility(I_cur)
        E_cur = energy_from_visibility(V_cur)

        grad = -w * E_cur * weights
        proposal = phases - lr * grad

        if use_pson:
            delta_perp = project_noise_metric_orthogonal(grad=grad, precision=precision, rng=rng)
            noise = (delta_perp / (np.sqrt(precision) + 1e-12)) * noise_scale
            candidate = proposal + noise
        else:
            candidate = proposal

        attempts += 1
        I_new = simulate_intensity(gaps_um, candidate, signal_per_gap, phase_gain)
        V_new = calculate_visibility(I_new)
        E_new = energy_from_visibility(V_new)

        if E_new <= E_cur:
            phases = candidate
            accepted += 1
            continue

        if use_pson:
            attempts += 1
            I_det = simulate_intensity(gaps_um, proposal, signal_per_gap, phase_gain)
            V_det = calculate_visibility(I_det)
            E_det = energy_from_visibility(V_det)
            if E_det <= E_cur:
                phases = proposal
                accepted += 1
                continue

    # Final
    I_final = simulate_intensity(gaps_um, phases, signal_per_gap, phase_gain)
    V_final = calculate_visibility(I_final)

    return {
        "init_V": V0,
        "final_V": V_final,
        "accept_rate": accepted / attempts if attempts > 0 else 0.0,
    }


# =============================================================================
# Multi-Seed Experiment
# =============================================================================

def run_multi_seed_experiment(
    steps: int,
    w: float,
    lr: float,
    noise: float,
    seeds: List[int],
    signals: List[str],
    phase_gain: float,
    use_mpmath: bool,
) -> Tuple[List[Dict], Dict]:
    """Run experiment across multiple seeds and signals."""
    gaps = build_gaps_primes()
    signal_rng = np.random.default_rng(12345)  # Fixed for signal generation
    
    rows: List[Dict] = []
    
    for signal_name in signals:
        # Generate signal
        if signal_name == "zeta":
            Si = signal_zeta_per_gap(gaps, sigma=0.55, t_scale=10.0, use_mpmath=use_mpmath)
        elif signal_name == "sinmix":
            Si = signal_sinmix(len(gaps), freqs=[np.sqrt(3.0), np.sqrt(5.0), 14.134725])
        elif signal_name == "one_over_f":
            Si = signal_one_over_f(len(gaps), beta=1.0, rng=signal_rng)
        elif signal_name == "chirp":
            Si = signal_chirp(len(gaps), f0=2.0, f1=20.0)
        elif signal_name == "turbulence":
            Si = signal_turbulence(len(gaps), rng=signal_rng)
        else:
            Si = None
        
        # Normalize signal
        if Si is not None:
            Si = (Si - float(np.mean(Si))) / (float(np.std(Si)) + 1e-12)
        
        for seed in seeds:
            # Run without PSON
            res_no = run_homeostat(
                gaps_um=gaps,
                steps=steps,
                w=w,
                lr=lr,
                noise_scale=noise,
                use_pson=False,
                seed=seed,
                signal_per_gap=Si,
                phase_gain=phase_gain,
            )
            
            # Run with PSON
            res_pson = run_homeostat(
                gaps_um=gaps,
                steps=steps,
                w=w,
                lr=lr,
                noise_scale=noise,
                use_pson=True,
                seed=seed,
                signal_per_gap=Si,
                phase_gain=phase_gain,
            )
            
            rows.append({
                "signal": signal_name,
                "seed": seed,
                "final_V_no_pson": res_no["final_V"],
                "final_V_pson": res_pson["final_V"],
                "pson_gain": res_pson["final_V"] - res_no["final_V"],
                "accept_rate_pson": res_pson["accept_rate"],
            })
    
    # Compute summary statistics
    summary: Dict[str, Dict[str, float]] = {}
    
    for signal_name in signals:
        signal_rows = [r for r in rows if r["signal"] == signal_name]
        gains = [r["pson_gain"] for r in signal_rows]
        vis_pson = [r["final_V_pson"] for r in signal_rows]
        vis_no = [r["final_V_no_pson"] for r in signal_rows]
        
        summary[signal_name] = {
            "mean_gain": float(np.mean(gains)),
            "std_gain": float(np.std(gains)),
            "min_gain": float(np.min(gains)),
            "max_gain": float(np.max(gains)),
            "mean_V_pson": float(np.mean(vis_pson)),
            "std_V_pson": float(np.std(vis_pson)),
            "mean_V_no_pson": float(np.mean(vis_no)),
            "std_V_no_pson": float(np.std(vis_no)),
            "win_rate": float(np.mean([1 if g > 0 else 0 for g in gains])),
            "n_seeds": len(gains),
        }
    
    # Overall statistics
    all_gains = [r["pson_gain"] for r in rows]
    summary["_overall"] = {
        "mean_gain": float(np.mean(all_gains)),
        "std_gain": float(np.std(all_gains)),
        "min_gain": float(np.min(all_gains)),
        "max_gain": float(np.max(all_gains)),
        "ci_95_low": float(np.percentile(all_gains, 2.5)),
        "ci_95_high": float(np.percentile(all_gains, 97.5)),
        "win_rate": float(np.mean([1 if g > 0 else 0 for g in all_gains])),
        "total_runs": len(all_gains),
    }
    
    return rows, summary


def save_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def plot_box(rows: List[Dict], signals: List[str], out_path: str) -> None:
    """Box plot of PSON gains per signal type."""
    data = []
    labels = []
    for signal_name in signals:
        gains = [r["pson_gain"] for r in rows if r["signal"] == signal_name]
        data.append(gains)
        labels.append(signal_name.capitalize())
    
    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(data, labels=labels, patch_artist=True)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(signals)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.ylabel("PSON Gain (Visibility)")
    plt.title("PSON Gain Distribution by Signal Type (Multi-Seed)")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_gain_hist(rows: List[Dict], out_path: str) -> None:
    """Histogram of all PSON gains."""
    gains = [r["pson_gain"] for r in rows]
    
    plt.figure(figsize=(10, 6))
    plt.hist(gains, bins=30, color='#3498db', edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero gain')
    plt.axvline(x=np.mean(gains), color='green', linestyle='-', linewidth=2, 
                label=f'Mean: {np.mean(gains):.3f}')
    
    plt.xlabel("PSON Gain (Visibility)")
    plt.ylabel("Count")
    plt.title(f"Distribution of PSON Gains (n={len(gains)} runs)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Validation Experiment")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--w", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--seeds", type=str, default="42,123,456,789,1000,2024,3141,4242,5555,6789",
                        help="Comma-separated seeds (default: 10 seeds)")
    parser.add_argument("--signals", type=str, default="zeta,sinmix,one_over_f,chirp,turbulence")
    parser.add_argument("--phase_gain", type=float, default=0.5)
    parser.add_argument("--no_mpmath", action="store_true")
    args = parser.parse_args()
    
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    signals = [s.strip() for s in args.signals.split(",") if s.strip()]
    use_mpmath = HAS_MPMATH and (not args.no_mpmath)
    
    print(f"[multi-seed] Running with {len(seeds)} seeds × {len(signals)} signals = {len(seeds) * len(signals)} total runs")
    print(f"[multi-seed] Seeds: {seeds}")
    print(f"[multi-seed] Signals: {signals}")
    
    rows, summary = run_multi_seed_experiment(
        steps=args.steps,
        w=args.w,
        lr=args.lr,
        noise=args.noise,
        seeds=seeds,
        signals=signals,
        phase_gain=args.phase_gain,
        use_mpmath=use_mpmath,
    )
    
    # Save artifacts
    save_csv("multi_seed_validation_001_results.csv", rows)
    
    out = {
        "summary": summary,
        "params": {
            "steps": args.steps,
            "w": args.w,
            "lr": args.lr,
            "noise": args.noise,
            "seeds": seeds,
            "signals": signals,
            "phase_gain": args.phase_gain,
            "mpmath": use_mpmath,
        },
    }
    with open("multi_seed_validation_001_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    
    plot_box(rows, signals, "multi_seed_validation_001_box.png")
    plot_gain_hist(rows, "multi_seed_validation_001_gain_dist.png")
    
    # Print summary
    print("\n=== Multi-Seed Validation Summary ===")
    print(f"\n{'Signal':<15} {'Mean Gain':>10} {'Std':>8} {'Min':>8} {'Max':>8} {'Win%':>8}")
    print("-" * 60)
    for signal_name in signals:
        s = summary[signal_name]
        print(f"{signal_name:<15} {s['mean_gain']:>10.4f} {s['std_gain']:>8.4f} "
              f"{s['min_gain']:>8.4f} {s['max_gain']:>8.4f} {s['win_rate']*100:>7.1f}%")
    
    overall = summary["_overall"]
    print("-" * 60)
    print(f"{'OVERALL':<15} {overall['mean_gain']:>10.4f} {overall['std_gain']:>8.4f} "
          f"{overall['min_gain']:>8.4f} {overall['max_gain']:>8.4f} {overall['win_rate']*100:>7.1f}%")
    print(f"\n95% CI for PSON gain: [{overall['ci_95_low']:.4f}, {overall['ci_95_high']:.4f}]")
    print(f"Total runs: {overall['total_runs']}")
    
    print("\nArtifacts saved:")
    print("  - multi_seed_validation_001_results.csv")
    print("  - multi_seed_validation_001_summary.json")
    print("  - multi_seed_validation_001_box.png")
    print("  - multi_seed_validation_001_gain_dist.png")


if __name__ == "__main__":
    main()

