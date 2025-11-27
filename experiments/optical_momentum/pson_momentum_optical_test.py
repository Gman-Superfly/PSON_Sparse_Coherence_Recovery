"""
PSON+Momentum Test on Full Optical Problem
==========================================

Tests whether momentum improves PSON on the full sparse optical coherence
problem with all signal types and coupling modes.

Usage:
    uv run python experiments/pson_momentum_optical_test.py
"""

import json
from typing import Dict, List, Tuple, Callable, Optional
import numpy as np

# Import from airtight experiments
from airtight_experiments_001 import (
    build_gaps_primes,
    compute_precision_and_weights,
    calculate_visibility,
    energy_from_visibility,
    project_noise_metric_orthogonal,
    simulate_intensity,
    signal_zeta_per_screen,
    signal_zeta_per_gap,
    signal_sinmix,
    signal_one_over_f,
    signal_chirp,
    signal_turbulence_like,
    _X_SCREEN,
    HAS_MPMATH,
)


def run_homeostat_with_momentum(
    gaps_um: List[float],
    steps: int,
    w: float,
    lr: float,
    noise_scale: float,
    use_pson: bool,
    use_momentum: bool,
    momentum_beta: float,
    seed: int,
    simulate_fn: Callable[[np.ndarray], np.ndarray],
) -> Dict[str, object]:
    """Run homeostat with optional momentum."""
    rng = np.random.default_rng(seed)
    d = len(gaps_um)
    phases = np.zeros(d, dtype=float)
    precision, weights = compute_precision_and_weights(gaps_um)
    
    # Momentum velocity
    velocity = np.zeros(d, dtype=float) if use_momentum else None

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
        
        # Compute step with optional momentum
        step = lr * grad
        if use_momentum and velocity is not None:
            velocity = momentum_beta * velocity + step
            step = velocity
        
        proposal = phases - step

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


def run_momentum_comparison(
    steps: int = 200,
    seeds: List[int] = [42, 123, 456, 789, 1000],
    signals: List[str] = ["zeta", "sinmix", "one_over_f", "chirp", "turbulence"],
    use_mpmath: bool = True,
) -> Dict:
    """Compare PSON vs PSON+Momentum across all scenarios."""
    
    gaps = build_gaps_primes()
    rng = np.random.default_rng(12345)
    
    # Hyperparameters
    w = 0.2
    lr = 0.1
    noise = 0.02
    phase_gain = 0.5
    amp_gain = 0.2
    momentum_beta = 0.9
    
    results = {
        "pson_baseline": [],
        "pson_momentum": [],
    }
    
    couplings = ["phase", "amplitude"]
    dependencies = ["per_gap", "per_screen"]
    
    print("=" * 70)
    print("PSON vs PSON+Momentum on Full Optical Problem")
    print("=" * 70)
    print(f"Steps: {steps}, Seeds: {seeds}, Signals: {signals}")
    print()
    
    total_scenarios = len(signals) * len(couplings) * len(dependencies)
    scenario_num = 0
    
    for signal in signals:
        for coupling in couplings:
            for dependency in dependencies:
                scenario_num += 1
                key = f"{signal}|{coupling}|{dependency}"
                
                # Prepare signal sources
                if dependency == "per_screen":
                    if signal == "zeta":
                        Sx = signal_zeta_per_screen(sigma=0.55, t_scale=10.0, use_mpmath=use_mpmath and HAS_MPMATH)
                    elif signal == "sinmix":
                        freqs = [np.sqrt(2.0), np.pi, np.e, 14.134725, 21.022040]
                        Sx = signal_sinmix(length=_X_SCREEN.shape[0], freqs=freqs, phases=None)
                    elif signal == "one_over_f":
                        Sx = signal_one_over_f(length=_X_SCREEN.shape[0], beta=1.0, rng=rng)
                    elif signal == "chirp":
                        Sx = signal_chirp(length=_X_SCREEN.shape[0], f0=3.0, f1=50.0, phase0=0.0)
                    elif signal == "turbulence":
                        Sx = signal_turbulence_like(length=_X_SCREEN.shape[0], rng=rng)
                    Si = None
                else:
                    if signal == "zeta":
                        Si = signal_zeta_per_gap(gaps_um=gaps, sigma=0.55, t_scale=10.0, use_mpmath=use_mpmath and HAS_MPMATH)
                    elif signal == "sinmix":
                        freqs = [np.sqrt(3.0), np.sqrt(5.0), 14.134725, 21.022040]
                        Si = signal_sinmix(length=len(gaps), freqs=freqs, phases=None)
                    elif signal == "one_over_f":
                        Si = signal_one_over_f(length=len(gaps), beta=1.0, rng=rng)
                    elif signal == "chirp":
                        Si = signal_chirp(length=len(gaps), f0=2.0, f1=20.0, phase0=0.0)
                    elif signal == "turbulence":
                        Si = signal_turbulence_like(length=len(gaps), rng=rng)
                    Sx = None
                
                # Normalize signals
                if Sx is not None:
                    Sx = (Sx - float(np.mean(Sx))) / (float(np.std(Sx) + 1e-12))
                if Si is not None:
                    Si = (Si - float(np.mean(Si))) / (float(np.std(Si) + 1e-12))
                
                def make_sim_fn(dep, coup, Sx_arr, Si_arr):
                    def _inner(phases):
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
                
                baseline_vis = []
                momentum_vis = []
                
                for seed in seeds:
                    # PSON baseline
                    res_base = run_homeostat_with_momentum(
                        gaps_um=gaps,
                        steps=steps,
                        w=w,
                        lr=lr,
                        noise_scale=noise,
                        use_pson=True,
                        use_momentum=False,
                        momentum_beta=0.0,
                        seed=seed,
                        simulate_fn=sim_fn,
                    )
                    baseline_vis.append(res_base["final_V"])
                    
                    # PSON + Momentum
                    res_mom = run_homeostat_with_momentum(
                        gaps_um=gaps,
                        steps=steps,
                        w=w,
                        lr=lr,
                        noise_scale=noise,
                        use_pson=True,
                        use_momentum=True,
                        momentum_beta=momentum_beta,
                        seed=seed,
                        simulate_fn=sim_fn,
                    )
                    momentum_vis.append(res_mom["final_V"])
                
                mean_base = np.mean(baseline_vis)
                mean_mom = np.mean(momentum_vis)
                improvement = (mean_mom - mean_base) / (mean_base + 1e-8) * 100
                
                results["pson_baseline"].append({
                    "scenario": key,
                    "mean_V": float(mean_base),
                    "std_V": float(np.std(baseline_vis)),
                })
                results["pson_momentum"].append({
                    "scenario": key,
                    "mean_V": float(mean_mom),
                    "std_V": float(np.std(momentum_vis)),
                })
                
                winner = "Momentum" if mean_mom > mean_base else "Baseline"
                print(f"  [{scenario_num}/{total_scenarios}] {key}")
                print(f"      Baseline: {mean_base:.4f}, +Momentum: {mean_mom:.4f}, Change={improvement:+.1f}% ({winner})")
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_base = [r["mean_V"] for r in results["pson_baseline"]]
    all_mom = [r["mean_V"] for r in results["pson_momentum"]]
    
    mean_base = np.mean(all_base)
    mean_mom = np.mean(all_mom)
    overall_improvement = (mean_mom - mean_base) / (mean_base + 1e-8) * 100
    
    wins_momentum = sum(1 for b, m in zip(all_base, all_mom) if m > b)
    wins_baseline = sum(1 for b, m in zip(all_base, all_mom) if b > m)
    ties = sum(1 for b, m in zip(all_base, all_mom) if abs(m - b) < 1e-6)
    
    print(f"Overall PSON Baseline:    {mean_base:.4f}")
    print(f"Overall PSON+Momentum:    {mean_mom:.4f}")
    print(f"Overall Improvement:      {overall_improvement:+.2f}%")
    print()
    print(f"Momentum wins: {wins_momentum}/{total_scenarios}")
    print(f"Baseline wins: {wins_baseline}/{total_scenarios}")
    print(f"Ties:          {ties}/{total_scenarios}")
    
    results["summary"] = {
        "mean_baseline_V": float(mean_base),
        "mean_momentum_V": float(mean_mom),
        "overall_improvement_pct": float(overall_improvement),
        "momentum_wins": wins_momentum,
        "baseline_wins": wins_baseline,
        "ties": ties,
        "total_scenarios": total_scenarios,
    }
    
    # Save results
    with open("pson_momentum_optical_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved: pson_momentum_optical_results.json")
    
    return results


if __name__ == "__main__":
    run_momentum_comparison()

