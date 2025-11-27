"""
Partial Observability Test 001
==============================
Demonstrate PSON's advantage under degraded observability conditions.

Key insight: CMA-ES excels when you have clean, accurate function evaluations.
PSON is designed for partial observability where:
  - Only global scalar feedback is available (no local gradients)
  - Observations may be noisy, delayed, or quantized
  - The landscape is aliased/non-convex

This experiment tests PSON vs CMA-ES under progressively degraded observability:
  1. Observation noise (additive Gaussian noise on energy measurement)
  2. Quantized feedback (discrete levels instead of continuous)
  3. Sparse feedback (evaluate every N steps, hold stale value between)
  4. Combined degradation (all of the above)

Expected outcome: PSON's relative performance improves as observability degrades.

Artifacts:
  - partial_observability_001_results.csv
  - partial_observability_001_summary.json
  - partial_observability_001_degradation.png
"""

import argparse
import csv
import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False
    print("[partial-obs] CMA-ES not available (pip install cma)")


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


def simulate_interference_vector(gaps_um: List[float], phases: np.ndarray) -> np.ndarray:
    """Compute I(x) by averaging interference patterns over gaps (vectorized for 1.7x speedup)."""
    assert len(gaps_um) == phases.shape[0]
    lambda_nm = 633.0
    k = 2 * np.pi / (lambda_nm * 1e-9)
    theta = _THETA
    amp_per_slit = 0.5

    # Vectorized: gaps (N,1) * theta (1,M) -> phi (N,M)
    gaps = np.asarray(gaps_um) * 1e-6  # Convert to meters
    phi = k * gaps[:, np.newaxis] * np.sin(theta[np.newaxis, :]) + phases[:, np.newaxis]
    field1 = amp_per_slit
    field2 = amp_per_slit * np.exp(1j * phi)
    layer_intensities = np.abs(field1 + field2) ** 2  # Shape: (N, M)

    return np.mean(layer_intensities, axis=0)


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


# =============================================================================
# Degraded Observation Models
# =============================================================================

@dataclass
class ObservationModel:
    """Model for degraded observations."""
    name: str
    noise_std: float = 0.0        # Additive Gaussian noise on energy
    quantize_levels: int = 0      # 0 = continuous, >0 = discrete levels
    stale_period: int = 1         # Evaluate every N steps, hold value between
    
    def observe(self, true_energy: float, rng: np.random.Generator, step: int, 
                last_obs: float) -> float:
        """Return observed energy under this degradation model."""
        # Sparse feedback: use stale value between evaluations
        if self.stale_period > 1 and step % self.stale_period != 0:
            return last_obs
        
        obs = true_energy
        
        # Add noise
        if self.noise_std > 0:
            obs += rng.normal(0, self.noise_std)
            obs = max(0, obs)  # Energy is non-negative
        
        # Quantize
        if self.quantize_levels > 0:
            # Quantize to [0, 1] range with N levels
            obs = np.clip(obs, 0, 1)
            obs = np.round(obs * self.quantize_levels) / self.quantize_levels
        
        return obs


def make_observation_models() -> List[ObservationModel]:
    """Create a range of degradation conditions."""
    return [
        # Clean (baseline)
        ObservationModel(name="Clean", noise_std=0.0, quantize_levels=0, stale_period=1),
        
        # Noisy observations
        ObservationModel(name="Noise-Low", noise_std=0.01, quantize_levels=0, stale_period=1),
        ObservationModel(name="Noise-Med", noise_std=0.05, quantize_levels=0, stale_period=1),
        ObservationModel(name="Noise-High", noise_std=0.1, quantize_levels=0, stale_period=1),
        
        # Quantized feedback
        ObservationModel(name="Quant-10", noise_std=0.0, quantize_levels=10, stale_period=1),
        ObservationModel(name="Quant-5", noise_std=0.0, quantize_levels=5, stale_period=1),
        ObservationModel(name="Quant-3", noise_std=0.0, quantize_levels=3, stale_period=1),
        
        # Sparse/stale feedback
        ObservationModel(name="Stale-2", noise_std=0.0, quantize_levels=0, stale_period=2),
        ObservationModel(name="Stale-5", noise_std=0.0, quantize_levels=0, stale_period=5),
        ObservationModel(name="Stale-10", noise_std=0.0, quantize_levels=0, stale_period=10),
        
        # Combined degradation (realistic worst-case)
        ObservationModel(name="Combined-Mild", noise_std=0.02, quantize_levels=10, stale_period=2),
        ObservationModel(name="Combined-Severe", noise_std=0.05, quantize_levels=5, stale_period=5),
    ]


# =============================================================================
# Optimizers with Degraded Observations
# =============================================================================

def true_objective(gaps_um: List[float], phases: np.ndarray) -> float:
    """True (clean) objective function."""
    I = simulate_interference_vector(gaps_um, phases)
    V = calculate_visibility(I)
    return energy_from_visibility(V)


def run_pson_degraded(
    gaps_um: List[float],
    steps: int,
    w: float,
    lr: float,
    noise_scale: float,
    seed: int,
    obs_model: ObservationModel,
) -> Tuple[float, float, int]:
    """PSON with degraded observations. Returns (final_true_V, final_obs_V, func_evals)."""
    rng = np.random.default_rng(seed)
    d = len(gaps_um)
    phases = np.zeros(d, dtype=float)
    precision, weights = compute_precision_and_weights(gaps_um)
    
    func_evals = 0
    last_obs = 1.0  # Start with worst-case observation
    
    for step in range(steps):
        # Get TRUE energy (for internal physics), then OBSERVED energy (for decision)
        true_E = true_objective(gaps_um, phases)
        func_evals += 1
        obs_E = obs_model.observe(true_E, rng, step, last_obs)
        last_obs = obs_E
        
        # Use OBSERVED energy for gradient computation (partial observability!)
        grad = -w * obs_E * weights
        proposal = phases - lr * grad
        
        # PSON exploration
        delta_perp = project_noise_metric_orthogonal(grad=grad, precision=precision, rng=rng)
        noise = (delta_perp / (np.sqrt(precision) + 1e-12)) * noise_scale
        candidate = proposal + noise
        
        # Evaluate candidate with degraded observation
        true_E_cand = true_objective(gaps_um, candidate)
        func_evals += 1
        obs_E_cand = obs_model.observe(true_E_cand, rng, step, last_obs)
        
        # Accept based on OBSERVED energy (realistic partial observability)
        if obs_E_cand <= obs_E:
            phases = candidate
            last_obs = obs_E_cand
        else:
            # Try deterministic fallback
            true_E_det = true_objective(gaps_um, proposal)
            func_evals += 1
            obs_E_det = obs_model.observe(true_E_det, rng, step, last_obs)
            if obs_E_det <= obs_E:
                phases = proposal
                last_obs = obs_E_det
    
    # Final TRUE performance (not observed)
    final_I = simulate_interference_vector(gaps_um, phases)
    final_V = calculate_visibility(final_I)
    
    return final_V, 1.0 - np.sqrt(last_obs), func_evals


def run_cma_degraded(
    gaps_um: List[float],
    budget: int,
    seed: int,
    obs_model: ObservationModel,
) -> Tuple[float, float, int]:
    """CMA-ES with degraded observations."""
    if not HAS_CMA:
        return float('nan'), float('nan'), 0
    
    d = len(gaps_um)
    rng = np.random.default_rng(seed + 1000)  # Different seed for observation noise
    
    func_evals = [0]
    step_counter = [0]
    last_obs = [1.0]
    
    def degraded_objective(x):
        phases = np.array(x)
        true_E = true_objective(gaps_um, phases)
        func_evals[0] += 1
        obs_E = obs_model.observe(true_E, rng, step_counter[0], last_obs[0])
        last_obs[0] = obs_E
        step_counter[0] += 1
        return obs_E  # CMA-ES optimizes the OBSERVED (degraded) objective
    
    x0 = [0.0] * d
    sigma0 = 1.0
    
    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'seed': seed,
        'maxfevals': budget,
        'verbose': -9,
    })
    
    best_true_V = 0.0
    best_phases = np.zeros(d)
    
    while not es.stop() and func_evals[0] < budget:
        solutions = es.ask()
        fitnesses = [degraded_objective(x) for x in solutions]
        es.tell(solutions, fitnesses)
        
        # Track best TRUE performance
        for x in solutions:
            phases = np.array(x)
            I = simulate_interference_vector(gaps_um, phases)
            V = calculate_visibility(I)
            if V > best_true_V:
                best_true_V = V
                best_phases = phases.copy()
    
    return best_true_V, 1.0 - np.sqrt(last_obs[0]), func_evals[0]


# =============================================================================
# Main Experiment
# =============================================================================

def run_degradation_experiment(
    steps: int,
    w: float,
    lr: float,
    noise: float,
    seeds: List[int],
) -> Tuple[List[Dict], Dict]:
    """Run PSON vs CMA-ES across all degradation conditions."""
    gaps = build_gaps_primes()
    budget = steps * 3
    obs_models = make_observation_models()
    
    rows: List[Dict] = []
    summary: Dict[str, Dict] = {}
    
    for obs_model in obs_models:
        pson_vis: List[float] = []
        cma_vis: List[float] = []
        
        for seed in seeds:
            # PSON
            pson_V, _, pson_evals = run_pson_degraded(
                gaps, steps, w, lr, noise, seed, obs_model
            )
            pson_vis.append(pson_V)
            
            # CMA-ES
            if HAS_CMA:
                cma_V, _, cma_evals = run_cma_degraded(
                    gaps, budget, seed, obs_model
                )
                cma_vis.append(cma_V)
            
            rows.append({
                "condition": obs_model.name,
                "seed": seed,
                "pson_visibility": pson_V,
                "cma_visibility": cma_V if HAS_CMA else float('nan'),
                "pson_advantage": pson_V - cma_V if HAS_CMA else float('nan'),
            })
        
        # Summary for this condition
        summary[obs_model.name] = {
            "pson_mean": float(np.mean(pson_vis)),
            "pson_std": float(np.std(pson_vis)),
            "cma_mean": float(np.mean(cma_vis)) if cma_vis else float('nan'),
            "cma_std": float(np.std(cma_vis)) if cma_vis else float('nan'),
            "pson_advantage": float(np.mean(pson_vis) - np.mean(cma_vis)) if cma_vis else float('nan'),
            "pson_win_rate": float(np.mean([1 if p > c else 0 for p, c in zip(pson_vis, cma_vis)])) if cma_vis else float('nan'),
            "noise_std": obs_model.noise_std,
            "quantize_levels": obs_model.quantize_levels,
            "stale_period": obs_model.stale_period,
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


def plot_degradation(summary: Dict, out_path: str) -> None:
    """Plot PSON vs CMA-ES performance across degradation conditions."""
    conditions = list(summary.keys())
    pson_means = [summary[c]["pson_mean"] for c in conditions]
    pson_stds = [summary[c]["pson_std"] for c in conditions]
    cma_means = [summary[c]["cma_mean"] for c in conditions]
    cma_stds = [summary[c]["cma_std"] for c in conditions]
    
    x = np.arange(len(conditions))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top plot: Absolute performance
    ax1.bar(x - width/2, pson_means, width, yerr=pson_stds, capsize=3, 
            label='PSON', color='#2ecc71', edgecolor='black')
    ax1.bar(x + width/2, cma_means, width, yerr=cma_stds, capsize=3,
            label='CMA-ES', color='#3498db', edgecolor='black')
    ax1.set_ylabel('Final Visibility')
    ax1.set_title('PSON vs CMA-ES Under Degraded Observability')
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=pson_means[0], color='#27ae60', linestyle='--', alpha=0.5)
    ax1.axhline(y=cma_means[0], color='#2980b9', linestyle='--', alpha=0.5)
    
    # Bottom plot: PSON advantage (relative performance)
    advantages = [summary[c]["pson_advantage"] for c in conditions]
    colors = ['#2ecc71' if a > 0 else '#e74c3c' for a in advantages]
    ax2.bar(x, advantages, color=colors, edgecolor='black')
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.set_ylabel('PSON Advantage (PSON V - CMA-ES V)')
    ax2.set_xlabel('Degradation Condition')
    ax2.set_title('PSON Advantage: Positive = PSON Wins')
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Partial Observability Test")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--w", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--seeds", type=str, default="42,123,456,789,1000")
    args = parser.parse_args()
    
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    
    print(f"[partial-obs] Testing PSON vs CMA-ES under degraded observability")
    print(f"[partial-obs] Seeds: {seeds}")
    print(f"[partial-obs] CMA-ES available: {HAS_CMA}")
    
    rows, summary = run_degradation_experiment(
        steps=args.steps,
        w=args.w,
        lr=args.lr,
        noise=args.noise,
        seeds=seeds,
    )
    
    # Save artifacts
    save_csv("partial_observability_001_results.csv", rows)
    
    out = {
        "summary": summary,
        "params": {
            "steps": args.steps,
            "w": args.w,
            "lr": args.lr,
            "noise": args.noise,
            "seeds": seeds,
        },
    }
    with open("partial_observability_001_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    
    plot_degradation(summary, "partial_observability_001_degradation.png")
    
    # Print summary
    print("\n=== Partial Observability Test Results ===")
    print(f"\n{'Condition':<20} {'PSON V':>10} {'CMA-ES V':>10} {'Advantage':>10} {'PSON Wins':>10}")
    print("-" * 65)
    for cond, stats in summary.items():
        win_pct = f"{stats['pson_win_rate']*100:.0f}%" if not np.isnan(stats['pson_win_rate']) else "N/A"
        adv = f"{stats['pson_advantage']:+.4f}" if not np.isnan(stats['pson_advantage']) else "N/A"
        print(f"{cond:<20} {stats['pson_mean']:>10.4f} {stats['cma_mean']:>10.4f} {adv:>10} {win_pct:>10}")
    
    # Key findings
    print("\n=== Key Findings ===")
    clean = summary.get("Clean", {})
    severe = summary.get("Combined-Severe", {})
    
    if clean and severe:
        clean_adv = clean.get("pson_advantage", 0)
        severe_adv = severe.get("pson_advantage", 0)
        print(f"Clean conditions: PSON advantage = {clean_adv:+.4f}")
        print(f"Severe degradation: PSON advantage = {severe_adv:+.4f}")
        print(f"Change in advantage: {severe_adv - clean_adv:+.4f}")
        
        if severe_adv > clean_adv:
            print("\n[OK] PSON's relative advantage INCREASES under degraded observability!")
            print(f"     Gap closed by {severe_adv - clean_adv:+.4f} (from {clean_adv:+.4f} to {severe_adv:+.4f})")
        else:
            print("\n[--] PSON's relative advantage decreases under degradation.")
    
    print("\nArtifacts saved:")
    print("  - partial_observability_001_results.csv")
    print("  - partial_observability_001_summary.json")
    print("  - partial_observability_001_degradation.png")


if __name__ == "__main__":
    main()

