"""
Baseline Comparison Experiment 001
==================================
Compare PSON against standard optimization baselines:
1. Random Search
2. Simulated Annealing
3. CMA-ES (if cma package available)
4. Vanilla Gradient Descent (finite differences)

This experiment validates that PSON's gains are not trivially achievable
by standard black-box optimization methods.

Artifacts:
  - baseline_comparison_001_results.csv
  - baseline_comparison_001_summary.json
  - baseline_comparison_001_bar.png
"""

import argparse
import csv
import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Check for optional CMA-ES
try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False
    print("[baseline] CMA-ES not available (pip install cma to enable)")


# =============================================================================
# Shared Optics Primitives (from airtight_experiments_001.py)
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
    assert len(gaps_um) == phases.shape[0], "phases must align with gaps"
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
    """Fringe visibility (contrast)."""
    I_max = float(np.max(I))
    I_min = float(np.min(I))
    return (I_max - I_min) / (I_max + I_min + 1e-8)


def energy_from_visibility(V: float) -> float:
    """Energy to minimize: lower when visibility is higher."""
    return (1.0 - V) ** 2


def compute_precision_and_weights(gaps_um: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Derive per-parameter precision and weights from gap irregularity."""
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
    """Metric-orthogonal projection of noise relative to grad."""
    z = rng.normal(0.0, 1.0, size=grad.shape[0])
    Mz = precision * z
    Mg = precision * grad
    denom = float(np.dot(grad, Mg)) + 1e-12
    if abs(denom) < 1e-18:
        return z
    alpha = float(np.dot(grad, Mz)) / denom
    return z - alpha * grad


# =============================================================================
# Objective Function
# =============================================================================

def make_objective(gaps_um: List[float]) -> Callable[[np.ndarray], float]:
    """Create objective function: phases -> energy (to minimize)."""
    def objective(phases: np.ndarray) -> float:
        I = simulate_interference_vector(gaps_um, phases)
        V = calculate_visibility(I)
        return energy_from_visibility(V)
    return objective


# =============================================================================
# Optimizers
# =============================================================================

@dataclass
class OptResult:
    """Result from an optimizer run."""
    name: str
    final_energy: float
    final_visibility: float
    final_phases: np.ndarray
    energies: List[float]
    func_evals: int


def run_pson(
    gaps_um: List[float],
    steps: int,
    w: float,
    lr: float,
    noise_scale: float,
    seed: int,
) -> OptResult:
    """PSON optimizer (our method)."""
    rng = np.random.default_rng(seed)
    d = len(gaps_um)
    phases = np.zeros(d, dtype=float)
    precision, weights = compute_precision_and_weights(gaps_um)
    
    energies: List[float] = []
    func_evals = 0
    
    # Initial
    I0 = simulate_interference_vector(gaps_um, phases)
    V0 = calculate_visibility(I0)
    E0 = energy_from_visibility(V0)
    energies.append(E0)
    func_evals += 1
    
    for _ in range(steps):
        I_cur = simulate_interference_vector(gaps_um, phases)
        V_cur = calculate_visibility(I_cur)
        E_cur = energy_from_visibility(V_cur)
        func_evals += 1
        
        # Non-local gradient
        grad = -w * E_cur * weights
        proposal = phases - lr * grad
        
        # PSON
        delta_perp = project_noise_metric_orthogonal(grad=grad, precision=precision, rng=rng)
        noise = (delta_perp / (np.sqrt(precision) + 1e-12)) * noise_scale
        candidate = proposal + noise
        
        # Accept/reject
        I_new = simulate_interference_vector(gaps_um, candidate)
        V_new = calculate_visibility(I_new)
        E_new = energy_from_visibility(V_new)
        func_evals += 1
        
        if E_new <= E_cur:
            phases = candidate
            energies.append(E_new)
        else:
            # Try deterministic fallback
            I_det = simulate_interference_vector(gaps_um, proposal)
            V_det = calculate_visibility(I_det)
            E_det = energy_from_visibility(V_det)
            func_evals += 1
            if E_det <= E_cur:
                phases = proposal
                energies.append(E_det)
            else:
                energies.append(E_cur)
    
    final_I = simulate_interference_vector(gaps_um, phases)
    final_V = calculate_visibility(final_I)
    final_E = energy_from_visibility(final_V)
    
    return OptResult(
        name="PSON",
        final_energy=final_E,
        final_visibility=final_V,
        final_phases=phases.copy(),
        energies=energies,
        func_evals=func_evals,
    )


def run_random_search(
    gaps_um: List[float],
    budget: int,
    seed: int,
) -> OptResult:
    """Random search baseline."""
    rng = np.random.default_rng(seed)
    d = len(gaps_um)
    objective = make_objective(gaps_um)
    
    best_phases = np.zeros(d)
    best_energy = objective(best_phases)
    energies = [best_energy]
    func_evals = 1
    
    for _ in range(budget - 1):
        # Random phases in [-π, π]
        candidate = rng.uniform(-np.pi, np.pi, size=d)
        E = objective(candidate)
        func_evals += 1
        if E < best_energy:
            best_energy = E
            best_phases = candidate.copy()
        energies.append(best_energy)
    
    final_V = 1.0 - np.sqrt(best_energy)
    
    return OptResult(
        name="Random Search",
        final_energy=best_energy,
        final_visibility=final_V,
        final_phases=best_phases,
        energies=energies,
        func_evals=func_evals,
    )


def run_simulated_annealing(
    gaps_um: List[float],
    budget: int,
    seed: int,
    T_init: float = 0.1,
    T_min: float = 0.001,
) -> OptResult:
    """Simulated annealing baseline."""
    rng = np.random.default_rng(seed)
    d = len(gaps_um)
    objective = make_objective(gaps_um)
    
    phases = np.zeros(d)
    E_cur = objective(phases)
    best_phases = phases.copy()
    best_energy = E_cur
    energies = [E_cur]
    func_evals = 1
    
    # Cooling schedule
    alpha = (T_min / T_init) ** (1.0 / max(1, budget - 1))
    T = T_init
    
    for _ in range(budget - 1):
        # Propose neighbor
        step_size = 0.5 * T / T_init  # Shrink steps as T decreases
        candidate = phases + rng.normal(0, step_size, size=d)
        E_new = objective(candidate)
        func_evals += 1
        
        # Accept?
        delta = E_new - E_cur
        if delta < 0 or rng.random() < np.exp(-delta / T):
            phases = candidate
            E_cur = E_new
            if E_cur < best_energy:
                best_energy = E_cur
                best_phases = phases.copy()
        
        energies.append(best_energy)
        T *= alpha
    
    final_V = 1.0 - np.sqrt(best_energy)
    
    return OptResult(
        name="Simulated Annealing",
        final_energy=best_energy,
        final_visibility=final_V,
        final_phases=best_phases,
        energies=energies,
        func_evals=func_evals,
    )


def run_finite_diff_gd(
    gaps_um: List[float],
    steps: int,
    lr: float,
    seed: int,
    eps: float = 1e-4,
) -> OptResult:
    """Vanilla gradient descent with finite difference gradients."""
    d = len(gaps_um)
    objective = make_objective(gaps_um)
    
    phases = np.zeros(d)
    E_cur = objective(phases)
    energies = [E_cur]
    func_evals = 1
    
    for _ in range(steps):
        # Compute gradient via finite differences
        grad = np.zeros(d)
        for i in range(d):
            phases_plus = phases.copy()
            phases_plus[i] += eps
            phases_minus = phases.copy()
            phases_minus[i] -= eps
            grad[i] = (objective(phases_plus) - objective(phases_minus)) / (2 * eps)
            func_evals += 2
        
        # Gradient step
        phases = phases - lr * grad
        E_cur = objective(phases)
        func_evals += 1
        energies.append(E_cur)
    
    final_V = 1.0 - np.sqrt(E_cur)
    
    return OptResult(
        name="Finite-Diff GD",
        final_energy=E_cur,
        final_visibility=final_V,
        final_phases=phases.copy(),
        energies=energies,
        func_evals=func_evals,
    )


def run_cma_es(
    gaps_um: List[float],
    budget: int,
    seed: int,
) -> OptResult:
    """CMA-ES baseline (requires cma package)."""
    if not HAS_CMA:
        return OptResult(
            name="CMA-ES",
            final_energy=float('nan'),
            final_visibility=float('nan'),
            final_phases=np.zeros(len(gaps_um)),
            energies=[],
            func_evals=0,
        )
    
    d = len(gaps_um)
    objective = make_objective(gaps_um)
    
    energies: List[float] = []
    func_evals = [0]  # Use list to allow mutation in callback
    
    def tracked_objective(x):
        E = objective(np.array(x))
        func_evals[0] += 1
        return E
    
    # CMA-ES
    x0 = [0.0] * d
    sigma0 = 1.0
    
    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'seed': seed,
        'maxfevals': budget,
        'verbose': -9,  # Suppress output
    })
    
    best_energy = float('inf')
    best_phases = np.zeros(d)
    
    while not es.stop() and func_evals[0] < budget:
        solutions = es.ask()
        fitnesses = [tracked_objective(x) for x in solutions]
        es.tell(solutions, fitnesses)
        
        min_idx = np.argmin(fitnesses)
        if fitnesses[min_idx] < best_energy:
            best_energy = fitnesses[min_idx]
            best_phases = np.array(solutions[min_idx])
        energies.append(best_energy)
    
    final_V = 1.0 - np.sqrt(best_energy)
    
    return OptResult(
        name="CMA-ES",
        final_energy=best_energy,
        final_visibility=final_V,
        final_phases=best_phases,
        energies=energies,
        func_evals=func_evals[0],
    )


# =============================================================================
# Main Experiment
# =============================================================================

def run_comparison(
    steps: int,
    w: float,
    lr: float,
    noise: float,
    seeds: List[int],
    include_cma: bool,
) -> Tuple[List[Dict], Dict]:
    """Run all optimizers and compare."""
    gaps = build_gaps_primes()
    budget = steps * 3  # Approximate func evals for PSON
    
    rows: List[Dict] = []
    summary: Dict[str, Dict[str, float]] = {}
    
    for method in ["PSON", "Random Search", "Simulated Annealing", "Finite-Diff GD", "CMA-ES"]:
        if method == "CMA-ES" and (not include_cma or not HAS_CMA):
            continue
        
        vis_list: List[float] = []
        energy_list: List[float] = []
        evals_list: List[int] = []
        
        for seed in seeds:
            if method == "PSON":
                result = run_pson(gaps, steps, w, lr, noise, seed)
            elif method == "Random Search":
                result = run_random_search(gaps, budget, seed)
            elif method == "Simulated Annealing":
                result = run_simulated_annealing(gaps, budget, seed)
            elif method == "Finite-Diff GD":
                result = run_finite_diff_gd(gaps, steps, lr * 0.1, seed)  # Smaller lr for FD
            elif method == "CMA-ES":
                result = run_cma_es(gaps, budget, seed)
            else:
                continue
            
            rows.append({
                "method": method,
                "seed": seed,
                "final_visibility": result.final_visibility,
                "final_energy": result.final_energy,
                "func_evals": result.func_evals,
            })
            
            vis_list.append(result.final_visibility)
            energy_list.append(result.final_energy)
            evals_list.append(result.func_evals)
        
        if vis_list:
            summary[method] = {
                "mean_visibility": float(np.mean(vis_list)),
                "std_visibility": float(np.std(vis_list)),
                "mean_energy": float(np.mean(energy_list)),
                "mean_func_evals": float(np.mean(evals_list)),
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


def plot_comparison(summary: Dict[str, Dict[str, float]], out_path: str) -> None:
    """Bar chart comparing final visibility across methods."""
    methods = list(summary.keys())
    means = [summary[m]["mean_visibility"] for m in methods]
    stds = [summary[m].get("std_visibility", 0) for m in methods]
    
    colors = ["#2ecc71" if m == "PSON" else "#3498db" for m in methods]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(methods))
    bars = plt.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black")
    
    plt.axhline(y=means[methods.index("PSON")] if "PSON" in methods else 0, 
                color="#27ae60", linestyle="--", linewidth=2, alpha=0.7, label="PSON")
    
    plt.xticks(x, methods, rotation=15, ha="right")
    plt.ylabel("Final Visibility")
    plt.title("Baseline Comparison: PSON vs Standard Optimizers")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Baseline Comparison Experiment")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--w", type=float, default=0.2, help="Non-local gain")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--noise", type=float, default=0.02, help="PSON noise scale")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,1000", help="Comma-separated seeds")
    parser.add_argument("--no_cma", action="store_true", help="Skip CMA-ES even if available")
    args = parser.parse_args()
    
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    include_cma = not args.no_cma
    
    print(f"[baseline] Running comparison with {len(seeds)} seeds...")
    print(f"[baseline] Methods: PSON, Random Search, Simulated Annealing, Finite-Diff GD" + 
          (", CMA-ES" if include_cma and HAS_CMA else ""))
    
    rows, summary = run_comparison(
        steps=args.steps,
        w=args.w,
        lr=args.lr,
        noise=args.noise,
        seeds=seeds,
        include_cma=include_cma,
    )
    
    # Save artifacts
    save_csv("baseline_comparison_001_results.csv", rows)
    
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
    with open("baseline_comparison_001_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    
    plot_comparison(summary, "baseline_comparison_001_bar.png")
    
    # Print summary
    print("\n=== Baseline Comparison Summary ===")
    print(f"{'Method':<20} {'Mean V':>10} {'Std V':>10} {'Mean Evals':>12}")
    print("-" * 54)
    for method, stats in summary.items():
        print(f"{method:<20} {stats['mean_visibility']:>10.4f} {stats.get('std_visibility', 0):>10.4f} {stats['mean_func_evals']:>12.0f}")
    
    print("\nArtifacts saved:")
    print("  - baseline_comparison_001_results.csv")
    print("  - baseline_comparison_001_summary.json")
    print("  - baseline_comparison_001_bar.png")


if __name__ == "__main__":
    main()

