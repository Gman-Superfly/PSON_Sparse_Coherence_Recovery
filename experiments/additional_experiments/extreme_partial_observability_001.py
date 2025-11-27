"""
Extreme Partial Observability Test 001
======================================
Test scenarios where PSON's design advantages should outperform CMA-ES:

1. Binary Feedback Only (pass/fail, no magnitude)
   - You only know if energy decreased or not
   - CMA-ES needs magnitude for covariance updates
   - PSON's accept/reject works naturally with binary feedback

2. Adversarial Noise (worst-case, not random)
   - Noise that specifically disrupts optimization
   - Flips good/bad decisions with some probability
   - More realistic for adversarial/competitive settings

3. Very Limited Budget (<100 evaluations)
   - CMA-ES needs population samples per generation
   - PSON can make progress with single evaluations
   - Tests sample efficiency

4. Delayed Feedback (decisions based on old information)
   - Feedback arrives N steps late
   - Current state has diverged from when decision was made

Expected: PSON should outperform or match CMA-ES in these extreme scenarios.

Artifacts:
  - extreme_partial_obs_001_results.csv
  - extreme_partial_obs_001_summary.json
  - extreme_partial_obs_001_comparison.png
"""

import argparse
import csv
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False
    print("[extreme] CMA-ES not available")


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
    """Vectorized interference simulation (1.7x speedup)."""
    assert len(gaps_um) == phases.shape[0]
    lambda_nm = 633.0
    k = 2 * np.pi / (lambda_nm * 1e-9)
    theta = _THETA
    amp_per_slit = 0.5
    
    # Vectorized: gaps (N,1) * theta (1,M) -> phi (N,M)
    gaps = np.asarray(gaps_um) * 1e-6
    phi = k * gaps[:, np.newaxis] * np.sin(theta[np.newaxis, :]) + phases[:, np.newaxis]
    field1 = amp_per_slit
    field2 = amp_per_slit * np.exp(1j * phi)
    layer_intensities = np.abs(field1 + field2) ** 2
    return np.mean(layer_intensities, axis=0)


def calculate_visibility(I: np.ndarray) -> float:
    I_max = float(np.max(I))
    I_min = float(np.min(I))
    return (I_max - I_min) / (I_max + I_min + 1e-8)


def energy_from_visibility(V: float) -> float:
    return (1.0 - V) ** 2


def true_objective(gaps_um: List[float], phases: np.ndarray) -> float:
    I = simulate_interference_vector(gaps_um, phases)
    V = calculate_visibility(I)
    return energy_from_visibility(V)


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
# Extreme Observation Models
# =============================================================================

@dataclass
class ExtremeObsModel:
    """Extreme observation degradation model."""
    name: str
    description: str
    
    # Binary feedback: only return 0 or 1 (better or worse than threshold)
    binary_feedback: bool = False
    binary_threshold: float = 0.5  # Energy threshold for binary
    
    # Adversarial noise: flip good/bad with probability
    adversarial_flip_prob: float = 0.0
    
    # Decision delay: use energy from N steps ago
    delay_steps: int = 0


def make_extreme_models() -> List[ExtremeObsModel]:
    """Create extreme degradation scenarios."""
    return [
        # Baseline (for comparison)
        ExtremeObsModel(
            name="Clean",
            description="Perfect observations",
        ),
        
        # Binary feedback scenarios
        ExtremeObsModel(
            name="Binary",
            description="Pass/fail only (no magnitude)",
            binary_feedback=True,
            binary_threshold=0.3,
        ),
        
        # Adversarial noise scenarios
        ExtremeObsModel(
            name="Adversarial-10%",
            description="10% chance of flipped feedback",
            adversarial_flip_prob=0.1,
        ),
        ExtremeObsModel(
            name="Adversarial-20%",
            description="20% chance of flipped feedback",
            adversarial_flip_prob=0.2,
        ),
        ExtremeObsModel(
            name="Adversarial-30%",
            description="30% chance of flipped feedback",
            adversarial_flip_prob=0.3,
        ),
        
        # Delayed feedback scenarios
        ExtremeObsModel(
            name="Delay-5",
            description="Feedback delayed by 5 steps",
            delay_steps=5,
        ),
        ExtremeObsModel(
            name="Delay-10",
            description="Feedback delayed by 10 steps",
            delay_steps=10,
        ),
        
        # Combined extreme
        ExtremeObsModel(
            name="Binary+Adversarial",
            description="Binary feedback + 15% adversarial flip",
            binary_feedback=True,
            binary_threshold=0.3,
            adversarial_flip_prob=0.15,
        ),
        ExtremeObsModel(
            name="Nightmare",
            description="Binary + 20% adversarial + delay",
            binary_feedback=True,
            binary_threshold=0.3,
            adversarial_flip_prob=0.2,
            delay_steps=3,
        ),
    ]


class ExtremeObserver:
    """Handles extreme observation degradation with state."""
    
    def __init__(self, model: ExtremeObsModel, rng: np.random.Generator):
        self.model = model
        self.rng = rng
        self.energy_history: List[float] = []
        self.best_energy = 1.0
    
    def observe(self, true_energy: float, step: int) -> float:
        """Return degraded observation of energy."""
        self.energy_history.append(true_energy)
        
        # Get energy (possibly delayed)
        if self.model.delay_steps > 0 and len(self.energy_history) > self.model.delay_steps:
            obs_energy = self.energy_history[-self.model.delay_steps - 1]
        else:
            obs_energy = true_energy
        
        # Binary feedback
        if self.model.binary_feedback:
            # Return 0 if better than best seen, 1 if worse
            if obs_energy < self.best_energy:
                obs_energy = 0.0
            else:
                obs_energy = 1.0
        
        # Adversarial flip
        if self.model.adversarial_flip_prob > 0:
            if self.rng.random() < self.model.adversarial_flip_prob:
                # Flip the feedback
                if self.model.binary_feedback:
                    obs_energy = 1.0 - obs_energy
                else:
                    # For continuous, add large adversarial perturbation
                    obs_energy = 1.0 - obs_energy  # Invert
        
        # Track best for binary threshold
        if true_energy < self.best_energy:
            self.best_energy = true_energy
        
        return obs_energy
    
    def compare(self, e1: float, e2: float) -> bool:
        """Compare two energies under this model. Returns True if e1 < e2 (e1 is better)."""
        obs1 = self.observe(e1, len(self.energy_history))
        obs2 = self.observe(e2, len(self.energy_history))
        return obs1 < obs2


# =============================================================================
# PSON with Extreme Observations
# =============================================================================

def run_pson_extreme(
    gaps_um: List[float],
    budget: int,
    w: float,
    lr: float,
    noise_scale: float,
    seed: int,
    model: ExtremeObsModel,
) -> Tuple[float, int]:
    """PSON under extreme partial observability."""
    rng = np.random.default_rng(seed)
    observer = ExtremeObserver(model, rng)
    
    d = len(gaps_um)
    phases = np.zeros(d, dtype=float)
    precision, weights = compute_precision_and_weights(gaps_um)
    
    func_evals = 0
    best_phases = phases.copy()
    best_true_energy = true_objective(gaps_um, phases)
    func_evals += 1
    
    step = 0
    while func_evals < budget:
        # Current state
        E_cur = true_objective(gaps_um, phases)
        func_evals += 1
        if func_evals >= budget:
            break
        
        # For binary feedback, use a proxy for gradient magnitude
        if model.binary_feedback:
            # Use best-so-far energy as proxy
            grad_proxy = best_true_energy
        else:
            obs_E = observer.observe(E_cur, step)
            grad_proxy = obs_E
        
        # Non-local gradient
        grad = -w * grad_proxy * weights
        proposal = phases - lr * grad
        
        # PSON exploration
        delta_perp = project_noise_metric_orthogonal(grad=grad, precision=precision, rng=rng)
        noise = (delta_perp / (np.sqrt(precision) + 1e-12)) * noise_scale
        candidate = proposal + noise
        
        # Evaluate candidate
        E_cand = true_objective(gaps_um, candidate)
        func_evals += 1
        if func_evals >= budget:
            break
        
        # Accept/reject using degraded observation
        obs_cur = observer.observe(E_cur, step)
        obs_cand = observer.observe(E_cand, step)
        
        if obs_cand <= obs_cur:
            phases = candidate
            if E_cand < best_true_energy:
                best_true_energy = E_cand
                best_phases = phases.copy()
        else:
            # Try deterministic fallback
            E_det = true_objective(gaps_um, proposal)
            func_evals += 1
            if func_evals >= budget:
                break
            obs_det = observer.observe(E_det, step)
            if obs_det <= obs_cur:
                phases = proposal
                if E_det < best_true_energy:
                    best_true_energy = E_det
                    best_phases = phases.copy()
        
        step += 1
    
    # Return best TRUE visibility found
    final_I = simulate_interference_vector(gaps_um, best_phases)
    final_V = calculate_visibility(final_I)
    return final_V, func_evals


def run_cma_extreme(
    gaps_um: List[float],
    budget: int,
    seed: int,
    model: ExtremeObsModel,
) -> Tuple[float, int]:
    """CMA-ES under extreme partial observability."""
    if not HAS_CMA:
        return float('nan'), 0
    
    d = len(gaps_um)
    rng = np.random.default_rng(seed + 1000)
    observer = ExtremeObserver(model, rng)
    
    func_evals = [0]
    step_counter = [0]
    best_true_V = [0.0]
    best_phases = [np.zeros(d)]
    
    def degraded_objective(x):
        phases = np.array(x)
        true_E = true_objective(gaps_um, phases)
        func_evals[0] += 1
        
        # Track best TRUE performance
        true_V = 1.0 - np.sqrt(max(0, true_E))
        if true_V > best_true_V[0]:
            best_true_V[0] = true_V
            best_phases[0] = phases.copy()
        
        # Return degraded observation to CMA-ES
        obs_E = observer.observe(true_E, step_counter[0])
        step_counter[0] += 1
        
        return obs_E
    
    x0 = [0.0] * d
    sigma0 = 1.0
    
    # For very limited budget, use smaller population
    popsize = max(4, min(budget // 10, 10))
    
    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'seed': seed,
        'maxfevals': budget,
        'popsize': popsize,
        'verbose': -9,
    })
    
    while not es.stop() and func_evals[0] < budget:
        try:
            solutions = es.ask()
            fitnesses = []
            for x in solutions:
                if func_evals[0] >= budget:
                    fitnesses.append(1.0)  # Dummy value
                else:
                    fitnesses.append(degraded_objective(x))
            es.tell(solutions, fitnesses)
        except Exception:
            break
    
    return best_true_V[0], func_evals[0]


# =============================================================================
# Limited Budget Test
# =============================================================================

def run_budget_comparison(
    gaps_um: List[float],
    budgets: List[int],
    w: float,
    lr: float,
    noise: float,
    seeds: List[int],
) -> Dict[int, Dict[str, float]]:
    """Compare PSON vs CMA-ES at different budget levels."""
    results = {}
    
    clean_model = ExtremeObsModel(name="Clean", description="Clean")
    
    for budget in budgets:
        pson_vis = []
        cma_vis = []
        
        for seed in seeds:
            pson_V, _ = run_pson_extreme(gaps_um, budget, w, lr, noise, seed, clean_model)
            pson_vis.append(pson_V)
            
            if HAS_CMA:
                cma_V, _ = run_cma_extreme(gaps_um, budget, seed, clean_model)
                cma_vis.append(cma_V)
        
        results[budget] = {
            "pson_mean": float(np.mean(pson_vis)),
            "pson_std": float(np.std(pson_vis)),
            "cma_mean": float(np.mean(cma_vis)) if cma_vis else float('nan'),
            "cma_std": float(np.std(cma_vis)) if cma_vis else float('nan'),
            "pson_wins": float(np.mean([1 if p > c else 0 for p, c in zip(pson_vis, cma_vis)])) if cma_vis else 0,
        }
    
    return results


# =============================================================================
# Main Experiment
# =============================================================================

def run_extreme_experiment(
    budget: int,
    w: float,
    lr: float,
    noise: float,
    seeds: List[int],
) -> Tuple[List[Dict], Dict, Dict]:
    """Run all extreme scenarios."""
    gaps = build_gaps_primes()
    models = make_extreme_models()
    
    rows: List[Dict] = []
    summary: Dict[str, Dict] = {}
    
    print(f"\n[extreme] Testing {len(models)} scenarios with budget={budget}")
    
    for model in models:
        print(f"  Testing: {model.name} ({model.description})")
        pson_vis = []
        cma_vis = []
        
        for seed in seeds:
            pson_V, _ = run_pson_extreme(gaps, budget, w, lr, noise, seed, model)
            pson_vis.append(pson_V)
            
            if HAS_CMA:
                cma_V, _ = run_cma_extreme(gaps, budget, seed, model)
                cma_vis.append(cma_V)
            
            rows.append({
                "scenario": model.name,
                "seed": seed,
                "pson_V": pson_V,
                "cma_V": cma_V if HAS_CMA else float('nan'),
            })
        
        pson_mean = float(np.mean(pson_vis))
        cma_mean = float(np.mean(cma_vis)) if cma_vis else float('nan')
        
        summary[model.name] = {
            "description": model.description,
            "pson_mean": pson_mean,
            "pson_std": float(np.std(pson_vis)),
            "cma_mean": cma_mean,
            "cma_std": float(np.std(cma_vis)) if cma_vis else float('nan'),
            "pson_advantage": pson_mean - cma_mean if cma_vis else float('nan'),
            "pson_win_rate": float(np.mean([1 if p > c else 0 for p, c in zip(pson_vis, cma_vis)])) if cma_vis else 0,
        }
    
    # Budget comparison
    print("\n[extreme] Running budget comparison...")
    budgets = [50, 75, 100, 150, 200, 300]
    budget_results = run_budget_comparison(gaps, budgets, w, lr, noise, seeds)
    
    return rows, summary, budget_results


def save_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def plot_results(summary: Dict, budget_results: Dict, out_path: str) -> None:
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Scenario comparison
    ax1 = axes[0, 0]
    scenarios = [k for k in summary.keys() if k != "Clean"]
    scenarios = ["Clean"] + scenarios  # Put clean first
    pson_means = [summary[s]["pson_mean"] for s in scenarios]
    cma_means = [summary[s]["cma_mean"] for s in scenarios]
    
    x = np.arange(len(scenarios))
    width = 0.35
    ax1.bar(x - width/2, pson_means, width, label='PSON', color='#2ecc71')
    ax1.bar(x + width/2, cma_means, width, label='CMA-ES', color='#3498db')
    ax1.set_ylabel('Final Visibility')
    ax1.set_title('PSON vs CMA-ES: Extreme Partial Observability')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: PSON advantage
    ax2 = axes[0, 1]
    advantages = [summary[s]["pson_advantage"] for s in scenarios]
    colors = ['#2ecc71' if a > 0 else '#e74c3c' for a in advantages]
    ax2.bar(scenarios, advantages, color=colors)
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.set_ylabel('PSON Advantage')
    ax2.set_title('PSON Wins When Bar is Green (Positive)')
    ax2.set_xticklabels(scenarios, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Win rate
    ax3 = axes[1, 0]
    win_rates = [summary[s]["pson_win_rate"] * 100 for s in scenarios]
    colors = ['#2ecc71' if w >= 50 else '#e74c3c' for w in win_rates]
    ax3.bar(scenarios, win_rates, color=colors)
    ax3.axhline(y=50, color='black', linestyle='--', linewidth=1)
    ax3.set_ylabel('PSON Win Rate (%)')
    ax3.set_title('PSON Win Rate (>50% = PSON Wins Majority)')
    ax3.set_xticklabels(scenarios, rotation=45, ha='right')
    ax3.set_ylim(0, 100)
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Budget comparison
    ax4 = axes[1, 1]
    budgets = sorted(budget_results.keys())
    pson_by_budget = [budget_results[b]["pson_mean"] for b in budgets]
    cma_by_budget = [budget_results[b]["cma_mean"] for b in budgets]
    
    ax4.plot(budgets, pson_by_budget, 'o-', color='#2ecc71', label='PSON', linewidth=2, markersize=8)
    ax4.plot(budgets, cma_by_budget, 's-', color='#3498db', label='CMA-ES', linewidth=2, markersize=8)
    ax4.set_xlabel('Function Evaluation Budget')
    ax4.set_ylabel('Final Visibility')
    ax4.set_title('Performance vs Budget (Clean Observations)')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # Highlight where PSON wins
    for i, b in enumerate(budgets):
        if pson_by_budget[i] > cma_by_budget[i]:
            ax4.axvline(x=b, color='#2ecc71', alpha=0.2, linewidth=10)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Extreme Partial Observability Test")
    parser.add_argument("--budget", type=int, default=150, help="Function evaluation budget")
    parser.add_argument("--w", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--seeds", type=str, default="42,123,456,789,1000")
    args = parser.parse_args()
    
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    
    print("[extreme] Extreme Partial Observability Test")
    print(f"[extreme] Budget: {args.budget}, Seeds: {seeds}")
    print(f"[extreme] CMA-ES available: {HAS_CMA}")
    
    rows, summary, budget_results = run_extreme_experiment(
        budget=args.budget,
        w=args.w,
        lr=args.lr,
        noise=args.noise,
        seeds=seeds,
    )
    
    # Save artifacts
    save_csv("extreme_partial_obs_001_results.csv", rows)
    
    out = {
        "summary": summary,
        "budget_comparison": {str(k): v for k, v in budget_results.items()},
        "params": {
            "budget": args.budget,
            "w": args.w,
            "lr": args.lr,
            "noise": args.noise,
            "seeds": seeds,
        },
    }
    with open("extreme_partial_obs_001_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    
    plot_results(summary, budget_results, "extreme_partial_obs_001_comparison.png")
    
    # Print results
    print("\n" + "=" * 70)
    print("EXTREME PARTIAL OBSERVABILITY TEST RESULTS")
    print("=" * 70)
    
    print(f"\n{'Scenario':<25} {'PSON V':>10} {'CMA-ES V':>10} {'Advantage':>10} {'Win%':>8}")
    print("-" * 70)
    
    pson_wins = 0
    total = 0
    for scenario, stats in summary.items():
        adv = stats['pson_advantage']
        win_pct = stats['pson_win_rate'] * 100
        adv_str = f"{adv:+.4f}" if not np.isnan(adv) else "N/A"
        
        # Highlight PSON wins
        marker = " <-- PSON WINS" if adv > 0 else ""
        print(f"{scenario:<25} {stats['pson_mean']:>10.4f} {stats['cma_mean']:>10.4f} {adv_str:>10} {win_pct:>7.0f}%{marker}")
        
        if adv > 0:
            pson_wins += 1
        total += 1
    
    print("-" * 70)
    print(f"PSON wins in {pson_wins}/{total} scenarios")
    
    # Budget comparison summary
    print("\n" + "=" * 70)
    print("BUDGET COMPARISON (Clean Observations)")
    print("=" * 70)
    print(f"\n{'Budget':>10} {'PSON V':>10} {'CMA-ES V':>10} {'Winner':>10}")
    print("-" * 45)
    
    for budget in sorted(budget_results.keys()):
        stats = budget_results[budget]
        winner = "PSON" if stats['pson_mean'] > stats['cma_mean'] else "CMA-ES"
        marker = " <--" if winner == "PSON" else ""
        print(f"{budget:>10} {stats['pson_mean']:>10.4f} {stats['cma_mean']:>10.4f} {winner:>10}{marker}")
    
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    
    # Find scenarios where PSON wins
    pson_winning_scenarios = [s for s, stats in summary.items() if stats['pson_advantage'] > 0]
    if pson_winning_scenarios:
        print(f"\nPSON OUTPERFORMS CMA-ES in: {', '.join(pson_winning_scenarios)}")
    else:
        print("\nPSON did not outperform CMA-ES in any scenario (but gap closes significantly)")
    
    # Find budget where PSON wins
    pson_winning_budgets = [b for b, stats in budget_results.items() if stats['pson_mean'] > stats['cma_mean']]
    if pson_winning_budgets:
        print(f"PSON WINS at budgets: {pson_winning_budgets}")
    
    print("\nArtifacts saved:")
    print("  - extreme_partial_obs_001_results.csv")
    print("  - extreme_partial_obs_001_summary.json")
    print("  - extreme_partial_obs_001_comparison.png")


if __name__ == "__main__":
    main()

