import argparse
import json
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Sparse Path Integral Approximation Test
# Goal:
# - Ground truth "full" reference: dense uniform gaps over [min(sparse), max(sparse)]
# - Sparse approximation: prime-gap subset with phases optimized via Wormhole + PSON
# - Objective: minimize MSE between I_sparse(x) and I_full(x)
# - Artifacts: overlay plot and JSON summary
# =============================================================================


def generate_first_n_primes(n: int) -> List[int]:
    """Simple incremental prime generator sufficient for n up to a few hundreds."""
    assert n >= 1, "n must be >= 1"
    primes: List[int] = []
    candidate = 2
    while len(primes) < n:
        isprime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                isprime = False
                break
        if isprime:
            primes.append(candidate)
        candidate += 1 if candidate == 2 else 2  # skip even numbers after 2
    return primes


def build_sparse_gaps_primes(count: int) -> List[float]:
    return [float(p * 10) for p in generate_first_n_primes(count)]


def build_dense_gaps_uniform(sparse_gaps_um: List[float], num_dense: int) -> List[float]:
    gmin = float(np.min(sparse_gaps_um))
    gmax = float(np.max(sparse_gaps_um))
    dense = np.linspace(gmin, gmax, num_dense)
    return [float(x) for x in dense]


def screen_and_theta() -> Tuple[np.ndarray, np.ndarray]:
    L = 1.0
    x_screen = np.linspace(-0.005, 0.005, 500)
    theta = x_screen / L
    return x_screen, theta


_X_SCREEN, _THETA = screen_and_theta()


def simulate_intensity_for_gaps(
    gaps_um: List[float],
    phases: np.ndarray,
    amps: np.ndarray = None,
    lambda_nm: float = 633.0,
) -> np.ndarray:
    """
    Two-slit per gap; average intensity across gaps.
    """
    assert len(gaps_um) == phases.shape[0], "phases must match gaps"
    k = 2 * np.pi / (lambda_nm * 1e-9)
    theta = _THETA
    amp_per_slit = 0.5
    if amps is None:
        amps = np.ones_like(phases)

    intensities = []
    for i, g_um in enumerate(gaps_um):
        d = g_um * 1e-6
        phi = k * d * np.sin(theta) + phases[i]
        field1 = amp_per_slit * np.exp(1j * 0.0)
        # amplitude DOF applies to the second slit multiplicatively (clipped >= 0)
        amp2 = amp_per_slit * max(0.0, float(amps[i]))
        field2 = amp2 * np.exp(1j * phi)
        I = np.abs(field1 + field2) ** 2
        intensities.append(I)
    return np.mean(intensities, axis=0)


def calculate_visibility(I: np.ndarray) -> float:
    I_max = float(np.max(I))
    I_min = float(np.min(I))
    return (I_max - I_min) / (I_max + I_min + 1e-8)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


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


def run_sparse_optimizer(
    gaps_um: List[float],
    I_target: np.ndarray,
    steps: int,
    w: float,
    lr: float,
    noise_scale: float,
    use_pson: bool,
    seed: int,
    objective: str,
    lambda_vis: float,
    amp_dof: bool,
    lr_amp: float,
    lambda_nm_train: float,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    d = len(gaps_um)
    phases = np.zeros(d, dtype=float)
    amps = np.ones(d, dtype=float) if amp_dof else None
    precision, weights = compute_precision_and_weights(gaps_um)

    mses: List[float] = []
    vis_errors: List[float] = []
    accepted = 0
    attempts = 0

    I0 = simulate_intensity_for_gaps(gaps_um, phases, amps=amps, lambda_nm=lambda_nm_train)
    mses.append(mse(I0, I_target))
    vis_errors.append(abs(calculate_visibility(I0) - calculate_visibility(I_target)))

    for _ in range(steps):
        I_cur = simulate_intensity_for_gaps(gaps_um, phases, amps=amps, lambda_nm=lambda_nm_train)
        V_cur = calculate_visibility(I_cur)
        V_t = calculate_visibility(I_target)
        E_mse = mse(I_cur, I_target)
        if objective == "hybrid":
            E_cur = E_mse + lambda_vis * (V_cur - V_t) ** 2
        else:
            E_cur = E_mse

        # Non-local "wormhole" gradient proxy
        grad = -w * E_cur * weights
        proposal = phases - lr * grad
        # Optional amplitude DOF (deterministic update)
        if amp_dof:
            grad_amp = -w * E_cur * weights
            amps_prop = np.clip(amps - lr_amp * grad_amp, 0.0, 10.0)
        else:
            amps_prop = None

        if use_pson:
            delta_perp = project_noise_metric_orthogonal(grad=grad, precision=precision, rng=rng)
            noise = (delta_perp / (np.sqrt(precision) + 1e-12)) * noise_scale
            candidate = proposal + noise
        else:
            candidate = proposal

        attempts += 1
        I_new = simulate_intensity_for_gaps(gaps_um, candidate, amps=amps_prop if amp_dof else amps, lambda_nm=lambda_nm_train)
        V_new = calculate_visibility(I_new)
        E_mse_new = mse(I_new, I_target)
        if objective == "hybrid":
            E_new = E_mse_new + lambda_vis * (V_new - V_t) ** 2
        else:
            E_new = E_mse_new
        if E_new <= E_cur:
            phases = candidate
            if amp_dof:
                amps = amps_prop
            accepted += 1
            mses.append(E_mse_new)
            vis_errors.append(abs(V_new - V_t))
            continue

        if use_pson:
            attempts += 1
            I_det = simulate_intensity_for_gaps(gaps_um, proposal, amps=amps_prop if amp_dof else amps, lambda_nm=lambda_nm_train)
            V_det = calculate_visibility(I_det)
            E_mse_det = mse(I_det, I_target)
            if objective == "hybrid":
                E_det = E_mse_det + lambda_vis * (V_det - V_t) ** 2
            else:
                E_det = E_mse_det
            if E_det <= E_cur:
                phases = proposal
                if amp_dof:
                    amps = amps_prop
                accepted += 1
                mses.append(E_mse_det)
                vis_errors.append(abs(V_det - V_t))
                continue

        mses.append(E_mse)
        vis_errors.append(abs(V_cur - V_t))

    I_final = simulate_intensity_for_gaps(gaps_um, phases, amps=amps, lambda_nm=lambda_nm_train)
    return {
        "phases": phases.tolist(),
        "amps": (amps.tolist() if amp_dof else None),
        "I_final": I_final.tolist(),
        "mse_curve": mses,
        "vis_error_curve": vis_errors,
        "final_mse": float(mses[-1]),
        "final_vis_error": float(vis_errors[-1]),
        "accept_rate": 0.0 if attempts == 0 else accepted / attempts,
    }


def plot_overlay(I_target: np.ndarray, I_sparse_init: np.ndarray, I_sparse_final: np.ndarray, out_path: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(I_target, label="Dense reference", linewidth=2)
    plt.plot(I_sparse_init, label="Sparse init", linestyle="--", alpha=0.8)
    plt.plot(I_sparse_final, label="Sparse final (optimized)", linewidth=2)
    plt.xlabel("Screen index")
    plt.ylabel("Intensity")
    plt.title("Sparse Path Integral Approximation: Intensity Overlay")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)


def random_baseline_mse(
    gaps_um: List[float],
    I_target: np.ndarray,
    trials: int,
    amp_dof: bool,
    lambda_nm: float,
    rng: np.random.Generator,
) -> float:
    best = float("inf")
    d = len(gaps_um)
    for _ in range(trials):
        phases = rng.uniform(-np.pi, np.pi, size=d)
        amps = np.abs(rng.normal(1.0, 0.2, size=d)) if amp_dof else None
        I = simulate_intensity_for_gaps(gaps_um, phases, amps=amps, lambda_nm=lambda_nm)
        best = min(best, mse(I, I_target))
    return best


def run_sweep_sparse_counts(
    sparse_counts: List[int],
    dense: int,
    steps: int,
    w: float,
    lr: float,
    noise: float,
    seed: int,
    use_pson: bool,
    objective: str,
    lambda_vis: float,
    amp_dof: bool,
    lr_amp: float,
    lambda_nm_train: float,
) -> Dict[str, List[float]]:
    rng = np.random.default_rng(seed)
    max_count = int(max(sparse_counts)) if len(sparse_counts) > 0 else 1
    sparse_all = build_sparse_gaps_primes(max_count)
    results = {"sparse_counts": [], "final_mse": [], "rand_mse": []}
    # Build dense reference from the first count's range for consistency per count
    for count in sparse_counts:
        gaps = sparse_all[:count]
        dense_gaps = build_dense_gaps_uniform(gaps, num_dense=dense)
        I_target = simulate_intensity_for_gaps(dense_gaps, np.zeros(len(dense_gaps)), lambda_nm=lambda_nm_train)
        # init sparse
        I_sparse_init = simulate_intensity_for_gaps(gaps, np.zeros(len(gaps)), lambda_nm=lambda_nm_train)
        _ = mse(I_sparse_init, I_target)
        # optimize
        res = run_sparse_optimizer(
            gaps_um=gaps,
            I_target=I_target,
            steps=steps,
            w=w,
            lr=lr,
            noise_scale=noise,
            use_pson=use_pson,
            seed=seed,
            objective=objective,
            lambda_vis=lambda_vis,
            amp_dof=amp_dof,
            lr_amp=lr_amp,
            lambda_nm_train=lambda_nm_train,
        )
        rb = random_baseline_mse(
            gaps_um=gaps,
            I_target=I_target,
            trials=20,
            amp_dof=amp_dof,
            lambda_nm=lambda_nm_train,
            rng=rng,
        )
        results["sparse_counts"].append(count)
        results["final_mse"].append(res["final_mse"])
        results["rand_mse"].append(rb)
    # plot
    plt.figure(figsize=(8, 5))
    plt.plot(results["sparse_counts"], results["final_mse"], marker="o", label="Optimized MSE")
    plt.plot(results["sparse_counts"], results["rand_mse"], marker="x", label="Random-phase baseline MSE")
    plt.xlabel("Number of sparse (prime) paths")
    plt.ylabel("Final MSE vs dense reference")
    plt.title("Sparse Path Integral Approximation: MSE vs number of sparse paths")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("sparse_path_integral_mse_vs_primes.png")
    return results


def main():
    parser = argparse.ArgumentParser(description="Sparse Path Integral Approximation: dense vs sparse with wormhole+PSON")
    parser.add_argument("--dense", type=int, default=200, help="Number of dense uniform gaps for reference")
    parser.add_argument("--sparse_primes", type=int, default=25, help="Number of prime gaps to use")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--w", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_pson", action="store_true", help="Enable PSON exploration")
    parser.add_argument("--objective", type=str, default="mse", choices=["mse", "hybrid"], help="Objective type")
    parser.add_argument("--lambda_vis", type=float, default=0.2, help="Weight for visibility term in hybrid objective")
    parser.add_argument("--amp_dof", action="store_true", help="Enable amplitude DOF per sparse path")
    parser.add_argument("--lr_amp", type=float, default=0.05, help="Learning rate for amplitude DOF")
    parser.add_argument("--cv_lambda_scale", type=float, default=1.01, help="Cross-validation wavelength scale (e.g., 1.01)")
    parser.add_argument("--sparse_list", type=str, default="", help="Comma-separated list of sparse counts to sweep (e.g., 5,10,15,20,25)")
    parser.add_argument("--higher_density_test", action="store_true", help="Preset: dense=10000, sparse_primes=100")
    args = parser.parse_args()

    if args.higher_density_test:
        args.dense = 10000
        args.sparse_primes = 100

    # Build gaps
    if args.sparse_primes < 1:
        raise ValueError("--sparse_primes must be >= 1")
    sparse_all = build_sparse_gaps_primes(args.sparse_primes)
    sparse_gaps = sparse_all
    dense_gaps = build_dense_gaps_uniform(sparse_gaps, num_dense=args.dense)

    # Reference intensity (dense)
    phases_dense = np.zeros(len(dense_gaps), dtype=float)
    I_target = simulate_intensity_for_gaps(dense_gaps, phases_dense)

    # Sparse baseline (init)
    phases_sparse_init = np.zeros(len(sparse_gaps), dtype=float)
    I_sparse_init = simulate_intensity_for_gaps(sparse_gaps, phases_sparse_init)

    # Optimize sparse to match dense
    res = run_sparse_optimizer(
        gaps_um=sparse_gaps,
        I_target=I_target,
        steps=args.steps,
        w=args.w,
        lr=args.lr,
        noise_scale=args.noise,
        use_pson=args.use_pson,
        seed=args.seed,
        objective=args.objective,
        lambda_vis=args.lambda_vis,
        amp_dof=args.amp_dof,
        lr_amp=args.lr_amp,
        lambda_nm_train=633.0,
    )
    I_sparse_final = np.asarray(res["I_final"], dtype=float)

    # Cross-validation on perturbed wavelength
    lambda_cv = 633.0 * float(args.cv_lambda_scale)
    I_target_cv = simulate_intensity_for_gaps(dense_gaps, phases_dense, lambda_nm=lambda_cv)
    I_sparse_cv = simulate_intensity_for_gaps(
        sparse_gaps,
        np.asarray(res["phases"], dtype=float),
        amps=(np.asarray(res["amps"], dtype=float) if args.amp_dof else None),
        lambda_nm=lambda_cv,
    )
    cv_mse = mse(I_sparse_cv, I_target_cv)
    cv_vis_err = abs(calculate_visibility(I_sparse_cv) - calculate_visibility(I_target_cv))

    # Optional sweep
    sweep_results = None
    if args.sparse_list:
        sparse_counts = [int(x) for x in args.sparse_list.split(",") if x.strip()]
        sweep_results = run_sweep_sparse_counts(
            sparse_counts=sparse_counts,
            dense=args.dense,
            steps=args.steps,
            w=args.w,
            lr=args.lr,
            noise=args.noise,
            seed=args.seed,
            use_pson=args.use_pson,
            objective=args.objective,
            lambda_vis=args.lambda_vis,
            amp_dof=args.amp_dof,
            lr_amp=args.lr_amp,
            lambda_nm_train=633.0,
        )

    # Metrics
    init_mse = mse(I_sparse_init, I_target)
    final_mse = res["final_mse"]
    target_vis = calculate_visibility(I_target)
    init_vis = calculate_visibility(I_sparse_init)
    final_vis = calculate_visibility(I_sparse_final)

    # Artifacts
    plot_overlay(I_target, I_sparse_init, I_sparse_final, out_path="sparse_path_integral_overlay.png")
    summary = {
        "params": {
            "dense": args.dense,
            "sparse_primes": args.sparse_primes,
            "steps": args.steps,
            "w": args.w,
            "lr": args.lr,
            "noise": args.noise,
            "seed": args.seed,
            "use_pson": args.use_pson,
            "objective": args.objective,
            "lambda_vis": args.lambda_vis,
            "amp_dof": args.amp_dof,
            "lr_amp": args.lr_amp,
            "cv_lambda_scale": args.cv_lambda_scale,
        },
        "metrics": {
            "init_mse": float(init_mse),
            "final_mse": float(final_mse),
            "mse_improvement": float(init_mse - final_mse),
            "target_visibility": float(target_vis),
            "init_visibility": float(init_vis),
            "final_visibility": float(final_vis),
            "accept_rate": float(res["accept_rate"]),
            "cv_mse": float(cv_mse),
            "cv_vis_error": float(cv_vis_err),
        },
        "sweep": sweep_results,
    }
    with open("sparse_path_integral_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Sparse Path Integral Approximation Summary ===")
    print(json.dumps(summary, indent=2))
    print("Artifacts: sparse_path_integral_overlay.png, sparse_path_integral_summary.json")


if __name__ == "__main__":
    main()


