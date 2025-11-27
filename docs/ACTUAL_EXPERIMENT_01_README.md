## ACTUAL EXPERIMENT 01 — Homeostat Wormhole + PSON on Optical Interference

### TL;DR
- We implement a dynamic “Homeostat” that optimizes interference visibility in a double‑slit proxy by adjusting phase “gates.”
- The update law follows the Wormhole gradient from the Homeostat paper (non‑local credit; Eq. 3), plus precision‑aware exploration (PSON) projected orthogonal to the gradient.
- In 1D (single global phase), true PSON is not meaningful; we therefore moved to a vector model (one phase per gap) where PSON is valid.
- Results: On irregular prime gaps, PSON improves final visibility over a deterministic baseline; on uniform gaps, both converge to ~1.0 as expected.

---

## Why we’re doing this
Our broader program explores how “breaks” in a finite‑compute causal lattice (e.g., irregular prime sampling) can be “patched” by non‑local mechanisms (“wormholes”). The Neuro‑Symbolic Homeostat provides a grounded control‑theory mechanism for such non‑local credit assignment while maintaining stability and monotone energy descent.

This experiment brings that theory into a concrete, reproducible optical proxy:
- Aliasing mechanism: irregular prime gaps fold phase contributions and reduce fringe visibility.
- Non‑local patch: a Homeostat adjusts phase “gates” using a gradient independent of the current gate value (Wormhole, Eq. 3) and explores safely with precision‑scaled orthogonal noise (PSON).
- Stability: down‑only acceptance (monotone energy), precision‑aware steps.

---

## Theory primer (minimal)
- Energy: We minimize F = (1 − Visibility)^2. Higher visibility ⇒ lower energy.
- Wormhole (Eq. 3): ∂F/∂η_gate = −w · Δ_benefit. The gradient on a gate is driven by downstream benefit (non‑local), independent of the current η_gate.
- PSON (Eq. 1): Inject exploration noise δ in the subspace orthogonal to the gradient (with respect to a metric M), scaled by inverse precision. This preserves monotone descent under a down‑only acceptance rule or sufficiently small steps.
- Small‑gain/monotonicity: We accept a candidate step only if energy decreases; otherwise we back off to a deterministic proposal or reject.

Implication for 1D: In one dimension there is no non‑trivial subspace orthogonal to the gradient; true PSON is therefore degenerate. This motivates a multi‑parameter phase vector.

---

## Experimental design

### Optical model
- Two‑slit interference on a screen; we compute intensity I(x) from the superposition of two fields per “gap” and average over gaps.
- Gaps: either uniform (control) or primes×10μm (irregular).
- Visibility V = (I_max − I_min) / (I_max + I_min).

### State parameterization
- 1D (demo): `actual_experiment.py` evolves a single global phase. Useful to illustrate the loop but not suitable for PSON.
- Vector (primary): `homeostat_vector_test.py` evolves a phase per gap (25D for first 25 primes), enabling meaningful PSON.

### Objective and updates
- Energy: F = (1 − V)^2.
- Wormhole gradient (vector): g_i = −w · Δ_benefit_i. We set Δ_benefit_i ∝ weights_i · current_energy, where weights are derived from gap irregularity (larger weight for more irregular contributors).
- Precision Λ (diagonal): derived from gap irregularity; stiffer for regular gaps, slacker for irregular ones. Updates scale with Λ.
- PSON: draw z ~ N(0, I), project metric‑orthogonal to g (metric M = diag(Λ)), then scale by 1/√Λ and a global noise factor. Candidate accepted only if energy decreases; otherwise try deterministic proposal; otherwise reject.

### Metrics
- Final visibility V_final.
- ΔF90 steps: number of steps to reach 90% of the total energy drop (lower is faster).
- Acceptance rate: accepted / attempted proposals.

---

## Files and how to run (Windows PowerShell)

```powershell
# 1) One‑dimensional demo (global phase)
python .\actual_experiment.py

# Artifacts:
# - actual_experiment_trajectory.png (visibility vs relaxation steps)

# 2) Vector ablation (per-gap phases; PSON vs No‑PSON)
python .\homeostat_vector_test.py --steps 200 --w 0.2 --lr 0.1 --noise 0.02 --seed 42

# Artifacts:
# - homeostat_vector_ablation.png (energy curves for uniform/primes, PSON/no‑PSON)
# - Printed JSON summary in console (final V, ΔF90, acceptance rates)
```

Dependencies: Python 3.12+, NumPy, Matplotlib, SciPy (see repository for setup).

---

## Findings (seed=42, steps=200, w=0.2, lr=0.1, noise=0.02)

### Summary table

| Config   | V_final (No‑PSON) | V_final (PSON) | ΔF90 (No‑PSON) | ΔF90 (PSON) | Accept (No‑PSON) | Accept (PSON) |
|----------|-------------------:|---------------:|---------------:|------------:|-----------------:|--------------:|
| Uniform  | 0.99996            | 0.99996        | 181            | 181         | 1.00             | 0.50          |
| Primes   | 0.39746            | 0.53904        | 176            | 186         | 1.00             | 0.678         |

Notes:
- Uniform gaps: both methods converge to near‑perfect visibility (as expected in a smooth landscape). PSON neither helps nor hurts final V here.
- Prime gaps (irregular): PSON improves final visibility (0.54 vs 0.40), at a modest cost in ΔF90 (slower to 90% of total drop due to exploratory rejections), with a healthy acceptance rate.

Interpretation:
- In a rough, irregular landscape, orthogonal exploration helps avoid poor basins and ends at a lower energy (higher coherence). This supports the core claim that non‑local corrections plus safe exploration can “patch” aliasing‑induced breaks more effectively than a purely deterministic descent.

---

## What this establishes (and what it doesn’t)
Established:
- A faithful implementation of the Homeostat principles in a compact optical proxy: wormhole gradient, precision‑aware orthogonal exploration (PSON), and monotone acceptance.
- Clear behavioral difference between smooth (uniform) and rough (primes) regimes, with PSON providing measurable benefit in the latter.

Not yet established:
- Zeta‑coupled phase modulation (φ(t, ζ)): current optical experiments use geometric phase; zeta coupling will be added next.
- RH‑style sweeps (on‑line vs off‑line zeros) and their impact on aliasing/visibility within this vector framework.

---

## Limitations & safeguards
- 1D PSON is degenerate; use vector phases to enable true orthogonal exploration.
- Acceptance guard is essential for monotonicity; we also provide deterministic fallback when PSON proposals are rejected.
- Precision/weights from gap irregularity are proxies; richer precision models (e.g., curvature estimates, per‑layer SNR) may improve performance.

---

## Next steps
1) Zeta‑coupled optics: modulate phase by Re(ζ(½ + i t)) as documented in the main writeups; ablate geometric vs zeta‑coupled runs.
2) Parameterization study: compare per‑gap vs per‑layer phases (expressivity vs overfit) with the same PSON/guard machinery.
3) Statistical robustness: bootstrap confidence intervals for V_final and ΔV across seeds; add effect sizes.
4) Tuning sweeps: grid over {lr, noise, w}; target ΔF90 improvements without sacrificing V_final under primes.

---

## Alignment with Datamutant system rules
- Assert early, fail fast: energy monotonicity via down‑only acceptance; deterministic fallback; explicit checks on dimensionality (PSON only in ≥2D).
- One function, one purpose: separate simulate, visibility, energy, projection, and loop logic.
- Type everything: functions in `homeostat_vector_test.py` are structured for explicit typing and straightforward extension.
- Event‑driven/observable: artifacts saved per run; JSON summary printed for quick analysis.

---

## Citation
If you use this repository in your research, please cite it as below.

**Authors:** Oscar Goldman — Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業

```bibtex
@software{homeostat_actual_experiment_01_2025,
  title        = {ACTUAL EXPERIMENT 01 — Homeostat Wormhole + PSON on Optical Interference},
  author       = {Goldman, Oscar},
  organization = {Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業},
  year         = {2025},
  note         = {Vector homeostat with PSON and monotone acceptance; reproducible ablation}
}
```

---

## Appendix: Parameters (this run)
- steps: 200
- w (wormhole gain): 0.2
- lr (learning rate): 0.1
- noise (PSON scale): 0.02
- seed: 42
- gaps: first 25 primes × 10μm (irregular) and uniform 100μm (control)


