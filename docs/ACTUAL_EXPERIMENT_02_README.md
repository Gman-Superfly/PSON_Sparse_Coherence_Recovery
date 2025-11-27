## ACTUAL EXPERIMENT 02 — Zeta‑Coupled Vector Homeostat (Wormhole + PSON)

### TL;DR
- We add a zeta‑driven phase term to the optical interference model and optimize per‑gap phases with a vector Homeostat.
- Updates combine a non‑local Wormhole gradient (Eq. 3) and precision‑aware orthogonal exploration (PSON) with a down‑only acceptance guard.
- On irregular prime gaps, PSON improves final visibility over a deterministic baseline; on uniform gaps, visibility is already ~1.0.

---

## What we are doing and why
We test whether non‑local credit assignment (Wormhole) plus safe exploration (PSON) can recover coherence in an optical proxy where irregular sampling (primes) introduces aliasing. Unlike Experiment 01 (geometric phase only), Experiment 02 injects structure from number theory by modulating phases with a zeta signal, then lets the Homeostat learn per‑gap phase corrections.

Why this matters:
- The zeta term creates a structured, potentially rough energy landscape (especially with prime gaps).
- The vector Homeostat provides the mechanism to navigate that landscape and “patch” coherence, mirroring the paper’s non‑local correction story.

---

## Minimal theory
- Energy: F = (1 − Visibility)². Minimize F → maximize interference visibility.
- Wormhole (Eq. 3): ∂F/∂η_gate = −w · Δ_benefit. Update a gate based on downstream benefit, independent of its current value.
- PSON (Eq. 1): Inject noise in the subspace orthogonal to the gradient (with metric M = diag(Λ)), scaled by inverse precision. Accept only if F decreases.
- Precision/weights: Derived from gap irregularity (proxy for uncertainty/importance). Irregular gaps get lower precision (more exploration) and higher wormhole weight (more credit).

---

## Experimental design

### Zeta‑coupled optics
- Per‑gap phase: φ_i(x) = k d_i sin(θ) + ζ_gain · Re(ζ(1/2 + i t_i)) + η_i
- t_i = (gap_i / max_gap) · t_scale. If `mpmath` is present we use true ζ; otherwise a zeta‑like sine mixture fallback.

### State parameterization
- Vector of per‑gap phases η ∈ ℝ^25 (first 25 primes). This enables true PSON (orthogonal exploration), unlike the 1D case.

### Update and guards
- Gradient: g_i = −w · Δ_benefit_i, with Δ_benefit_i ∝ weight_i · current_energy.
- Deterministic step: η ← η − lr · g
- PSON proposal: δ ← proj_{⊥_M} N(0, I), scaled by 1/√Λ and `noise` → candidate = deterministic + noise
- Acceptance: down‑only; if noisy candidate rejected, try deterministic; otherwise keep current state.

### Metrics and artifacts
- Final visibility V_final; ΔF90 steps to reach 90% of total energy drop; acceptance rate.
- Figure: `actual_experiment_002_energy.png`
- Script: `actual_experiment_002.py`

---

## How to run (Windows PowerShell)

```powershell
python .\actual_experiment_002.py --steps 200 --w 0.2 --lr 0.1 --noise 0.02 --zeta_gain 0.2 --seed 42
```

Outputs:
- Console JSON summary (uniform/primes; baseline, final V, ΔF90, acceptance)
- Plot: `actual_experiment_002_energy.png`

---

## Results (this run)
Parameters: steps=200, w=0.2, lr=0.1, noise=0.02, zeta_gain=0.2, seed=42.

Summary:

- Uniform gaps
  - baseline V: 0.99999977
  - final V (no‑PSON): 0.99999977
  - final V (PSON): 0.99999977
  - ΔF90: 0 (both)
  - acceptance: no‑PSON 1.00, PSON 0.50

- Prime gaps (first 25 primes × 10 μm)
  - baseline V: 0.40290248
  - final V (no‑PSON): 0.40290248
  - final V (PSON): 0.52780533
  - ΔF90: no‑PSON 0, PSON 182
  - acceptance: no‑PSON 0.00, PSON 0.299

Interpretation:
- Uniform is already optimal; updates cannot materially improve visibility.
- For primes, PSON raises final visibility notably (≈0.40 → ≈0.53), at the cost of more steps to reach 90% of the total energy drop due to exploration and rejections—consistent with “explore to find better minima.”

---

## Why tuning matters (zeta_gain, lr, noise)
- zeta_gain: sets the strength of the ζ term; too low → weak guidance, too high → chaotic curvature. Proper gain yields informative structure with manageable gradients.
- lr: controls deterministic progress vs overshoot/rejects. Right‑sized lr lowers ΔF90 without hurting V_final.
- noise: balances exploration and acceptance. Sufficient to escape poor basins (higher V_final), not so large that most steps reject (worse ΔF90).

Goal: higher V_final with fewer steps on primes by co‑tuning these three.

---

## Limitations & notes
- Precision/weights from gap irregularity are proxies; curvature‑based Λ or per‑layer SNR may be better.
- The ζ component here is gap‑indexed (t_i from gaps). A screen‑position‑dependent ζ modulation is a possible extension.
- Down‑only acceptance enforces monotonicity; alternative line‑search or small‑gain capping can be added for smoother progress.

---

## Next steps
1) Tuning sweep: small grid over {zeta_gain, lr, noise} on primes; track V_final, ΔF90, acceptance.
2) Robust stats: multiple seeds with CIs and effect sizes for ΔV.
3) Alternative precisions: try curvature estimates for Λ; compare to irregularity proxy.
4) RH sweeps: on‑line vs off‑line zeros and impact on final V under the same controller.

---

## Alignment with Datamutant rules
- Assert early / fail fast: monotone acceptance; deterministic fallback if PSON candidate fails.
- One function, one purpose: simulate, visibility, energy, projection, loop kept modular in the script.
- Event‑driven/observable: metrics printed; figure saved for immediate inspection.

---

## Citation
If you use this repository in your research, please cite it as below.

**Authors:** Oscar Goldman — Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業

```bibtex
@software{actual_experiment_02_2025,
  title        = {ACTUAL EXPERIMENT 02 — Zeta-Coupled Vector Homeostat (Wormhole + PSON)},
  author       = {Goldman, Oscar},
  organization = {Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業},
  year         = {2025},
  note         = {Zeta-coupled phases; precision-orthogonal exploration with down-only acceptance; reproducible results}
}
```


