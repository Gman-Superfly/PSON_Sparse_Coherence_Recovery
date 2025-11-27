## ACTUAL EXPERIMENT 04 — RH Sweep: On‑line (σ=0.5) vs Off‑line (σ=0.6) with Vector Homeostat (Wormhole + PSON)

### TL;DR
- We evaluate sensitivity to the Riemann–Hilbert “critical line” by comparing ζ‑coupled runs at σ=0.5 (on‑line) vs σ=0.6 (off‑line).
- Controller: Vector Homeostat with non‑local credit (“Wormhole”) and precision‑scaled orthogonal exploration (PSON), down‑only acceptance.
- Robustness: 3 seeds and small parameter perturbations around the tuned config from Exp 03.
- Findings: Final visibility improves monotonically with higher zeta_gain; σ effects are mild in 0.50–0.60. At strong ζ gain, the controller largely compensates σ variation.
- Artifacts: CSV of all runs, JSON summary, bar plot of mean final visibility per σ, and mean energy curves across seeds.

---

## What and why
Experiment 03 tuned the ζ‑coupled controller and identified regimes where PSON materially improves final visibility on the “prime gaps” landscape. In Exp 04 we ask a focused question:

> Does operating “on‑line” (σ=0.5) vs “off‑line” (σ=0.6) change outcomes under the same controller?

We keep the model and controller fixed, vary σ between 0.5 and 0.6, and assess:
- Final visibility (primary quality metric)
- ΔF90 steps (convergence speed to 90% of total energy drop)
- Acceptance rate (stability/efficiency)

We also average the energy curve across seeds for the base configuration to visualize typical descent.

---

## Design

### Zeta coupling per gap
For a given σ, we build a per‑gap ζ signal:

```text
zeta_re[i] = Re( ζ(σ + i t_i) ),  where  t_i = (gap_i / max_gap) * t_scale
```

If `mpmath` is available, we compute true ζ; otherwise we use a ζ‑like sinusoidal fallback at known zero frequencies, with σ‑dependent damping. The ζ term is added to each per‑gap phase:

```text
φ_i(x) = k d_i sin(θ) + zeta_gain * zeta_re[i] + η_i
```

### Controller (vector Homeostat)
- Non‑local credit (“Wormhole”): `grad_i = −w * benefit * weight_i`, where `benefit = current_energy` and `weight_i` derives from gap irregularity.
- Exploration (PSON): metric‑orthogonal noise scaled by inverse precision, added to the deterministic proposal.
- Acceptance: down‑only; if noisy candidate is rejected but deterministic proposal improves energy, accept the deterministic step; else reject.

### Robustness setup
- σ values: 0.5 (on‑line), 0.6 (off‑line)
- Base config: from Exp 03 (zeta_gain≈0.3, lr≈0.1, noise≈0.03, w=0.2)
- Perturbations: `zeta_gain ∈ {base±0.05}`, `lr ∈ {base±0.02}`, `noise ∈ {base±0.01}`
- Seeds: `{41, 42, 43}`

We aggregate metrics per σ across all seeds and perturbations, and we compute the mean energy curve for the base config across seeds (per σ).

---

## How to run (Windows PowerShell)

```powershell
# Default run (200 steps; base_zeta_gain=0.3, base_lr=0.1, base_noise=0.03; seeds 41,42,43)
python .\actual_experiment_004.py

# Customize step count or base parameters
python .\actual_experiment_004.py --steps 300 --base_zeta_gain 0.35 --base_lr 0.12 --base_noise 0.03 --seeds 41,42,43

# If you do not have mpmath or want the fast fallback explicitly
python .\actual_experiment_004.py  # (the script will note fallback if mpmath is unavailable)
```

Artifacts produced:
- `actual_experiment_004_results.csv` — per‑run metrics (σ, zeta_gain, lr, noise, seed; final_V, ΔF90, accept_rate)
- `actual_experiment_004_summary.json` — aggregated metrics per σ, grid size, params, and `mpmath` availability
- `actual_experiment_004_finalV_bar.png` — mean final visibility per σ (bars with standard deviations)
- `actual_experiment_004_energy_mean.png` — mean energy curves across seeds for the base config (per σ)

---

## Results (representative)
- Final visibility increases with `zeta_gain` at both σ values (0.5 and 0.6).
- The difference between σ=0.5 (on‑line) and σ=0.6 (off‑line) is small within this range; the tuned controller largely compensates σ variation, especially at higher `zeta_gain`.
- ΔF90 tends to improve (fewer steps) at higher `zeta_gain` under these settings, with healthy acceptance rates.

Interpretation: Within σ ∈ [0.5, 0.6] under the tuned controller, σ sensitivity is **modest**. To expose stronger σ‑dependent behavior, consider widening σ (e.g., 0.45–0.80), adjusting the screen‑position mapping `t_scale`, or using the x‑dependent ζ modulation (see Exp 05) which amplifies σ effects.

---

## Limitations & notes
- Accurate ζ is compute‑intensive; we cache per‑σ signals within a run. If `mpmath` is unavailable, a fast sinusoidal fallback is used (documented in console and summary JSON).
- σ range and perturbations are intentionally small to isolate controller behavior; larger sweeps may reveal stronger σ dependence.
- Precision/weights based on gap irregularity are a proxy; curvature or SNR‑based precision may further stabilize/accelerate convergence.

---

## Next steps
1) Widen σ range and adjust `t_scale` to probe stronger σ sensitivity.  
2) Use Exp 05’s x‑dependent ζ modulation for amplified σ effects.  
3) Increase seeds and report confidence intervals for effect sizes.  
4) Explore alternative precision models (e.g., curvature estimates).  

---

## Citation
If you use this repository in your research, please cite it as below.

**Authors:** Oscar Goldman — Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業

```bibtex
@software{actual_experiment_04_2025,
  title        = {ACTUAL EXPERIMENT 04 — RH Sweep: On-line (σ=0.5) vs Off-line (σ=0.6) with Vector Homeostat (Wormhole + PSON)},
  author       = {Goldman, Oscar},
  organization = {Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業},
  year         = {2025},
  note         = {Zeta-coupled per-gap phases; robustness across seeds and perturbations; σ sensitivity}
}
```

