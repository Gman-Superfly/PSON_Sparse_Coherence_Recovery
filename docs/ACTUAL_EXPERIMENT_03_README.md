## ACTUAL EXPERIMENT 03 — Tuning zeta_gain, lr, and noise on Prime Gaps

### TL;DR
- We tune three knobs in the zeta‑coupled vector Homeostat (Experiment 02): `zeta_gain` (ζ strength), `lr` (wormhole step), and `noise` (PSON scale).
- Goal: maximize final visibility on prime gaps while keeping steps to convergence reasonable (ΔF90 small) under the down‑only acceptance guard.
- Result (seed=42, 200 steps): best config found on the default grid is `zeta_gain=0.3, lr=0.1, noise=0.03`, reaching V≈0.594 (from baseline ≈0.403).

---

## What and why
Experiment 02 introduced a zeta‑driven phase term and a vector Homeostat (per‑gap phases) with Wormhole updates and PSON exploration. Experiment 03 systematically tunes the controller’s key hyperparameters to:

- improve coherence recovery (higher final visibility V), and
- reduce the number of steps to achieve most of the energy drop (lower ΔF90),

on the rougher “prime gaps” landscape where aliasing is strongest. This strengthens the claim that non‑local credit + precision‑aware exploration can efficiently “patch” aliasing‑induced breaks.

---

## Method (tuning setup)
- Model: same as Experiment 02 (zeta‑coupled optics; per‑gap phases η ∈ ℝ²⁵; Wormhole gradient; PSON with metric‑orthogonal projection; down‑only acceptance).
- Grid (default):  
  - `zeta_gain ∈ {0.1, 0.2, 0.3}`  
  - `lr ∈ {0.05, 0.1, 0.15}`  
  - `noise ∈ {0.01, 0.02, 0.03}`  
  - total 27 configs.
- Scoring: prioritize final visibility with a mild penalty on steps to 90% energy drop (ΔF90):
  - Score = final_V_pson − 0.2 · (ΔF90_pson / steps)
- Metrics recorded per config:
  - baseline V (no updates, phases=0)
  - final V (no‑PSON), final V (PSON)
  - ΔF90 (no‑PSON), ΔF90 (PSON)
  - acceptance rates

---

## How to run (Windows PowerShell)

```powershell
python .\actual_experiment_003.py --steps 200 --w 0.2 --zeta_gains 0.1,0.2,0.3 --lrs 0.05,0.1,0.15 --noises 0.01,0.02,0.03 --seed 42
```

Artifacts:
- `actual_experiment_003_results.csv` — full grid results
- `actual_experiment_003_best_energy.png` — best config energy curves (PSON vs no‑PSON)
- `actual_experiment_003_summary.json` — best config and run metadata

---

## Results (this run)
Parameters: steps=200, w=0.2, seed=42.

Best config on the default grid:

- **zeta_gain**: 0.3  
- **lr**: 0.1  
- **noise**: 0.03  
- **baseline V**: 0.40290248  
- **final V (no‑PSON)**: 0.40655495  
- **final V (PSON)**: 0.59363614  
- **ΔF90 (no‑PSON)**: 0  
- **ΔF90 (PSON)**: 182  
- **acceptance (PSON)**: 0.299  
- **grid size**: 27  
- **mpmath**: true (true ζ used)

Interpretation:
- Prime gaps need exploration to escape poor basins: PSON significantly raises final V (≈0.594 vs ≈0.407 deterministic).  
- ΔF90 for PSON is higher (more steps) due to rejected proposals under the monotone guard—expected when exploring a rough surface. The score balances this trade‑off.

---

## Why tuning these three knobs helps
- **zeta_gain** adjusts the “shape” and curvature injected by ζ into the phase landscape. Right‑sizing it yields informative guidance without overwhelming the controller.
- **lr** sets deterministic progress speed; too small is slow (high ΔF90), too large causes rejections/oscillations under down‑only acceptance.
- **noise** (PSON scale) enables safe exploration orthogonal to the gradient; sufficient to escape poorer minima, but not so large that most proposals are rejected.

Together they determine quality (V_final) and efficiency (ΔF90, acceptance) on the primes landscape.

---

## Limitations & notes
- The precision/weight scheme uses gap irregularity as a proxy; curvature‑based Λ or SNR‑aware estimates may further stabilize/accelerate convergence.
- Score weights (0.2) are heuristic; adjusting them trades off quality vs speed.
- The grid is small; finer or adaptive searches can improve results and wall‑clock.

---

## Next steps
1) Narrow grid around the best config (e.g., zeta_gain ∈ {0.25, 0.3, 0.35}, lr ∈ {0.1, 0.12, 0.15}, noise ∈ {0.02, 0.025, 0.03}).  
2) Multi‑seed robustness with confidence intervals and effect sizes.  
3) Explore `w` (wormhole gain) and alternative precision models.  
4) Extend to RH sweeps (on‑line vs off‑line zeros) under tuned controller.


=====================================================================================

NOTES:
we can proceed with 004 now; refinements can come later.
What refinements mainly change
Narrow grid: improves efficiency (lower ΔF90), not the qualitative outcome.
Multi-seed: gives CIs and robustness; doesn’t alter the mechanism.
w/precision variants: shift final V and acceptance; unlikely to flip on‑line vs off‑line ordering if params are reasonable.
Risk if we skip: under/over‑exploration could dampen effect sizes and yield weaker separation, not false positives.
Recommended path
Run 004 now with the current best config from 003: zeta_gain=0.3, lr=0.1, noise=0.03, w=0.2.
Include a quick robustness check inside 004:
3 seeds (e.g., 41, 42, 43).
Tiny perturbations around best (±0.05 zeta_gain, ±0.02 lr, ±0.01 noise).
Do the full refinements after, for polished stats.

=====================================================================================

---

## Alignment with Datamutant rules
- **Assert/Failsafe**: monotone acceptance with deterministic fallback guarantees non‑increasing energy.  
- **One function, one purpose**: simulation, visibility, energy, projection, and optimization are cleanly separated in the scripts.  
- **Observability**: per‑run artifacts and CSV logs enable rapid inspection and comparison.

---

## Citation
If you use this repository in your research, please cite it as below.

**Authors:** Oscar Goldman — Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業

```bibtex
@software{actual_experiment_03_2025,
  title        = {ACTUAL EXPERIMENT 03 — Tuning zeta\_gain, lr, and noise on Prime Gaps},
  author       = {Goldman, Oscar},
  organization = {Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業},
  year         = {2025},
  note         = {Vector Homeostat (Wormhole + PSON); grid tuning with reproducible artifacts}
}
```


