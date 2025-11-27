## ACTUAL EXPERIMENT 05 (FAST) — x‑Dependent ζ With Caching and Progress Logging

### TL;DR
- This is the fast version of Experiment 005. It evaluates x‑dependent zeta modulation
  `Zx(x) = Re(ζ(σ + i t_x))` and caches `Zx` per `(σ, t_scale)` so it is computed once,
  then reused across all steps and runs.
- If `mpmath` is available we use true ζ; otherwise a ζ‑like sinusoidal fallback
  is used. You can force the fallback via `--no_mpmath`.
- Robust grid over `zeta_gain`, `lr`, `noise`, `seeds`, and multiple σ values,
  with lightweight progress logging and summarized artifacts.

---

## What this test does (differences vs experiment 005 SLOW)

Experiment 005 varies σ (e.g., 0.50, 0.55, 0.60, 0.65) while driving per‑gap phases with an
x‑dependent ζ term:

```text
t_x = linmap(x ∈ [x_min, x_max] → [0, t_scale])
Zx(x) = Re( ζ(σ + i t_x) )
φ_i(x) = k d_i sin(θ) + zeta_gain * Zx(x) + η_i
```

The FAST variant:
- Precomputes `Zx(x)` once per `(σ, t_scale, use_mpmath)` and caches it, avoiding repeated ζ calls in the inner loop.
- Adds `--no_mpmath` flag to force the fast fallback even if `mpmath` is installed.
- Adds `--progress_every` to print periodic run progress.
- Produces plots of Final‑V vs σ and mean energy curves across seeds for the base config.

Controller (unchanged): Vector Homeostat with non‑local credit (“Wormhole”), precision‑scaled orthogonal noise (PSON), and down‑only acceptance with deterministic fallback.

---

## How to run (Windows PowerShell)

Default grid (uses true ζ if available; cache enabled):
```powershell
python .\actual_experiment_005_fast.py
```

Force fast fallback (no mpmath calls):
```powershell
python .\actual_experiment_005_fast.py --no_mpmath
```

Quiet progress (disable progress prints):
```powershell
python .\actual_experiment_005_fast.py --progress_every 0
```

Custom grid (edit knobs as needed):
```powershell
python .\actual_experiment_005_fast.py --steps 200 `
  --w 0.2 --zeta_gains 0.3,0.5,0.7 --lrs 0.10,0.12,0.15 --noises 0.02,0.03 `
  --seeds 41,42,43 --progress_every 20
```

Quick smoke test (shorter steps):
```powershell
python .\actual_experiment_005_fast.py --steps 50 --zeta_gains 0.5 --lrs 0.12 --noises 0.02 --seeds 41
```

---

## Artifacts
- `actual_experiment_005_fast_results.csv` — full grid results (σ, zeta_gain, lr, noise, seed; final_V, ΔF90, accept_rate)
- `actual_experiment_005_fast_summary.json` — aggregated metrics per σ and per zeta_gain, base‑config mean energy curves, run parameters, and whether true ζ was used
- `actual_experiment_005_fast_finalV_vs_sigma.png` — error‑bar plot of mean Final‑V vs σ broken out by zeta_gain
- `actual_experiment_005_fast_energy_mean.png` — mean energy curve across seeds for the base config at each σ

---

## Parameters & flags (glossary)
- Core run:
  - `--steps` (default 200), `--w` (default 0.2)
  - `--zeta_gains` (e.g., 0.3,0.5,0.7), `--lrs` (e.g., 0.10,0.12,0.15), `--noises` (e.g., 0.02,0.03)
  - `--seeds` (e.g., 41,42,43)
- Performance / control:
  - `--no_mpmath`: force fast fallback (synthetic zeta‑like signal)
  - `--progress_every`: print progress every N completed runs (0 disables)

---

## Interpreting results (typical)
- Final visibility tends to increase with `zeta_gain`; at higher `zeta_gain` the controller
  largely compensates σ variation within 0.50–0.65.
- ΔF90 (steps to 90% energy drop) often improves (fewer steps) at higher `zeta_gain`; PSON acceptance
  remains moderate to high, indicating stable exploration.
- If you need stronger σ separation, widen σ range and/or adjust `t_scale`; 005 FAST makes those sweeps feasible by caching `Zx`.

---

## Limitations & notes
- True ζ via `mpmath` remains CPU‑intensive; caching removes per‑step overhead but the initial ζ grid computation still costs time proportional to screen length.

Our test ran:
“[fast-005] Using true ζ via mpmath (cached Zx per σ)” and the summary JSON has "mpmath": true.

NOTE:
- The synthetic fallback in the code is an approximation; the summary JSON records whether true ζ or fallback was used. Interpret results accordingly.
- Mean energy curves are reported for the base config only; include further configs if you want broader descent profiles.

It Exist just incase you want to run an enormous test and compute power is not available.

How it works:
We build a zeta‑like waveform from a small set of known Riemann zero frequencies and sum sines at those frequencies, then apply a simple σ‑dependent damping:
Zx_fallback(x) ≈ (0.5/σ) · Σk sin(2π · t_x · ρk)
ρk are the first few nontrivial zeros (e.g., 14.134725, 21.022040, …)
This preserves the “structured roughness” we need to stress the controller (multi‑scale, correlated), without doing heavy complex ζ(σ + i t) evaluations.
Why it exists:
True ζ via mpmath is expensive; evaluating it at hundreds of screen points for many runs can take hours.
The fallback keeps the experiment fast while retaining the key property we test (controller robustness on a structured, rough landscape).
When it’s used:
If mpmath is not installed, or you pass --no_mpmath.
If you want true ζ everywhere:
Install mpmath and don’t pass --no_mpmath

    python -m pip install mpmath
    python .\actual_experiment_005_fast.py

The summary JSON includes "mpmath": true/false so you can see which path ran.
---

## Citation
If you use this repository in your research, please cite it as below.

**Authors:** Oscar Goldman — Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業

```bibtex
@software{actual_experiment_05_fast_2025,
  title        = {ACTUAL EXPERIMENT 05 (FAST) — x-Dependent ζ With Caching and Progress Logging},
  author       = {Goldman, Oscar},
  organization = {Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業},
  year         = {2025},
  note         = {Zeta-coupled x-dependent modulation with cached Zx; reproducible σ sweeps and summarized artifacts}
}
```

