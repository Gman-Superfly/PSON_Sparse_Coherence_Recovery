# Sparse Coherence Recovery via PSON

**Validation of Precision-Scaled Orthogonal Exploration on Irregular Optical Arrays**


**Authors:** Oscar Goldman
**Date:** November 2025  
**Status:** Working code with reproducible experiments


## What this repository contains

### Contribution
This repository tests **PSON (Precision-Scaled Orthogonal Noise)** on sparse optical coherence and related discrete phase-control tasks. The strongest result is the fair optical-coherence suite, where PSON improved visibility over the deterministic non-local descent baseline under the stated 601-evaluation budget:

- `20/20` scenarios improved across 5 signal types, 2 coupling modes, and 2 dependency types.
- Both methods used 601 `simulate_fn` calls in the fair comparison mode.
- The multi-seed validation reported 95% CI `[+0.103, +0.185]` for the mean gain across 50 runs.
- In 9/20 scenarios the deterministic baseline stayed at 0% acceptance from the zero-phase initialization, work in progress...

### Results

| Application | PSON Performance | Key Finding |
|-------------|------------------|-------------|
| **Optical coherence** | 20/20 scenarios improved (+0.03 to +0.16 visibility) | Tested against deterministic non-local descent with equal budgets |
| **Static beamforming** | PSON-Subspace wins 3/3 (MSE: 0.05 vs 127) | Reported in the matched-initialization LMS comparison |
| **Moving target tracking** | PSON wins 2/3 (67%) | Measured in the matched-initialization dynamic test |
| **Adaptive jammer nulling** | LMS wins 3/3  | **PSON limitation identified** |
| **Massive MIMO (1024-2048)** | PSON-Subspace wins 2/3 | Benefit appears in this tested size range |

### Experimental Suite

**Core experiments:**
- `airtight_experiments_001.py` - 20-scenario validation (Section 6.1)
- `multi_seed_validation_001.py` - Statistical significance (Section 6.4)
- `baseline_comparison_001.py` - PSON vs CMA-ES/SA/Random (Section 6.5)
- `partial_observability_test_001.py` - Robustness under noise/quantization (Section 6.6)

**Discrete applications:**
- `phased_array_antenna_test.py` - 5G/radar beamforming (Section 7.2.1)
- `pson_vs_lms_fair_comparison.py` - PSON vs LMS fair comparison
- `pson_dynamic_scenarios_test.py` - Moving targets, massive MIMO (Section 7.2.1.2)
- `holographic_beam_steering_test.py` - LiDAR beam steering (Section 7.2.2)

**Analysis tools:**
- `sparse_path_integral_test.py` - Path integral approximation (Section 7.3)
- `pson_optical_scaling_test.py` - Optical scaling (100-4096 elements, Section 7.1.1)
- `pson_dynamic_scenarios_test.py` - Beamforming scaling (256-8192 elements, Section 7.2.1.2)
- `speed_benchmark.py` - Wall-clock performance vs CMA-ES (Section 6.5)

See [`docs/`](docs/) for detailed READMEs per experiment.

---

## The PSON algorithm

**TL;DR:** Orthogonal noise with a down-only acceptance guard.

### Core Loop
```python
for iteration in range(steps):
    # 1. Measure global state
    E_cur = energy(phases)
    
    # 2. Non-local gradient (no per-parameter derivatives needed)
    grad = -w * E_cur * weights  # weights from gap irregularity
    
    # 3. Deterministic proposal
    proposal = phases - lr * grad
    
    # 4. PSON exploration (orthogonal to gradient, precision-scaled)
    noise = orthogonal_noise(grad, precision) * noise_scale
    candidate = proposal + noise
    
    # 5. Down-only acceptance
    if energy(candidate) <= E_cur:
        phases = candidate  # Accept exploration
    elif energy(proposal) <= E_cur:
        phases = proposal   # Fallback to deterministic
    # else: reject both, stay at current
```

**Key properties:**
- **Orthogonal:** `grad ⊙ noise ≈ 0` (doesn't fight descent)
- **Precision-scaled:** Lower-precision parameters explore more
- **Accepted-step monotonicity:** Under the down-only guard, accepted updates do not increase the measured energy
- **No local gradients:** Only needs global scalar feedback

See [paper](Sparse_Coherence_Recovery_via_PSON_V1.md) Section 4 for full details.

---

## Paper highlights

### Main result: 20/20 improved scenarios under fair conditions

| Signal | Coupling | Dependency | Baseline V | PSON V | Gain |
|--------|----------|------------|------------|--------|------|
| Zeta | Phase | Per-gap | 0.442 | 0.611 | **+0.169** |
| Turbulence | Phase | Per-gap | 0.432 | 0.639 | **+0.207** |
| ... | ... | ... | ... | ... | ... |

**Average gain:** +0.112 visibility  
**Evaluation budget:** 601 (equal for both methods)  
**Fair test protocol:** See [`docs/airtight/Fair_Test_Validation.md`](docs/airtight/Fair_Test_Validation.md)

### The Deterministic Descent Failure Mode

In **9/20 scenarios**, deterministic gradient descent achieved **0% acceptance rate** (stuck at initialization). This happens because:

1. Initial gradient points toward energy increase
2. Deterministic step rejected → system stays at same position
3. Next iteration: same position, same gradient, same rejected step
4. The deterministic run repeats the same rejected proposal

In these runs, PSON avoided this deterministic trap by regenerating orthogonal noise each iteration.

See [paper](Sparse_Coherence_Recovery_via_PSON_V1.md) Section 6.1.1 for detailed analysis.

### Where PSON helped in the recorded experiments

- **Static beamforming:** PSON-Subspace reported MSE 0.03-0.06 vs LMS's 25-150 in the matched-initialization test.
- **Moving target tracking:** PSON won 2/3 seeds against LMS variants in the recorded dynamic test.
- **Optical coherence:** PSON improved all 20 scenarios against deterministic non-local descent under the fair evaluation budget.
- **Massive MIMO (1024-2048 elements):** PSON-Subspace reported 25-66% lower MSE than LMS in that tested range.

### Known Limitations

- **Adaptive jammer nulling:** LMS wins 3/3 when the jammer moves.
- PSON's monotonic constraint prevents adaptation to moving adversaries
- See [`docs/SVD-Jammer-problem.md`](docs/SVD-Jammer-problem.md) for ongoing research
- **Clean continuous optimization:** CMA-ES achieves higher visibility than PSON in the recorded clean baseline comparison.
- **Domain-specific algorithms:** Gerchberg-Saxton and greedy methods win on some structured optical or RIS tasks.

---

## Reproducibility

### System Requirements
- **OS:** Windows 10+ (PowerShell)
- **Python:** 3.12+
- **Package manager:** [uv](https://github.com/astral-sh/uv)

---

## Quick Start

```powershell
# Setup (Windows PowerShell with uv)
uv sync

# Run main validation (20 scenarios, fair evaluation budgets)
uv run python .\experiments\airtight_experiments_001.py --fair_evals

# View results
type airtight_experiments_001_summary.json
```

**Expected result:** The recorded run improves all 20 scenarios against the deterministic baseline under equal computational budgets.

---

### Installation
```powershell
# Clone repository
git clone https://github.com/yourusername/PSON_Sparse_Coherence_Recovery
cd PSON_Sparse_Coherence_Recovery

# Install dependencies
uv sync
```

### Run Experiments

**Main validation (Section 6.1):**
```powershell
uv run python .\experiments\airtight_experiments_001.py --fair_evals
```

**Statistical significance (Section 6.4):**
```powershell
uv run python .\experiments\additional_experiments\multi_seed_validation_001.py
```

**Baseline comparison (Section 6.5):**
```powershell
uv run python .\experiments\additional_experiments\baseline_comparison_001.py
```

**Partial observability (Section 6.6):**
```powershell
uv run python .\experiments\additional_experiments\partial_observability_test_001.py
```

**Discrete phase optimization (Section 7.2):**
```powershell
# Phased arrays
uv run python .\experiments\discrete_applications\phased_array_antenna_test.py

# PSON vs LMS fair comparison
uv run python .\experiments\discrete_applications\pson_vs_lms_fair_comparison.py

# Dynamic scenarios (moving targets, massive MIMO)
uv run python .\experiments\discrete_applications\pson_dynamic_scenarios_test.py
```

**Sparse path integrals (Section 7.3):**
```powershell
uv run python .\experiments\sparse_path_integral_test.py --use_pson
```

**Gap distribution ablation (Section 6.9):**
```powershell
uv run python .\experiments\prime_log_random_hard_distributions\prime_test.py
```

**Speed benchmark (Section 6.5):**
```powershell
uv run python .\experiments\speed_benchmark.py
```

All experiments save results to `results/` as CSV/JSON + plots.

---

## Repository Structure

```
PSON_Sparse_Coherence_Recovery/
├── Sparse_Coherence_Recovery_via_PSON_V1.md  # Main paper
├── README.md                               # This file
├── LICENSE                                 # MIT License
├── pyproject.toml                          # Dependencies (uv)
│
├── experiments/                            # All runnable experiments
│   ├── airtight_experiments_001.py        # 20-scenario validation (CORE)
│   ├── homeostat_vector_test.py           # Basic PSON loop reference
│   ├── sparse_path_integral_test.py       # Path integral approximation
│   │
│   ├── additional_experiments/            # Baselines, statistics, degraded observations
│   │   ├── baseline_comparison_001.py     # vs CMA-ES/SA/Random
│   │   ├── multi_seed_validation_001.py   # Statistical significance
│   │   ├── partial_observability_test_001.py  # Noise/quantization
│   │   └── extreme_partial_observability_001.py
│   │
│   ├── discrete_applications/             # Phased arrays, beamforming, LiDAR
│   │   ├── phased_array_antenna_test.py
│   │   ├── pson_vs_lms_fair_comparison.py
│   │   ├── pson_dynamic_scenarios_test.py
│   │   ├── holographic_beam_steering_test.py
│   │   ├── acoustic_beamforming_test.py
│   │   └── ... (5 more)
│   │
│   ├── optical_scaling/                   # Large-array tests (100-4096)
│   ├── optical_momentum/                  # PSON+Momentum validation
│   ├── path_integral_approximator/        # Sparse path integral core
│   ├── prime_log_random_hard_distributions/  # Gap distribution ablation
│   └── PSON_ml_optimized/                 # ML-style problems
│
├── docs/                                   # Detailed experiment documentation
│   ├── airtight/                          # Fairness validation docs
│   │   ├── Fair_Test_Validation.md        # Complete fairness analysis
│   │   └── FAIRNESS_SUMMARY.md            # Quick reference
│   ├── SVD-Jammer-problem.md              # Open problem documentation
│   ├── homeostat_reference/               # Neuro-Symbolic Homeostat paper
│   └── ACTUAL_EXPERIMENT_*_README.md      # Per-experiment guides
│
└── results/                                # Experiment outputs (CSV/JSON/PNG)
    ├── airtight_experiments_001_*.csv/json/png
    ├── additional_experiments/
    ├── discrete_applications/
    └── ... (organized by experiment)
```

---

## Citation
work in progress...

**Authors:** Oscar Goldman @ Datamutant.ai, subsidiary of 温心重工業.


---

## Related Work

**Theoretical foundation:**
- Goldman (2025). *Complexity from Constraints: The Neuro-Symbolic Homeostat.* (PSON algorithm origin)

**Comparison baselines:**
- Shubber, Jamel & Nahar (2025). *Beamforming Array Antenna: New Innovative Research Using Partial Update Adaptive Algorithms.* AIP Conf. Proc. (PU-BAA)
- Hansen & Ostermeier (2001). *CMA-ES.* (Black-box optimization baseline)

See [paper](Sparse_Coherence_Recovery_via_PSON_V1.md) Section 10 for complete references.

---

## License

MIT License
Copyright (c) 2025 Oscar Goldman

---

## Acknowledgments

Special thanks to ML twitter for being awesome.
