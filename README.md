# Sparse Coherence Recovery via PSON

**Empirical Validation of Precision-Scaled Orthogonal Exploration on Irregular Optical Arrays**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Authors:** Oscar Goldman, Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業  
**Date:** November 2025  
**Status:** Working code with reproducible experiments


## What This Repository Contains

### Contribution
Validation that **PSON (Precision-Scaled Orthogonal Noise)** improves optical coherence recovery on **irregular sparse arrays** where deterministic gradient descent fails:

- **20/20 win rate** across 5 signal types × 2 coupling modes × 2 dependency types
- **Fair comparison:** Equal evaluation budgets (601 simulate_fn calls each)
- **Statistically significant:** 95% CI [+0.103, +0.185] excludes zero (n=50 runs)
- **Identifies failure mode:** Deterministic baseline stuck (0% acceptance) in 9/20 scenarios

### Results

| Application | PSON Performance | Key Finding |
|-------------|------------------|-------------|
| **Optical coherence** | 20/20 wins (+0.03 to +0.16 visibility) | Core validation |
| **Static beamforming** | PSON-Subspace wins 3/3 (MSE: 0.05 vs 127) | Massive advantage |
| **Moving target tracking** | PSON wins 2/3 (67%) | Better than LMS |
| **Adaptive jammer nulling** | LMS wins 3/3  | **PSON limitation identified** |
| **Massive MIMO (1024-2048)** | PSON-Subspace wins 2/3 | Scale advantage |

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

## The PSON Algorithm

**TL;DR:** Safe exploration via orthogonal noise + monotonic descent guards.

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
    
    # 5. Down-only acceptance (monotonic descent guarantee)
    if energy(candidate) <= E_cur:
        phases = candidate  # Accept exploration
    elif energy(proposal) <= E_cur:
        phases = proposal   # Fallback to deterministic
    # else: reject both, stay at current
```

**Key properties:**
- **Orthogonal:** `grad ⊙ noise ≈ 0` (doesn't fight descent)
- **Precision-scaled:** Uncertain parameters explore more
- **Monotonic:** Energy never increases
- **No local gradients:** Only needs global scalar feedback

See [paper](Sparse_Coherence_Recovery_via_PSON_V1.md) Section 4 for full details.

---

## Paper Highlights

### Main Result: 20/20 Win Rate Under Fair Conditions

| Signal | Coupling | Dependency | Baseline V | PSON V | Gain |
|--------|----------|------------|------------|--------|------|
| Zeta | Phase | Per-gap | 0.442 | 0.611 | **+0.169** |
| Turbulence | Phase | Per-gap | 0.432 | 0.639 | **+0.207** |
| ... | ... | ... | ... | ... | ... |

**Average gain:** +0.112 visibility  
**Evaluation budget:** 601 (equal for both methods)  
**Fair test validated:** See [`docs/airtight/Fair_Test_Validation.md`](docs/airtight/Fair_Test_Validation.md)

### The Deterministic Descent Failure Mode

In **9/20 scenarios**, deterministic gradient descent achieved **0% acceptance rate** (stuck at initialization). This happens because:

1. Initial gradient points toward energy increase
2. Deterministic step rejected → system stays at same position
3. Next iteration: same position, same gradient, same rejected step
4. **Permanent trap** with no escape mechanism

**PSON solves this** by regenerating orthogonal noise each iteration, providing continuous exploration even when the deterministic gradient is trapped.

See [paper](Sparse_Coherence_Recovery_via_PSON_V1.md) Section 6.1.1 for detailed analysis.

### Where PSON Excels

 **Static beamforming:** PSON-Subspace achieves MSE 0.03-0.06 vs LMS's 25-150  
 **Moving target tracking:** 67% win rate vs LMS  
 **Optical coherence:** 100% win rate vs deterministic descent  
 **Massive MIMO (1024-2048 elements):** 25-66% better MSE than LMS

### Known Limitations

 **Adaptive jammer nulling:** LMS wins 3/3 when jammer moves  
- PSON's monotonic constraint prevents adaptation to moving adversaries
- See [`docs/SVD-Jammer-problem.md`](docs/SVD-Jammer-problem.md) for ongoing research
 **Some Smooth Landscapes, algo is built for noisy landscapes**

*Other limitation but also many wins, study main paper for full report*

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

**Expected result:** PSON wins **20/20 scenarios** vs deterministic baseline under equal computational budgets.

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
├── Sparse_Coherence_Recovery_via_PSON.md  # Main paper (1594 lines)
├── README.md                               # This file
├── LICENSE                                 # MIT License
├── pyproject.toml                          # Dependencies (uv)
│
├── experiments/                            # All runnable experiments
│   ├── airtight_experiments_001.py        # 20-scenario validation (CORE)
│   ├── homeostat_vector_test.py           # Basic PSON loop reference
│   ├── sparse_path_integral_test.py       # Path integral approximation
│   │
│   ├── additional_experiments/            # Baselines, statistics, robustness
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

If you use this repository in your research, please cite:

```bibtex
@software{goldman2025sparse_coherence,
  title        = {Sparse Coherence Recovery via PSON: Empirical Validation on Irregular Optical Arrays},
  author       = {Goldman, Oscar},
  organization = {Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業},
  year         = {2025},
  url          = {https://github.com/yourusername/PSON_Sparse_Coherence_Recovery},
  note         = {Fair evaluation validation with equal computational budgets}
}

@software{goldman2025homeostat,
  title        = {Complexity from Constraints: The Neuro-Symbolic Homeostat},
  author       = {Goldman, Oscar},
  organization = {Shogu Research Group @ Datamutant.ai subsidiary of 温心重工業},
  year         = {2025},
  note         = {PSON theoretical framework and stability projectors}
}
```

---

## Related Work

**Theoretical foundation:**
- Goldman (2025). *Complexity from Constraints: The Neuro-Symbolic Homeostat.* (PSON algorithm origin)

**Comparison baselines:**
- Shubber, Jamel & Nahar (2025). *Beamforming Array Antenna: New Innovative Research Using Partial Update Adaptive Algorithms.* AIP Conf. Proc. (PU-BAA)
- Hansen & Ostermeier (2001). *CMA-ES.* (Black-box optimization baseline)

See [paper](Sparse_Coherence_Recovery_via_PSON_V1.md) Section 10 for complete references.

---

## Contributing

This is research code released for reproducibility. For questions or contributions:

1. **Issues:** Report bugs or unclear documentation
2. **Pull requests:** Improvements to experiments or documentation welcome
3. **Research collaboration:** Contact via repository issues

---

## License

MIT License - see [LICENSE](LICENSE) file.

Copyright (c) 2025 Oscar Goldman

---

## Acknowledgments

**Shogu Research Group @ Datamutant.ai** subsidiary of 温心重工業

Special thanks for:
- Fair test validation methodology
- Deterministic descent failure mode identification
- Open problem documentation (SVD-Jammer)

