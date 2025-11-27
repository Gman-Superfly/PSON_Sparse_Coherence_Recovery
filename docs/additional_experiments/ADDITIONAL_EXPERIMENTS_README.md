# Additional Experiments for Publication

This document describes the additional experiments for strengthening the paper for different publication venues.

**Note:** All experiment scripts are located in `experiments/additional_experiments/`.

---

## 1. Baseline Comparison (for IEEE Signal Processing Letters)

**Script:** `experiments/additional_experiments/baseline_comparison_001.py`

**Purpose:** Compare PSON against standard black-box optimization methods to demonstrate that the gains are not trivially achievable.

### Baselines Included
| Method | Description |
|--------|-------------|
| **PSON** (ours) | Non-local credit + precision-scaled orthogonal noise |
| **Random Search** | Uniform random sampling in phase space |
| **Simulated Annealing** | Metropolis-Hastings with exponential cooling |
| **Finite-Diff GD** | Vanilla gradient descent with numerical gradients |
| **CMA-ES** | Covariance Matrix Adaptation (if `cma` package installed) |

### Usage
```powershell
# Default run (5 seeds)
python .\experiments\baseline_comparison_001.py

# With more seeds
python .\experiments\baseline_comparison_001.py --seeds "42,123,456,789,1000,2024,3141,4242"

# Skip CMA-ES
python .\experiments\baseline_comparison_001.py --no_cma
```

### Artifacts
- `baseline_comparison_001_results.csv` — Per-run results
- `baseline_comparison_001_summary.json` — Summary statistics
- `baseline_comparison_001_bar.png` — Bar chart comparison

### Expected Outcome
PSON should outperform Random Search and Simulated Annealing significantly, and compete favorably with CMA-ES while using fewer function evaluations.

---

## 2. Multi-Seed Validation (for NeurIPS/ICML)

**Script:** `experiments/additional_experiments/multi_seed_validation_001.py`

**Purpose:** Run PSON ablation across multiple random seeds to establish statistical significance with confidence intervals.

### Configuration
- **Default seeds:** 10 seeds (42, 123, 456, 789, 1000, 2024, 3141, 4242, 5555, 6789)
- **Signals:** All 5 (zeta, sinmix, one_over_f, chirp, turbulence)
- **Total runs:** 10 seeds × 5 signals × 2 (PSON/no-PSON) = 100 runs

### Usage
```powershell
# Default run (10 seeds × 5 signals)
python .\experiments\multi_seed_validation_001.py

# Fewer seeds (faster)
python .\experiments\multi_seed_validation_001.py --seeds "42,123,456"

# More seeds (more statistical power)
python .\experiments\multi_seed_validation_001.py --seeds "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"

# Single signal focus
python .\experiments\multi_seed_validation_001.py --signals "turbulence"
```

### Artifacts
- `multi_seed_validation_001_results.csv` — Per-run results (seed × signal)
- `multi_seed_validation_001_summary.json` — Summary with confidence intervals
- `multi_seed_validation_001_box.png` — Box plot of gains per signal
- `multi_seed_validation_001_gain_dist.png` — Histogram of all gains

### Key Metrics Computed
| Metric | Description |
|--------|-------------|
| Mean Gain | Average PSON improvement over baseline |
| Std Gain | Standard deviation (robustness indicator) |
| 95% CI | Confidence interval for mean gain |
| Win Rate | Percentage of runs where PSON > baseline |

### Expected Outcome
- **Win rate:** Should be >95% (ideally 100%)
- **95% CI:** Should exclude zero (statistically significant)
- **Low variance:** Gains should be consistent across seeds

---

## 3. Partial Observability Test

**Script:** `experiments/additional_experiments/partial_observability_test_001.py`

**Purpose:** Validate PSON's robustness under degraded observation conditions compared to CMA-ES.

### Degradation Conditions Tested
| Condition | Description |
|-----------|-------------|
| **Clean** | Perfect observations (baseline) |
| **Noise-Low/Med/High** | Additive Gaussian noise (σ=0.01, 0.05, 0.1) |
| **Quant-10/5/3** | Quantized feedback (10, 5, or 3 discrete levels) |
| **Stale-2/5/10** | Delayed feedback (observe every N steps) |
| **Combined-Mild** | Noise + quantization + staleness (mild) |
| **Combined-Severe** | Noise + quantization + staleness (severe) |

### Usage
```powershell
# Default run
uv run python .\experiments\additional_experiments\partial_observability_test_001.py

# With more seeds
uv run python .\experiments\additional_experiments\partial_observability_test_001.py --seeds "42,123,456,789,1000"
```

### Artifacts
- `partial_observability_001_results.csv` — Per-run results
- `partial_observability_001_summary.json` — Summary statistics
- `partial_observability_001_degradation.png` — Comparison plot

### Key Finding
The performance gap between CMA-ES and PSON closes from **-0.41 to -0.04** as observability degrades, validating PSON's robustness.

---

## 4. Extreme Partial Observability Test

**Script:** `experiments/additional_experiments/extreme_partial_observability_001.py`

**Purpose:** Test scenarios where PSON's design advantages are most relevant.

### Extreme Scenarios Tested
| Scenario | Description |
|----------|-------------|
| **Binary** | Pass/fail feedback only (no magnitude information) |
| **Adversarial-10/20/30%** | Feedback flipped with probability 10-30% |
| **Delay-5/10** | Feedback delayed by 5-10 steps |
| **Binary+Adversarial** | Binary feedback + 15% adversarial flip |
| **Nightmare** | Binary + 20% adversarial + delay (worst case) |

### Usage
```powershell
# Default run (budget=150)
uv run python .\experiments\additional_experiments\extreme_partial_observability_001.py

# Limited budget (where PSON's efficiency matters more)
uv run python .\experiments\additional_experiments\extreme_partial_observability_001.py --budget 50

# Very limited budget
uv run python .\experiments\additional_experiments\extreme_partial_observability_001.py --budget 40 --seeds "42,123,456,789,1000,2024,3141,4242,5555,6789"
```

### Artifacts
- `extreme_partial_obs_001_results.csv` — Per-run results
- `extreme_partial_obs_001_summary.json` — Summary statistics
- `extreme_partial_obs_001_comparison.png` — 4-panel comparison plot

### Key Findings
| Scenario | Gap (Clean) | Gap (Extreme) | Interpretation |
|----------|-------------|---------------|----------------|
| Clean | -0.10 | — | CMA-ES dominates |
| Binary+Adversarial | — | **-0.03** | Nearly tied |
| Nightmare | — | **-0.04** | Gap closed 60% |

- CMA-ES remains superior in most scenarios (it's a state-of-the-art optimizer)
- Under extreme degradation, **PSON achieves up to 40% win rate**
- PSON's value is **robustness and guarantees**, not raw performance

---

## Quick Reference

### For baseline comparison
```powershell
# Run baseline comparison
uv run python .\experiments\additional_experiments\baseline_comparison_001.py --seeds "42,123,456,789,1000"

# Key result to report: PSON vs CMA-ES and Random Search
```

### For Run all experiments
```powershell
# Run all experiments
uv run python .\experiments\additional_experiments\multi_seed_validation_001.py
uv run python .\experiments\additional_experiments\baseline_comparison_001.py
uv run python .\experiments\additional_experiments\partial_observability_test_001.py
uv run python .\experiments\additional_experiments\extreme_partial_observability_001.py --budget 40

# Key results: 95% CI, win rate, robustness under degradation
```

### For Demonstrating PSON's Robustness Advantage
```powershell
# Show gap closing under degraded observability
uv run python .\experiments\additional_experiments\partial_observability_test_001.py

# Show near-parity under extreme conditions
uv run python .\experiments\additional_experiments\extreme_partial_observability_001.py --budget 40
```

---

## Adding Results to Paper

After running, add a new section to the paper:

### For Baseline Comparison (Section 6.X)
```markdown
### 6.X Comparison with Standard Optimizers

| Method | Mean V | Std | Func Evals |
|--------|--------|-----|------------|
| PSON (ours) | 0.XX | 0.XX | XXX |
| CMA-ES | 0.XX | 0.XX | XXX |
| Simulated Annealing | 0.XX | 0.XX | XXX |
| Random Search | 0.XX | 0.XX | XXX |
```

### For Multi-Seed (add to existing results)
```markdown
**Statistical Validation (n=50 runs):**
- Mean PSON gain: +0.XXX ± 0.XXX
- 95% CI: [0.XXX, 0.XXX]
- Win rate: XX.X%
```

### For Partial Observability (Section 6.6)
```markdown
### 6.6 Partial Observability Validation

| Condition | PSON V | CMA-ES V | Gap |
|-----------|--------|----------|-----|
| Clean | 0.537 | 0.946 | -0.409 |
| Combined-Severe | 0.399 | 0.435 | -0.036 |

**Key Finding:** Gap closes from -0.41 to -0.04 as observability degrades.
```

### For Extreme Scenarios (Section 6.7)
```markdown
### 6.7 Extreme Degradation

| Scenario | PSON V | CMA-ES V | Gap | Win Rate |
|----------|--------|----------|-----|----------|
| Binary+Adversarial | 0.404 | 0.432 | -0.028 | 40% |

**Conclusion:** PSON approaches parity under extreme conditions.
```

---

## Dependencies

Core (required):
- numpy
- matplotlib

Optional:
- `cma` — For CMA-ES baseline (`pip install cma`)
- `mpmath` — For accurate ζ function (`pip install mpmath`)

---

## Citation

If you use these experiments, please cite the main paper:

```bibtex
@software{goldman2025sparse_coherence,
  title  = {Sparse Coherence Recovery via PSON},
  author = {Goldman, Oscar},
  year   = {2025}
}
```

