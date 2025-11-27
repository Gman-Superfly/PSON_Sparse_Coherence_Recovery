# The SVD-Jammer Problem: PSON-Subspace Fails on Moving Adversaries

**Status:** Open Problem  
**Date:** November 2025  
**Author:** Oscar Goldman, Shogu Research Group @ Datamutant.ai

---

## Problem Statement

PSON-Subspace achieves excellent performance on **static** beamforming (3/3 wins on MSE), but fails completely on **moving jammer** scenarios (0/3 wins). This document analyzes the root cause and explores potential solutions.

---

## Background: How PSON-Subspace Works

### The Subspace Projection

1. **Compute SVD** of received signal matrix X at initialization:
   ```python
   U, S, Vh = np.linalg.svd(X[:100], full_matrices=False)
   V_k = Vh[:k].T  # Top k singular vectors
   ```

2. **Project weights** to k-dimensional subspace:
   ```python
   w_sub = V_k.T.conj() @ w_full  # (k,) instead of (N,)
   ```

3. **Optimize in subspace** (much fewer dimensions):
   ```python
   # Gradient and PSON noise in k dimensions
   candidate_sub = w_sub + lr * grad_sub + pson_noise
   ```

4. **Lift back** to full space:
   ```python
   w_full = V_k @ w_sub  # (N,)
   ```

### Why It Works on Static

The signal subspace is determined by:
- Target direction: 1 rank
- Interferer directions: K ranks
- Total effective rank: 1 + K << N

If these don't change, the subspace V_k computed at t=0 remains valid for all t.

---

## The Failure Mode: Stale Subspace

### What Happens When Jammer Moves

```
Time t=0:                          Time t=100 (jammer moved):
                                   
Jammer at -30°                     Jammer at +45°
     ↓                                  ↓
   ╔═══╗                              ╔═══╗
   ║ V ║ ← SVD captures this          ║ V ║ ← Still using old V!
   ╚═══╝   subspace                   ╚═══╝   
     ↓                                  ↓
Subspace contains                  Subspace does NOT contain
jammer direction                   new jammer direction
     ↓                                  ↓
Can null jammer ✅                 Cannot null jammer ❌
```

### Empirical Evidence (Fair Comparison Test)

| Algorithm | Static MSE | Moving Jammer SINR |
|-----------|------------|-------------------|
| **PSON-Sub** | **0.03** (best) | **-5 dB** (jammer wins) |
| PSON | 0.55 | -3 dB |
| Full-LMS | 97 | **+21 dB** (signal wins) |

PSON-Sub's MSE advantage completely vanishes when the scene changes.

---

## Root Cause Analysis

### Problem 1: One-Shot SVD

The SVD is computed **once** at initialization:
```python
# This line runs ONCE at t=0
U, S, Vh = np.linalg.svd(X[:100], full_matrices=False)
V_k = Vh[:k].T  # Fixed for all iterations!
```

When jammer moves at t=50, V_k is stale but we keep using it.

### Problem 2: Subspace Doesn't Include New Jammer

If jammer was at -30° and moves to +45°:
- Old V_k spans directions around -30°
- New jammer at +45° is **orthogonal** to V_k
- Projecting to V_k removes the jammer component
- We can't form a null in a direction we can't represent

### Problem 3: Monotonic Descent Trap

Even standard PSON fails on jammer because:
```python
# PSON's acceptance rule:
if mse_new <= mse_cur:
    accept()
else:
    reject()  # Problem!
```

When jammer moves:
1. MSE jumps up (environment changed, not algorithm's fault)
2. Adaptive step would reduce MSE relative to new jammer position
3. But MSE is still higher than before jammer moved
4. PSON rejects the step as "worse"
5. Stuck with weights optimized for old jammer position

---

## Why LMS Wins on Moving Jammer

LMS has no monotonic constraint:
```python
# LMS update (always accepts):
w = w + mu * gradient
```

- Gradient points toward nulling **current** jammer (not past)
- No rejection of "worse" steps
- Naturally tracks moving interference

**Tradeoff:** LMS can diverge or amplify interference if learning rate is wrong. PSON's monotonic constraint prevents this—but also prevents adaptation.

---

## Potential Solutions (To Investigate)

### Approach 1: Adaptive Subspace Update

**Idea:** Periodically recompute SVD as scene changes.

```python
if iteration % subspace_update_interval == 0:
    # Recompute subspace from recent data
    U, S, Vh = np.linalg.svd(X_recent, full_matrices=False)
    V_k = Vh[:k].T
    
    # Re-project current weights
    w_sub = V_k.T.conj() @ w_full
```

**Challenge:** How often? Too frequent = expensive. Too rare = stale.

**Potential:** Use energy/MSE spike as trigger for recomputation.

### Approach 2: Expanding Subspace

**Idea:** When MSE spikes, add new directions to subspace.

```python
if mse_spike_detected:
    # Find direction of new interference
    residual = X - X @ V_k @ V_k.T.conj()  # What we're missing
    new_direction = dominant_eigenvector(residual)
    
    # Expand subspace
    V_k = np.hstack([V_k, new_direction])
    k = k + 1
```

**Challenge:** Subspace grows unbounded. Need pruning strategy.

### Approach 3: Relaxed Monotonicity

**Idea:** Allow temporary MSE increase when environment change detected.

```python
# Detect environment change (MSE jump without weight change)
if mse_cur > 1.5 * mse_prev and weights_unchanged:
    environment_changed = True
    best_mse = mse_cur  # Reset baseline

# Relaxed acceptance
if environment_changed:
    if mse_new < mse_cur:  # Just needs to improve from new baseline
        accept()
```

**Challenge:** How to distinguish "environment changed" from "bad step"?

### Approach 4: Hybrid PSON-LMS

**Idea:** Use PSON for exploration, LMS for tracking.

```python
if stable_environment:
    # PSON mode: safe exploration with monotonic descent
    use_pson_update()
else:
    # LMS mode: fast tracking without monotonic constraint
    use_lms_update()
```

**Challenge:** Mode switching logic. Wrong mode at wrong time = worst of both.

### Approach 5: Full-Space PSON with Adaptive Noise

**Idea:** Don't use subspace. Instead, adapt noise magnitude to uncertainty.

```python
# Track how "stale" our knowledge is
uncertainty = time_since_last_improvement

# Scale PSON noise by uncertainty
noise_scale = base_noise * (1 + uncertainty_factor * uncertainty)

# Larger exploration when stuck
pson_noise = noise_scale * orthogonal_noise
```

**Challenge:** Still discrete phases. May need continuous relaxation.

---

## Experiments To Run

### Experiment 1: Subspace Update Frequency

Vary `subspace_update_interval` from 1 to 100:
- Measure: MSE, SINR, computation time
- Goal: Find sweet spot between freshness and efficiency

### Experiment 2: MSE-Triggered Update

Implement spike detection:
```python
if mse_cur > alpha * moving_average_mse:
    recompute_subspace()
```
Vary alpha from 1.2 to 2.0.

### Experiment 3: Monotonicity Relaxation

Compare:
- Strict monotonic (current)
- Reset-on-spike (Approach 3)
- Probabilistic acceptance (simulated annealing style)

### Experiment 4: Hybrid System

Implement PSON-LMS switching:
- Measure: SINR over time with jammer trajectory
- Compare: Pure PSON, Pure LMS, Hybrid

---

## Success Criteria

A solution should achieve:

| Metric | Target | Current PSON-Sub |
|--------|--------|------------------|
| Static MSE | < 0.1 | ✅ 0.03 |
| Moving target tracking | < 5° error | ✅ ~75° (bad but workable) |
| Moving jammer SINR | > 0 dB | ❌ -5 dB (fails) |

**Goal:** Maintain PSON-Sub's static performance while achieving positive SINR on moving jammer.

---

## References

- Fair comparison test: `experiments/discrete_applications/pson_vs_lms_fair_comparison.py`
- Results: `results/fair_comparison/pson_vs_lms_fair_comparison_results.json`
- Paper section: `Sparse_Coherence_Recovery_via_PSON.md`, Appendix E.9

---

## Next Steps

1. [ ] Implement Experiment 1 (subspace update frequency)
2. [ ] Implement Experiment 2 (MSE-triggered update)
3. [ ] Implement Experiment 3 (monotonicity relaxation)
4. [ ] Analyze which approach best balances static/dynamic performance
5. [ ] Update paper with findings

---

*This is an open research problem. Solutions welcome.*

