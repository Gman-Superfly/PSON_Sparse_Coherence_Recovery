# Fair Evaluation Budget Verification

## Evaluation Counting Logic

PSON vs multiple algos airtight test

### Initial State (Line 248-253)
```python
I0 = simulate_fn(phases)          # +1 eval
func_evals = 1
```
**Running total: 1**

### Fixed-Step Mode (eval_budget=None, Lines 257-299)
Each iteration:
1. `I_cur = simulate_fn(phases)` → +1 eval (line 258, 259)
2. `I_new = simulate_fn(candidate)` → +1 eval (line 274, 275)
3. If PSON and rejected:
   - `I_det = simulate_fn(proposal)` → +1 eval (line 287, 288)

**Per iteration cost:**
- Baseline (no PSON): 2 evals/iter
- PSON (best case, accept candidate): 2 evals/iter  
- PSON (worst case, try fallback): 3 evals/iter

**After 200 steps:**
- Baseline: 1 + 200×2 = **401 evals**
- PSON: 1 + 200×(2 to 3) = **401-601 evals** (depends on acceptance)

**❌ PROBLEM: Asymmetric evaluation usage!**

---

## Budget Mode (eval_budget set, Lines 300-362)

Budget set at line 473: `eval_budget = (1 + steps * 3) = 1 + 200×3 = 601`

Each iteration:
1. Check budget for I_cur (line 307-308)
2. `I_cur = simulate_fn(phases)` → +1 eval (line 309-310)
3. Check budget for candidate (line 326-327)
4. `I_new = simulate_fn(candidate)` → +1 eval (line 329-330)
5. If rejected and PSON:
   - Check budget for fallback (line 343)
   - `I_det = simulate_fn(proposal)` → +1 eval (line 345-346)
6. Break when `func_evals + 1 > eval_budget`

**Termination:**
- Both methods run until they hit the **same budget (601)**
- Budget is checked BEFORE each eval → never exceeds limit
- Both return final `func_evals` count

**✅ FAIR: Both methods consume same evaluation budget**

---

## Result Validation

From experiment output:
```json
"mean_func_evals_no_pson": 601.0,
"mean_func_evals_pson": 601.0
```

**✅ VERIFIED: Both methods used exactly 601 evaluations**

---

## Remaining Concern: Iterations vs Evaluations

### Question: Do both methods get the same number of *optimization steps*?

**Answer: NO, and that's intentional for fairness.**

- Baseline uses 2 evals/iter → gets ~300 iterations from 601 budget
- PSON uses 2-3 evals/iter → gets ~200-300 iterations from 601 budget

This is **fair** because:
1. Real-world constraint is compute budget (function evaluations), not iterations
2. PSON's extra fallback eval is part of its design
3. Both algorithms get the same resources to find the best solution

### Alternative "Fairness" Definition (Equal Iterations)

If we wanted equal iterations instead:
- Give baseline 200 iterations → 401 evals
- Give PSON 200 iterations → 401-601 evals (acceptance-dependent)

This would be **unfair** because PSON gets more compute resources.

---

## Conclusion

✅ **The implementation is correct for evaluation-budget fairness.**

Both algorithms:
- Start from the same initial phases (zeros)
- Use the same RNG seed
- Run on the same signal/coupling/dependency
- Consume exactly the same function evaluation budget (601)
- The only difference is the algorithm itself

PSON wins 20/20 scenarios under these fair conditions.

