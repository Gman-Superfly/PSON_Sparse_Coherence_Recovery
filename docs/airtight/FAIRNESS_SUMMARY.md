# Fairness Validation Summary

## What We Did

1. **Added fair evaluation budget mode** to `experiments/airtight_experiments_001.py`
   - Both PSON and baseline get exactly 601 function evaluations
   - Budget enforced before every `simulate_fn` call
   - Results tracked: `func_evals_no_pson`, `func_evals_pson`

2. **Made fallback structure symmetric**
   - Both methods try: (1) main candidate, (2) deterministic fallback
   - Baseline: candidate=proposal (so fallback is redundant but symmetric)
   - PSON: candidate=proposal+noise (so fallback provides genuine alternative)

3. **Documented the deterministic descent failure mode**
   - Added section 6.1.1 to paper explaining why baseline gets stuck
   - Identified 9/20 scenarios where baseline achieves 0% acceptance
   - Explained how PSON's repeated orthogonal exploration avoids this failure mode in the recorded runs

## Key Results

 **PSON wins 20/20 scenarios under fair conditions**
- Equal budget: 601 evaluations each
- Same initialization: zero phases
- Same scenarios: identical signals/couplings
- Same structure: candidate + fallback

 **Baseline failure mode documented**
- 9/20 scenarios: 0% acceptance (stuck at initialization)
- Deterministic gradient points toward energy increase
- No exploration mechanism → permanent trap
- PSON escapes in the recorded runs via orthogonal noise regeneration

## Files Modified

- `experiments/airtight_experiments_001.py`: Added fair evaluation mode
- `Sparse_Coherence_Recovery_via_PSON_V1.md`: Added section 6.1.1 explaining failure mode
- `docs/airtight/Fair_Test_Validation.md`: Complete fairness analysis document

## Run Fair Test

```powershell
uv run python .\experiments\airtight_experiments_001.py --fair_evals
```

## Conclusion

Under the stated protocol, the test uses equal evaluation budgets, shared initialization, and identical scenarios. The 20/20 result is valid for that protocol. The baseline getting stuck is not a bug; it documents a failure mode of this deterministic baseline under the zero-phase initialization.

