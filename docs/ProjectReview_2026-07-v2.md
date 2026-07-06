# ChargeShield-FL — Project Re-Review (Post-Fix, July 2026)
**Date:** 2026-07-06  
**Baseline:** `docs/ProjectReview_2026-07.md`  
**Scope:** Verification of all 12 previous fixes + new issues found

---

## 1. Executive Summary

The documentation fixes from the previous review are complete and correct. However, the implementation of feature normalization (the highest-priority code fix) had a **critical P0 bug**: the two split assignment lines (`train_sessions = sessions[:split]` / `holdout_sessions = sessions[split:]`) were missing, causing a `NameError` crash on any experiment run. This has been corrected in this review cycle.

The new test file `test_sprint5.py` adds good coverage for `AutoencoderTrainer`, `GradientManager`, and `FedAvgAggregator` — a significant improvement. However, the core experiment functions (`run_fedmia`, `enrich_sessions`, `normalize_sessions`, `compute_feature_stats`) still have zero test coverage.

**Before the paper can be submitted:** the normalization fix must be validated by re-running at least one full experiment and confirming that (a) no crash occurs, and (b) without DP the AUC-ROC is measurably > 0.5 (confirming the MIA signal actually exists on normalized data).

---

## 2. Fix Verification Table

| # | Priority | Issue | Status |
|---|---|---|---|
| 1 | CRITICAL | No feature normalization in code | **FIXED** — `compute_feature_stats()` (line 118) and `normalize_sessions()` (line 137) implemented. Split assignment lines now restored (this review). Min-max scaling applied before FL training and MIA scoring. |
| 2 | CRITICAL | Wrong data partition in CaseStudies.md (50/25/25) | **FIXED** — §2.2.3 now shows 80/20 table, no shadow public set. |
| 3 | CRITICAL | Formal DP claim not sound (weight perturbation, not DP-SGD) | **PARTIAL** — Code comment acknowledges the limitation. CaseStudies.md §2.4.3 still presents the Gaussian Mechanism as providing full (ε,δ)-DP without qualification. Paper text update pending. |
| 4 | HIGH | `run_fedmia()` has zero test coverage | **STILL OPEN** — No tests for `run_fedmia()`, `enrich_sessions()`, `normalize_sessions()`, `compute_feature_stats()`. |
| 5 | HIGH | `FedAvgAggregator`, `AutoencoderTrainer`, `GradientManager` untested | **FIXED** — `test_sprint5.py` adds comprehensive coverage for all three. |
| 6 | MEDIUM | Makefile sweep targets missing `--config` | **STILL OPEN** |
| 7 | MEDIUM | `Architecture.md` — `enrich_sessions()` attribution, `epochs=5` vs. 3 | **STILL OPEN** |
| 8 | MEDIUM | `Architecture.md` — Two FedMIA paths not distinguished | **STILL OPEN** |
| 9 | MEDIUM | Shadow-model FedMIA dead code | **PARTIAL** — Comment added in `run_ids()`. Duplicate instantiation at `fedmia.py` lines 107–108 is a new active-but-overwritten bug (see §4.1). |
| 10 | MEDIUM | Cluster assignment is temporal split, not non-IID behavioral | **STILL OPEN** |
| 11 | LOW | Comment says `run_experiment.py` (singular) | **STILL OPEN** |
| 12 | LOW | `test_sprint4.py:46` docstring says "7 feature" | **STILL OPEN** |

---

## 3. Critical Issues Remaining

### 3.1 CaseStudies.md §2.4.3 — DP Overclaim
Line 188 still reads: *"This formulation is the standard Gaussian Mechanism for (epsilon, delta)-DP [Dwork et al., 2014]. Gradient clipping to `max_grad_norm` before noise addition ensures that the sensitivity of the mechanism is bounded."*

This is incorrect for multi-epoch local training. The sensitivity of the full weight vector to a single training sample is not bounded by `max_grad_norm` when `epochs > 1`. Add the following note immediately after the sigma formula block:

> **Limitation.** ChargeShield-FL applies the Gaussian Mechanism to the aggregated weight vector after local training (weight perturbation), not to per-sample gradients during training (DP-SGD). With `epochs=3` local rounds per FL round, the formal (ε,δ)-DP guarantee for specific ε values requires the per-sample sensitivity to be bounded by `max_grad_norm`, which holds only for `epochs=1`. In the current configuration, ε values should be interpreted as noise parameters (higher ε = less noise) rather than formal DP budgets. Additionally, running T FL rounds compounds the per-round budget: total ε ≈ T × ε_per_round under naive composition.

### 3.2 No Integration Test for Normalization
`normalize_sessions()` and `compute_feature_stats()` are new functions with no tests. A regression in either would silently produce values outside [0,1] and corrupt the MSE computation without raising an error.

### 3.3 Re-run Required to Validate Normalization Effect
The sweep results in `experiments/` were produced without normalization. Those AUC-ROC values (≈0.48–0.52) may be artifacts of the Sigmoid saturation on unnormalized inputs. At minimum, one run of 100 rounds at ε=1.0 and one run with no DP (ε=100 or similar) must be executed after this fix to confirm:
- No crash
- Without DP: AUC-ROC measurably > 0.5 (MIA signal exists)
- With DP at ε=1.0: AUC-ROC approaches 0.5 (DP effective)

---

## 4. New Issues Found

### 4.1 `fedmia.py` — Active Duplicate Instantiation (lines 107–108)
```python
self._shadow_model = Autoencoder().to(self._device)           # line 107 — overwritten immediately
self._shadow_model = Autoencoder(input_dim=input_dim).to(self._device)  # line 108 — the real one
```
Line 107 creates an Autoencoder with the default `input_dim=6` and immediately discards it. Remove line 107.

### 4.2 `test_sprint5.py` — Wrong Schema in `test_train_local_none_feature_skipped`
The test fixture uses a session with keys `voltage_v`, `current_a`, `power_kw`, `energy_kwh`, `temperature_c`, `soc_percent`, `timestamp` — none of which appear in `AutoencoderTrainer.CONTINUOUS_FEATURES`. The test passes but does not verify the `None`-skip logic on the actual ACN schema. Replace one key with a real feature (e.g. `total_energy_kwh: None`) to make the test meaningful.

### 4.3 `test_sprint5.py` — `test_aggregate_weighted_average` Misses BatchNorm Buffers
Constructs weights from `model.parameters()` instead of `model.state_dict()`. Does not exercise the int64 dtype-restoration path for `num_batches_tracked` buffers. Not a crash, but leaves a code path untested.

### 4.4 `test_sprint5.py` — `min_participants` Check Applied Before `valid` Filter
`FedAvgAggregator.aggregate()` checks `len(updates) < min_participants` before filtering zero-sample updates. This means 3 collected updates with 2 invalid ones passes the `min_participants=2` check with only 1 valid participant. The design may be intentional but is undocumented and the test at lines 318–332 does not surface this behavior explicitly.

### 4.5 `Architecture.md` — `epochs` Mismatch
§6.3 states `epochs=5` as default; `experiment.yaml` and the code default to `epochs=3`. Minor but affects reproducibility claims.

---

## 5. Documentation Consistency — Current State

| Document | Status | Remaining Gaps |
|---|---|---|
| `docs/CaseStudies.md` | 🟡 Mostly correct | §2.4.3 DP overclaim; §2.3 says "default: 5 epochs" |
| `docs/Architecture.md` | 🔴 Several gaps | `enrich_sessions()` attribution; `epochs=5`; Two FedMIA paths undistinguished; PluginRegistry |
| `docs/ThreatModel.md` | 🟢 Consistent | No issues found |
| `docs/MLPlane.md` | 🟢 Consistent | No issues found |
| `config/experiment.yaml` | 🟢 Consistent | |
| `tests/test_sprint4.py` | 🟡 Minor | Line 46 docstring: "7 feature" |
| `scripts/run_experiments.py` | 🟢 Now correct | Split lines restored; normalization implemented |

---

## 6. Recommended Next Actions

| # | Priority | Action | File |
|---|---|---|---|
| 1 | P0 ✅ DONE | Insert `train_sessions`/`holdout_sessions` split lines | `run_experiments.py` lines 602–603 |
| 2 | P1 | Re-run experiment (100 rounds, ε=1.0 and no-DP) to validate normalization | CLI |
| 3 | P1 | Add tests for `run_fedmia()`, `normalize_sessions()`, `compute_feature_stats()`, `enrich_sessions()` | `tests/test_sprint5.py` or `test_sprint6.py` |
| 4 | P2 | Add DP limitation note to `CaseStudies.md` §2.4.3 | `docs/CaseStudies.md` |
| 5 | P2 | Add `--config config/experiment.yaml` to Makefile sweep targets | `Makefile` lines 113, 128 |
| 6 | P3 | Remove redundant line 107 in `fedmia.py` | `src/plugins/attacks/fedmia.py` |
| 7 | P3 | Fix `Architecture.md`: `enrich_sessions()` attribution, `epochs=3`, Two FedMIA paths | `docs/Architecture.md` |
| 8 | P4 | Fix stale docstrings: `test_sprint4.py:46`, `run_experiments.py:14`, `CaseStudies.md` §2.3 | various |
| 9 | P4 | Fix `test_sprint5.py` wrong-schema fixture; add BatchNorm buffer test | `tests/test_sprint5.py` |

---

*P0 fix applied in this review cycle. All remaining items are P1–P4 and do not block pipeline execution.*
