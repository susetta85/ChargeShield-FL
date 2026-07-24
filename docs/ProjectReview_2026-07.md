# ChargeShield-FL — Comprehensive Project Review
**Date:** July 2026  
**Reviewer role:** Senior Researcher (Distributed Systems / OT-ICS) + FL/Privacy Expert + DSN PC Reviewer perspective  
**Scope:** Security · Scalability · Test Coverage · Documentation · FedMIA Pipeline · FedProx / Split Correctness

> **Correction notice (2026-07-24).** Dated review snapshot, kept as historical
> record. Its "13,073 sessions" performance/scaling estimates (RAM footprint,
> forward-pass counts) predate the 2026-07-22 3-real-sites migration (`19d4c19`)
> — current combined raw total is ≈66,713 sessions across Caltech/JPL/Office1,
> roughly 5× larger, so the specific numbers in §"Performance" below are stale
> (the qualitative conclusion — not a bottleneck at this scale — likely still
> holds, but was not re-verified at the new scale as part of this notice).

---

## 1. Executive Summary

ChargeShield-FL is a well-structured FL privacy research framework. The core FedMIA pipeline in `run_experiments.py` is **scientifically valid**: the 80/20 train/holdout split is correct, membership scoring uses the global FL model (Yeom loss-based MIA), and FedProx is correctly implemented. However, **four issues threaten paper validity** and must be fixed before DSN 2027 submission:

1. **No feature normalization** — the Decoder uses `Sigmoid` (output in [0,1]) but raw input features like `minutes_available` reach 300+. MSE is dominated by large-valued features and the model trains suboptimally. CaseStudies.md claims normalization is applied; it is not.
2. **Formal DP claim does not hold** — weight perturbation with multi-epoch local training is not DP-SGD; the sensitivity bound of `max_grad_norm` is not sound for 3 epochs per round.
3. **Shadow model plugin (`fedmia.py`) is dead code** — never invoked by the experiment evaluator; tests for it exercise code that does not participate in any measurement.
4. **CaseStudies.md data partition is wrong** — describes a 3-way 50/25/25 split with a shadow public set; actual code uses 80/20, no shadow public set.

No critical security vulnerabilities found. Scalability is adequate for 4–16 nodes / 13k sessions.

---

## 2. FedMIA Pipeline Analysis

### 2.1 Train / Test Split — CORRECT ✅

`run_experiments.py` `main()` (lines ~547–564):

```python
random.seed(seed); random.shuffle(sessions)
split = max(1, int(len(sessions) * 0.8))
train_sessions   = sessions[:split]     # → FL training + MIA members
holdout_sessions = sessions[split:]     # → MIA non-members
fl_results  = run_fl_rounds(cfg, train_sessions)
mia_results = run_fedmia(cfg, train_sessions, holdout_sessions, fl_results)
```

The holdout is split **before** any FL training. `holdout_sessions` is never passed to `run_fl_rounds()`. This is correct.

### 2.2 Shadow Model — NOT USED IN THE EXPERIMENT EVALUATOR ⚠️

`run_fedmia()` does **not** use a shadow model. It loads `global_weights` directly from each FL round result into a fresh `Autoencoder` instance and scores sessions via `-MSE`. This is the Yeom et al. (2018) loss-based MIA — correct.

The shadow-model MIA lives in `src/plugins/attacks/fedmia.py` (`FedMIA` class) and is used only by `ChargingIDS.analyze_round()`. However, in `run_experiments.py::run_ids()`, `ChargingIDS` is instantiated **without** a `fedmia=` argument, so `self._fedmia is None` and the shadow-model path is silently disabled at runtime.

**Consequence:** `src/plugins/attacks/fedmia.py` is effectively dead code in the experiment pipeline. The Sprint 4 tests for `FedMIA` test a class that does not participate in any paper measurement.

### 2.3 Membership Scoring — CORRECT ✅

`run_fedmia()` iterates over `fl_results.items()`, loads `global_weights` from each round's `FedAvgAggregator.aggregate()` result, and builds a local `Autoencoder` instance. This is the post-DP, FedAvg-aggregated global model. Scoring is correct per the Yeom approach.

**Nuance to acknowledge in paper:** `members` = all `train_sessions` (80%), but each FL node only saw ~25% of them (one cluster's partition). Sessions in `train_sessions` that belonged to a different cluster's partition were never seen by the node whose model is being scored globally. The global model aggregates all four clusters, so the "member" signal is weaker than a single-node evaluation. This should be noted as a limitation.

### 2.4 AUC-ROC Interpretation

What is measured: *"Can the global FL model (after DP) reconstruct FL-training sessions better than holdout sessions?"* AUC-ROC = 0.5 → DP has erased the generalization gap. AUC-ROC > 0.5 → residual memorization. This is exactly the Yeom generalization-gap MIA. The sweep results (AUC ≈ 0.48–0.52 across all ε) are consistent with DP being effective at the configured noise levels.

### 2.5 FedProx — CORRECTLY IMPLEMENTED ✅

`config/experiment.yaml` sets `proximal_mu: 0.01`. `AutoencoderTrainer` reads it and applies the proximal term in `train_step()`:

```
loss += (mu/2) * ||w - w_global||²
```

Applied to trainable parameters only, after the first `apply_global_model()` call. Round 1 effectively runs without the proximal term (standard behavior; `_global_weights` is `None` on first call). This is correct per Li et al. (2020).

### 2.6 Round Sweep — CORRECTLY CONFIGURED ✅

The Makefile `experiment-full-sweep` target runs all combinations rounds ∈ {100, 200, 500, 1000} × ε ∈ {0.1, 0.5, 1.0, 2.0, 5.0}. AUC-ROC is computed **per round** for the full training trajectory. Output JSON stores `per_round[round]["auc_roc"]` for every round.

**Minor issue:** The sweep targets do not pass `--config` explicitly. If invoked from outside the project root, `load_config()` will raise `FileNotFoundError`. Fix: add `--config config/experiment.yaml` to both sweep targets in the Makefile.

### 2.7 Differential Privacy Claim — SIGNIFICANT ISSUE ❌

`GradientManager._compute_sigma()` computes:
```
sigma = max_grad_norm * sqrt(2 * ln(1.25/delta)) / epsilon
```

This applies **weight perturbation** (clip + noise the full weight vector), not DP-SGD (clip + noise per-sample gradients). The key difference: with 3 epochs of local training per round, the mapping from individual training samples to the final weight vector involves multiple gradient steps. The L2 sensitivity of the final weight vector to any single record is **not bounded by `max_grad_norm`** — that bound holds only for a single gradient step.

The formal (ε, δ)-DP guarantee for specific ε values does not hold. The empirical privacy measurement (AUC-ROC vs. noise level) remains valid and interesting, but the formal claim must be reframed.

Additionally, `sigma` is calibrated for a single round but used for T = 100–1000 rounds. By naive composition, total budget = T × ε. For T = 1000, ε = 0.1, the total privacy budget is ε_total = 100. This is not stated anywhere in the experiment reporting.

**Fix options:**
- **(a) Preferred for correctness:** Replace with DP-SGD via [Opacus](https://opacus.ai/) or [dp-accounting](https://github.com/google/differential-privacy/tree/main/python/dp_accounting).
- **(b) Fast path for paper:** Reframe all ε values as noise parameter (σ calibrated to per-round budget) without asserting formal (ε, δ)-DP. Report total budget in table footnotes. Acknowledge as a limitation.

---

## 3. Security Findings

| Severity | Location | Issue | Fix |
|---|---|---|---|
| LOW | `acn_dataset.py:105` | Path passed directly to `Path(path)` with no traversal check | `assert path.resolve().is_relative_to(datasets_root)` |
| LOW | `run_experiments.py:476` | `importlib.util.spec_from_file_location` + `exec_module()` loads external Python file — arbitrary code exec if `generate_excel_report.py` is writable | Use a fixed `import` or verify file hash |
| INFO | `config/experiment.yaml` | No hardcoded secrets — all paths relative | No action |
| INFO | `nvflare/workspace/` | mTLS certs generated here — verify `.gitignore` excludes this directory | Add to `.gitignore` if missing |
| INFO | `charging_ids.py:368` | `import logging as _logging` inside `@staticmethod` | Move to module top |
| INFO | Docker / Containerlab | `sudo containerlab` requires root; review `topology.clab.yml` for `privileged` containers | Audit container security context |

No hardcoded API keys, passwords, or private keys found. `yaml.safe_load()` is correctly used throughout.

---

## 4. Scalability Analysis

### 4.1 Node Count
The current design hardcodes `cluster_ids = ["highway", "urban", "residential", "corporate"]` in `run_fl_rounds()`. Scaling to more clusters requires only extending this list. `FedAvgAggregator._weighted_average()` is O(n_nodes × n_parameters) — linear, no bottleneck.

### 4.2 Memory
All 13,073 sessions loaded into RAM as a Python list (~5–10 MB). No streaming needed at this scale. For millions of sessions, a generator-based loader would be required.

### 4.3 FedMIA Scoring Complexity
For 1000 rounds × 13,073 sessions ÷ 256 per batch ≈ 51,000 forward passes total. CPU time: ~1–5 minutes per 1000-round run. Not a bottleneck.

### 4.4 GradientManager Sigma
Computed once at `__init__` — correct; sigma depends only on constants, no per-round overhead.

### 4.5 DataLoader `drop_last=True`
For `batch_size=32` with ~3,268 sessions per cluster, at most 31 sessions silently dropped per cluster per epoch. Minor labeling imprecision (those sessions are labeled "members" but were never trained on). Not a practical concern at this dataset size, but worth a comment.

---

## 5. Test Coverage Assessment

| Component | Tested? | Correct API? | Issues |
|---|---|---|---|
| `Autoencoder` (forward, reconstruction, is_anomaly, fit) | ✅ `test_sprint4.py` | ✅ | Correct |
| `AutoencoderTrainer` (train_local, FedProx proximal term) | ❌ | — | No test that proximal term is active |
| `FedAvgAggregator` (collect, aggregate, weighted average) | ❌ | — | Critical gap — correctness untested |
| `GradientManager` (clipping, noise, sigma formula) | ❌ | — | DP correctness untested |
| `FedMIA` plugin `src/plugins/attacks/fedmia.py` | ✅ `test_sprint4.py` | ✅ | Tests correct class but class is dead code in experiment |
| `run_fedmia()` (loss-based MIA, AUC-ROC) | ❌ | — | Core experiment function has ZERO test coverage |
| `enrich_sessions()` (hour_of_day, duration_hours) | ❌ | — | 6-feature set depends on this |
| 80/20 split in `main()` | ❌ | — | Untested |
| `ChargingIDS`, `CUSUMDetector`, `KrumDetector`, `GradientAnalyzer` | ✅ | ✅ | Good coverage |
| `PrivacyAuditor` | ❌ | — | Epsilon accumulation, threat thresholds untested |
| AUC-ROC > 0.5 without DP / ≈ 0.5 with DP | ❌ | — | No integration test validates the core finding |
| `test_sprint5.py` | ❌ (does not exist) | — | Sprint 5 deliverables have no dedicated test file |
| Feature dimension (6 everywhere) | ✅ code | ⚠️ docstring | `test_sprint4.py` line 47 says "7 feature" — stale docstring |

---

## 6. Documentation Consistency

| Document | Status | Discrepancies |
|---|---|---|
| `docs/Architecture.md` | ⚠️ Mostly consistent | `enrich_sessions()` described as `ACNDataset` method — it is a standalone function in `run_experiments.py`. `epochs=5` stated as default; config and code use `epochs=3`. `PluginRegistry` described but not found in source. |
| `docs/ThreatModel.md` | ✅ Consistent | Correctly distinguishes shadow-model plugin from experiment evaluator. One minor error: FL Clients placed at "Purdue L3"; Architecture.md and code place them at L2. |
| `docs/MLPlane.md` | ✅ Consistent | Observer pattern, `emit_event()` / `subscribe()` interface matches implementation. |
| `docs/CaseStudies.md` | ❌ **Inconsistent** | Section 2.2.3 describes a **3-way 50/25/25 split** with shadow model public set. Code uses **80/20**, no shadow public set. Section 2.2.2 claims features are "standardised to zero mean and unit variance" — normalization is **not implemented** in code. |
| Feature count | ✅ code | Stale `test_sprint4.py` docstring says "7 feature" |
| `config/experiment.yaml` | ✅ | Comment correctly documents that shadow-model parameters are unused |

---

## 7. Code Quality Issues

| Severity | Location | Issue |
|---|---|---|
| 🔴 HIGH | `autoencoder.py` Decoder, `autoencoder_trainer.py` | **No feature normalization.** Decoder output is `Sigmoid` (range [0,1]) but raw features (`minutes_available` 0–300+, `total_energy_kwh` 0–80) are far outside [0,1]. MSE dominated by large-valued features; Sigmoid will saturate. Autoencoder trains suboptimally. |
| 🔴 HIGH | `run_experiments.py:564` | `train_sessions` passed as MIA members, but `drop_last=True` in DataLoader means up to 31 sessions per cluster per epoch were never actually trained on but are labeled "members". Minor in practice but technically incorrect. |
| 🟡 MEDIUM | `gradient_manager.py:165` | `delta > 1e-2` triggers a warning but is not rejected. `delta=0.5` would pass validation and produce a formally valid but practically meaningless DP claim. |
| 🟡 MEDIUM | `run_experiments.py:130` | Cluster assignment by list position (temporal split), not behavioral or geographic features. The "non-IID" claim is actually a temporal split — defensibility requires acknowledgment in paper. |
| 🟡 MEDIUM | `fedmia.py:107` | Commented-out dead code: `#self._shadow_model = Autoencoder()...` — remove. |
| 🟢 LOW | `fedavg_aggregator.py:85` | Comment claims race condition protection; code is single-threaded. Misleading. |
| 🟢 LOW | `autoencoder.py` vs `autoencoder_trainer.py` | `Autoencoder.get_weights()` returns `dict` (state_dict); `AutoencoderTrainer.get_weights()` returns `list[Tensor]`. Same method name, different return types — potential confusion. |
| 🟢 LOW | `charging_ids.py:368` | `import logging as _logging` inside `@staticmethod` — move to module top. |
| 🟢 LOW | `privacy_auditor.py:190` | `round_epsilon = sensitivity / self._max_grad_norm` is a heuristic, but the field is named `epsilon` in `AuditReport` — may mislead readers. |

---

## 8. Priority Fix List

Ranked by impact on **paper validity and scientific credibility**:

| # | Priority | File(s) | Issue | Action |
|---|---|---|---|---|
| 1 | 🔴 CRITICAL | `autoencoder.py`, `autoencoder_trainer.py`, `run_experiments.py` | No feature normalization — Sigmoid decoder with unnormalized inputs | Add min-max scaling computed from `train_sessions`; apply to both `train_sessions` and `holdout_sessions` before FL and MIA scoring |
| 2 | 🔴 CRITICAL | `docs/CaseStudies.md` | Wrong data partition description (50/25/25 with shadow public) | Update to reflect actual 80/20 split and loss-based MIA (no shadow public) |
| 3 | 🔴 CRITICAL | `gradient_manager.py` | Formal DP claim not sound for multi-epoch weight perturbation | Either adopt Opacus DP-SGD, or reframe ε as noise parameter with explicit composition budget in results tables |
| 4 | 🟡 HIGH | `tests/` | `run_fedmia()` has zero test coverage | Add `test_sprint5.py` covering: correct return shape, member_score > non_member_score without DP, AUC-ROC in [0,1] |
| 5 | 🟡 HIGH | `tests/` | `FedAvgAggregator`, `AutoencoderTrainer`, `GradientManager` untested | Add unit tests for weighted average correctness, proximal term activation, sigma formula |
| 6 | 🟡 MEDIUM | `Makefile` | `experiment-sweep` / `experiment-full-sweep` missing `--config` | Add `--config config/experiment.yaml` to both targets |
| 7 | 🟡 MEDIUM | `docs/Architecture.md` | `enrich_sessions()` attribution wrong, `epochs=5` vs. 3, missing `PluginRegistry` clarification | Correct all three |
| 8 | 🟡 MEDIUM | `docs/Architecture.md` | Two FedMIA paths not clearly distinguished | Add section "Two FedMIA Paths" explaining (a) `fedmia.py` is IDS-only and currently disabled, (b) `run_fedmia()` is the experiment evaluator |
| 9 | 🟡 MEDIUM | `src/plugins/attacks/fedmia.py`, `run_experiments.py::run_ids()` | Shadow-model FedMIA plugin is dead code | Either activate it (pass `FedMIA` instance to `ChargingIDS`) or explicitly document and remove dead commented line at line 107 |
| 10 | 🟡 MEDIUM | `docs/CaseStudies.md`, `run_experiments.py` | Cluster assignment is temporal split, not non-IID behavioral split | Acknowledge in paper, or implement feature-distribution-aware assignment |
| 11 | 🟢 LOW | `run_experiments.py::run_experiments.py:14` | Comment says `run_experiment.py` (singular) | Trivial rename in comment |
| 12 | 🟢 LOW | `tests/test_sprint4.py:47` | Docstring says "7 feature" | Update to "6 feature" |

---

## 9. What Can Be "Stolen" Across Rounds

Given the current loss-based MIA setup, the following analysis applies:

| Rounds | Expected AUC-ROC (DP active) | What attacker learns | Interpretation |
|---|---|---|---|
| 100 | ~0.50 | Negligible — model has not converged, noise dominates | DP effective even at early convergence |
| 200 | ~0.50 | Still negligible | DP effective |
| 500 | ~0.50 | Still negligible | DP effective |
| 1000 | ~0.50 | Still negligible | DP effective |

**Current result: DP is effective at all tested ε values for all round counts.** The AUC-ROC uniformity at ≈0.48–0.52 is consistent with the noise level (σ calibrated to ε=1.0, δ=1e-5) being sufficient to prevent generalization-gap exploitation.

**However:** due to the missing normalization, the MSE values are likely dominated by `minutes_available` and `total_energy_kwh` magnitudes rather than learned structure. This means the "null result" (AUC ≈ 0.5) may be an artifact of the model not actually learning meaningful representations rather than DP being effective. **This must be verified after adding normalization.**

For the paper's CS3 evaluation to be credible, experiments must be re-run after fix #1 (normalization). The expected outcome: without DP, AUC-ROC should be measurably > 0.5 (the model can distinguish members from non-members). With DP at small ε, AUC-ROC should converge toward 0.5. If AUC ≈ 0.5 even without DP after normalization, the loss-based MIA signal is too weak for this dataset and the attack should be strengthened (e.g., using a threshold-based MIA tuned per cluster, or a more discriminative feature representation).

---

*Review completed: 16 source files read, 12 issues identified, 4 critical for paper validity.*
