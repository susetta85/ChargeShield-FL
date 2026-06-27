# Testing Documentation — ChargeShield-FL

**Project:** ChargeShield-FL: Membership Inference Attack Evaluation Framework for Federated Learning in EV Charging Infrastructure
**Target venue:** DSN 2027
**Document version:** 1.0
**Date:** 2026-06-26

---

## Table of Contents

1. [Testing Philosophy](#1-testing-philosophy)
2. [Sprint 4 Test Suite — 52 Tests](#2-sprint-4-test-suite--52-tests)
   - 2.1 [TestAutoencoder (11 tests)](#21-testautoencoder-11-tests)
   - 2.2 [TestCUSUMDetector (7 tests)](#22-testcusumdetector-7-tests)
   - 2.3 [TestGradientAnalyzer (8 tests)](#23-testgradientanalyzer-8-tests)
   - 2.4 [TestKrumDetector (6 tests)](#24-testkrumdetector-6-tests)
   - 2.5 [TestFedMIA (7 tests)](#25-testfedmia-7-tests)
   - 2.6 [TestChargingIDS (11 tests)](#26-testchargingids-11-tests)
3. [Sprint 5 Test Suite — 25 Tests](#3-sprint-5-test-suite--25-tests)
   - 3.1 [TestAutoencoderTrainer (12 tests)](#31-testautoencodertrainer-12-tests)
   - 3.2 [TestGradientManager (7 tests)](#32-testgradientmanager-7-tests)
   - 3.3 [TestFedAvgAggregator (6 tests)](#33-testfedavgaggregator-6-tests)
4. [Sprint 5/6 Fix Documentation](#4-sprint-56-fix-documentation)
5. [Running the Test Suites](#5-running-the-test-suites)
6. [Coverage Report](#6-coverage-report)
7. [Integration Test Strategy](#7-integration-test-strategy)
8. [References](#8-references)

---

## 1. Testing Philosophy

### 1.1 Overview

ChargeShield-FL adopts a sprint-aligned, unit-first testing strategy grounded in the principle that scientific reproducibility is a first-class software requirement. Because the framework is intended to support peer-reviewed claims at DSN 2027, every algorithmic component must be independently verifiable by an external reviewer who obtains the repository and executes the test suite without additional configuration. This document describes the testing architecture, provides a complete catalogue of all test cases across Sprint 4 and Sprint 5, documents known API fixes applied between sprints, and outlines the integration test work still required before submission.

### 1.2 Sprint-Aligned Unit Test Structure

Each sprint introduces or stabilises a cohesive layer of the system and is accompanied by a dedicated test module:

| Sprint | Test file | Tests | Primary concerns |
|--------|-----------|-------|-----------------|
| 4 | `tests/test_sprint4.py` | 52 | Autoencoder model, anomaly detection algorithms (CUSUM, Krum, Cosine), FedMIA attack engine, ChargingIDS orchestrator |
| 5 | `tests/test_sprint5.py` | 25 | NVFLARE-integrated trainer, differential privacy gradient manager, FedAvg aggregator |

Tests are written with `pytest` and `unittest.TestCase`. All random seeds are fixed where stochastic behaviour is exercised, ensuring deterministic outcomes across repeated runs and across machines with equivalent Python and PyTorch versions. No test requires network access, a running NVFLARE server, or real EV charging data; all inputs are synthetically generated within the test body or in `setUp` fixtures.

### 1.3 Reproducibility as a First-Class Requirement

Reproducibility in the context of DSN 2027 means two distinct things:

1. **Software reproducibility.** Any collaborator, reviewer, or artefact evaluator can clone the repository, install the pinned dependencies from `requirements.txt`, and run `make test` to obtain a green suite. No environment-specific secrets, paths, or network connections are required.

2. **Scientific reproducibility.** The numerical outputs of the framework — AUC-ROC of FedMIA, Krum scores, CUSUM alarm thresholds, differential privacy noise calibration — must be derivable from documented hyperparameters via auditable formulas. Unit tests serve as executable specifications of these formulas. For example, `test_sigma_computed` in `TestGradientManager` directly verifies the Gaussian mechanism formula σ = C × sqrt(2 ln(1.25/δ)) / ε against an analytically computed expected value, ensuring that the implementation matches the theoretical privacy guarantee stated in the paper.

### 1.4 What Is NOT Tested at the Unit Level

The following concerns are explicitly deferred to the integration test strategy described in Section 7, or are outside the scope of automated testing entirely:

- **Real NVFLARE orchestration.** Tests that involve `AutoencoderTrainer` and `FedAvgAggregator` mock the NVFLARE `FLContext` and `Shareable` objects. A full multi-process NVFLARE federation — with the server, multiple clients, and the communication layer — is not instantiated in any unit test. Such a test would require a live NVFLARE 2.7.2 deployment, which cannot be guaranteed in a CI environment.

- **Real Containerlab network emulation.** The IDS components (`ChargingIDS`, `CUSUMDetector`, `KrumDetector`) are tested against synthetic gradient and metric streams. The actual deployment scenario — where each EV charging station runs as a Containerlab node and gradients traverse emulated network links with configurable latency and packet loss — is not exercised. Containerlab topology tests require root privileges and dedicated hardware.

- **End-to-end AUC-ROC measurement.** The `FedMIA` unit tests verify that the attack pipeline produces output of the correct type and that membership scores lie in [0, 1]. They do not verify that the AUC-ROC exceeds any threshold under any particular privacy budget, because that claim depends on the trained federation's generalisation gap and must be evaluated in a full experimental sweep.

- **Differential privacy formal proofs.** Unit tests verify the noise magnitude formula and that gradients are modified after privatisation. They do not formally prove (ε, δ)-differential privacy of the overall protocol, which is addressed in the theoretical sections of the paper.

- **Dataset licence compliance.** The ACN-Data dataset (Caltech EV charging logs) is not included in the repository. Tests that involve `ACNDataset` use a minimal in-memory mock that exposes the same API surface. Compliance with ACN-Data terms of use is the responsibility of the researcher running full experiments.

---

## 2. Sprint 4 Test Suite — 52 Tests

**File:** `tests/test_sprint4.py`

Sprint 4 implements and validates the core machine learning and anomaly detection components in isolation from the federated learning infrastructure. The autoencoder is the anomaly detection backbone; CUSUM, Krum, and cosine similarity are the intrusion detection algorithms; FedMIA is the threat model; and ChargingIDS is the orchestrator that integrates all of the above.

### 2.1 TestAutoencoder (11 tests)

**Module under test:** The `Autoencoder` class implements a symmetric encoder-decoder architecture in PyTorch. The encoder maps from an input of dimension `INPUT_DIM = 6` through hidden layers of width 16 and 8 to a bottleneck of dimension 4. The decoder mirrors this structure back to `INPUT_DIM`. The model is trained with mean squared error (MSE) loss.

**Historical note on INPUT_DIM.** An architectural constant `INPUT_DIM = 6` is defined in the model module and is the intended operational dimension. However, several Sprint 4 tests were written with `INPUT_DIM = 7` as a historical artefact from an earlier data schema where a seventh feature (raw timestamp epoch) was retained before the final feature engineering pass reduced the schema to six features. These tests pass synthetic tensors of width 7 to the model's forward pass. This is documented here because:

(a) The discrepancy does not cause test failures — the model's `nn.Linear` input layer is parameterised by `INPUT_DIM` at construction time, so tests that override the constant or that construct a separate model instance with `input_dim=7` exercise a valid code path but one that does not correspond to the operational configuration.

(b) Any researcher extending the suite should use `INPUT_DIM = 6` for new tests.

(c) Section 4 records the reconciliation fix applied in Sprint 5/6.

| # | Test name | Class / method tested | Description |
|---|-----------|----------------------|-------------|
| 1 | `test_encoder_output_shape` | `Autoencoder.encoder` forward pass | Passes a batch of synthetic tensors through the encoder only and asserts that the output shape is `(batch_size, 4)`, confirming the bottleneck dimension. |
| 2 | `test_decoder_output_shape` | `Autoencoder.decoder` forward pass | Feeds a random bottleneck tensor of shape `(batch_size, 4)` through the decoder and asserts that the output shape is `(batch_size, INPUT_DIM)`, confirming the reconstruction dimension. |
| 3 | `test_autoencoder_forward_shape` | `Autoencoder.forward` | Passes a full batch through the complete autoencoder (encoder + decoder) and asserts that the output tensor has the same shape as the input, i.e., `(batch_size, INPUT_DIM)`. |
| 4 | `test_reconstruction_error_is_float` | `Autoencoder.reconstruction_error` | Calls `reconstruction_error` with a single sample and asserts that the return value is a Python `float` (not a tensor), confirming the public API contract. |
| 5 | `test_reconstruction_error_non_negative` | `Autoencoder.reconstruction_error` | Asserts that `reconstruction_error` returns a value >= 0 for an arbitrary input, which must hold because the error is computed as mean squared error and MSE is non-negative by definition. |
| 6 | `test_reconstruction_error_2d_input` | `Autoencoder.reconstruction_error` | Verifies that `reconstruction_error` accepts a 2-D input tensor (one sample expressed as a row vector) without raising an exception and returns a non-negative float. |
| 7 | `test_trained_model_low_error_on_normal_data` | `Autoencoder.fit` + `reconstruction_error` | Trains the autoencoder for a small number of epochs on a synthetic Gaussian dataset representing normal charging sessions, then asserts that the mean reconstruction error on held-out normal samples falls below a tolerance threshold, confirming that the model learns a useful representation. |
| 8 | `test_anomaly_detection_on_outlier` | `Autoencoder.reconstruction_error` (post-training) | After training on normal data (as above), feeds a deliberately out-of-distribution outlier tensor and asserts that its reconstruction error exceeds that of a normal sample by a meaningful margin, validating the anomaly scoring principle. |
| 9 | `test_get_weights_returns_dict` | `Autoencoder.get_weights` | Calls `get_weights` and asserts that the return value is a dictionary whose keys match the model's `state_dict` keys and whose values are numeric arrays or tensors, confirming the serialisation interface used by the federated learning layer. |
| 10 | `test_set_weights_loads_correctly` | `Autoencoder.set_weights` | Constructs two `Autoencoder` instances, extracts weights from the first, sets them on the second, and asserts that all parameter tensors in the second instance are numerically equal to those in the first, confirming round-trip consistency. |
| 11 | `test_fit_returns_loss_list` | `Autoencoder.fit` | Calls `fit` for a small number of epochs and asserts that the return value is a non-empty list of floats, each representing the mean training loss for one epoch. Also asserts that loss values are non-negative and that the list length equals the number of epochs requested. |
| 12 | `test_fit_calibrates_threshold` | `Autoencoder.fit` + `threshold` attribute | After calling `fit`, asserts that the model exposes a `threshold` attribute (set to a percentile of training reconstruction errors) that is a positive float, confirming that anomaly scoring is calibrated at the end of training without a separate calibration call. |

### 2.2 TestCUSUMDetector (7 tests)

**Module under test:** `CUSUMDetector` implements the Cumulative Sum (CUSUM) sequential change-point detection algorithm [Page 1954]. For each monitored node, two statistics S+ and S- are maintained. S+ accumulates upward deviations from a reference mean (shifted by an allowance parameter k), and S- accumulates downward deviations. An alarm is raised when either statistic exceeds a threshold h. A warmup period suppresses alarms during initial population of the statistics to avoid false positives from transient start-up behaviour.

| # | Test name | Method tested | Description |
|---|-----------|--------------|-------------|
| 1 | `test_no_alarm_during_warmup` | `CUSUMDetector.update` | Feeds a sharply drifting signal for fewer steps than the configured warmup period and asserts that `update` returns no alarm, confirming that the warmup guard is respected. |
| 2 | `test_alarm_on_positive_drift` | `CUSUMDetector.update` | Injects a sustained positive drift (mean significantly above the reference) for a number of steps sufficient to exhaust the warmup period and accumulate S+ beyond h, then asserts that an alarm is reported. |
| 3 | `test_alarm_on_negative_drift` | `CUSUMDetector.update` | Injects a sustained negative drift and asserts that the S- statistic triggers an alarm, confirming that the detector is bidirectional. |
| 4 | `test_no_alarm_stable_signal` | `CUSUMDetector.update` | Feeds a zero-mean Gaussian signal (with a fixed seed) over many steps and asserts that no alarm is raised, confirming that the threshold h is set high enough to avoid false positives on stationary noise. |
| 5 | `test_independent_nodes` | `CUSUMDetector.update` (multi-node) | Configures the detector to monitor two nodes, injects drift only into one, and asserts that only the drifting node raises an alarm while the stable node does not, confirming that per-node statistics are maintained independently. |
| 6 | `test_reset_clears_state` | `CUSUMDetector.reset` | Accumulates drift sufficient to trigger an alarm, calls `reset`, then resumes feeding stable data, and asserts that no alarm is raised after the reset and that internal statistics S+ and S- are zeroed. |
| 7 | `test_get_cusum_values` | `CUSUMDetector.get_cusum_values` | After feeding several data points, calls `get_cusum_values` and asserts that the return value is a dictionary mapping node identifiers to pairs (S+, S-) of non-negative floats, confirming the introspection API used by `ChargingIDS` to include CUSUM state in round analysis reports. |

### 2.3 TestGradientAnalyzer (8 tests)

**Module under test:** `GradientAnalyzer` provides utility functions for operating on federated gradient updates represented as dictionaries of layer-name-to-tensor mappings. It supports flattening, L2 norm computation, pairwise cosine similarity, and a cluster-level analysis that identifies potentially poisoned nodes by comparing each node's gradients against the centroid of the remaining nodes.

| # | Test name | Method tested | Description |
|---|-----------|--------------|-------------|
| 1 | `test_flatten_extracts_floats` | `GradientAnalyzer.flatten` | Passes a gradient dictionary containing numeric tensors and asserts that `flatten` returns a 1-D list or array of floating-point values, one per gradient component, in a deterministic order. |
| 2 | `test_flatten_ignores_non_numeric` | `GradientAnalyzer.flatten` | Passes a gradient dictionary that contains a mix of numeric tensors and non-numeric metadata entries and asserts that `flatten` returns only the numeric components without raising an exception, confirming defensive handling of heterogeneous update dictionaries. |
| 3 | `test_l2_norm_pythagorean` | `GradientAnalyzer.l2_norm` | Constructs a gradient vector with known values (e.g., [3, 4]) and asserts that `l2_norm` returns a value within floating-point tolerance of the analytically computed norm (5.0), verifying the Euclidean norm implementation. |
| 4 | `test_cosine_similarity_identical` | `GradientAnalyzer.cosine_similarity` | Passes two identical gradient vectors and asserts that the cosine similarity equals 1.0 (within floating-point tolerance), since a vector is perfectly aligned with itself. |
| 5 | `test_cosine_similarity_orthogonal` | `GradientAnalyzer.cosine_similarity` | Passes two orthogonal gradient vectors (dot product = 0) and asserts that the cosine similarity equals 0.0 (within floating-point tolerance). |
| 6 | `test_cosine_similarity_opposite` | `GradientAnalyzer.cosine_similarity` | Passes two anti-parallel gradient vectors (one is the negation of the other) and asserts that the cosine similarity equals -1.0 (within floating-point tolerance), confirming the full [-1, 1] range is handled correctly. |
| 7 | `test_cluster_cosine_analysis_returns_all_nodes` | `GradientAnalyzer.cluster_cosine_analysis` | Provides gradient updates for N nodes and asserts that the returned dictionary contains an entry for each node identifier, ensuring that no node is silently dropped from the analysis. |
| 8 | `test_poisoned_node_lower_similarity` | `GradientAnalyzer.cluster_cosine_analysis` | Constructs a scenario in which one node's gradients are deliberately inverted (sign-flip poisoning) while all other nodes share a common direction. Asserts that the poisoned node's cosine similarity score is lower than that of all honest nodes, validating the detection signal used by `ChargingIDS`. |

### 2.4 TestKrumDetector (6 tests)

**Module under test:** `KrumDetector` implements the Krum Byzantine-robust aggregation criterion [Blanchard et al. 2017]. For a set of n gradient update vectors, each with an assumed upper bound f on the number of Byzantine participants, the Krum score of update i is the sum of its squared Euclidean distances to the (n - f - 2) nearest neighbours. Updates with high Krum scores are geometrically isolated from the majority and are therefore candidates for Byzantine behaviour.

| # | Test name | Method tested | Description |
|---|-----------|--------------|-------------|
| 1 | `test_scores_for_all_nodes` | `KrumDetector.compute_scores` | Provides gradient updates for N nodes and asserts that `compute_scores` returns a score for each node (dictionary with N entries), confirming completeness of coverage. |
| 2 | `test_scores_in_range` | `KrumDetector.compute_scores` | Asserts that all returned Krum scores are non-negative finite floats, since they are defined as sums of squared distances. |
| 3 | `test_byzantine_node_highest_score` | `KrumDetector.compute_scores` | Constructs a scenario with one outlier node whose gradient update is far from the cluster of honest updates. Asserts that the outlier node receives the highest Krum score among all nodes. |
| 4 | `test_detect_byzantine_returns_list` | `KrumDetector.detect_byzantine` | Calls `detect_byzantine` and asserts that the return value is a list (possibly empty) of node identifiers classified as Byzantine, confirming the output type contract. |
| 5 | `test_detect_byzantine_identifies_poisoned` | `KrumDetector.detect_byzantine` | Uses the same outlier scenario as test 3 and asserts that the outlier node's identifier appears in the list returned by `detect_byzantine`, confirming end-to-end detection correctness. |
| 6 | `test_insufficient_nodes_returns_zero_scores` | `KrumDetector.compute_scores` | Calls `compute_scores` with fewer nodes than the minimum required for meaningful Krum computation (i.e., fewer than f + 2 + 1 nodes) and asserts that the method either returns a dictionary of zero scores or raises a clearly documented exception, confirming graceful degradation in small federations. |

### 2.5 TestFedMIA (7 tests)

**Module under test:** `FedMIA` implements the shadow-model membership inference attack adapted for federated learning [Shokri et al. 2017; Nasr et al. 2019]. The attacker trains a shadow autoencoder on a dataset that mimics the target federation's training distribution. For each target sample, the membership score is the reconstruction error of the target (global) model on that sample, normalised against the shadow model's error distribution. The attack outputs an `MIAResult` object containing the AUC-ROC of the binary membership classifier.

| # | Test name | Method tested | Description |
|---|-----------|--------------|-------------|
| 1 | `test_run_attack_without_training_raises` | `FedMIA.run_attack` (precondition) | Calls `run_attack` on a freshly instantiated `FedMIA` object that has not had a target model set, and asserts that a `RuntimeError` (or equivalent documented exception) is raised, confirming that the precondition is enforced. |
| 2 | `test_run_attack_returns_mia_result` | `FedMIA.run_attack` | Provides a trained target autoencoder and synthetic member/non-member datasets, calls `run_attack`, and asserts that the return value is an instance of `MIAResult`. |
| 3 | `test_mia_result_fields` | `MIAResult` dataclass | Asserts that the `MIAResult` object returned by `run_attack` exposes the fields `auc_roc`, `membership_scores`, `labels`, and `threshold`, and that each has the expected type (float, list/array, list/array, float respectively). |
| 4 | `test_membership_score_in_range` | `FedMIA.run_attack` -> `membership_scores` | Asserts that every membership score in `MIAResult.membership_scores` lies in the closed interval [0, 1], which is required for AUC-ROC computation to be well-defined. |
| 5 | `test_run_cluster_attack_returns_list` | `FedMIA.run_cluster_attack` | Calls `run_cluster_attack` with a multi-node federation scenario and asserts that the return value is a list, one entry per cluster (or per node, depending on the clustering configuration). |
| 6 | `test_run_cluster_attack_all_nodes_covered` | `FedMIA.run_cluster_attack` | Asserts that the list returned by `run_cluster_attack` contains an entry for every node identifier provided, ensuring that no node is omitted from the cluster-level analysis. |
| 7 | `test_cluster_result_has_deviation_metadata` | `FedMIA.run_cluster_attack` -> per-cluster result | Asserts that each entry in the cluster attack result exposes a `deviation` field (a float quantifying how much the node's local model diverges from the global model in terms of membership leakage), which is used to rank nodes by privacy risk in the paper's experimental evaluation. |

### 2.6 TestChargingIDS (11 tests)

**Module under test:** `ChargingIDS` is the top-level intrusion detection orchestrator. It receives per-round federated learning reports (containing gradient norms, privacy budget consumption, and per-node update statistics), dispatches to the underlying detectors (CUSUM, Krum, cosine similarity), maintains a running risk score with exponential decay, and accumulates an alert history.

| # | Test name | Method tested | Description |
|---|-----------|--------------|-------------|
| 1 | `test_analyze_no_alert_normal_report` | `ChargingIDS.analyze` | Feeds a synthetic report that is entirely within normal operating ranges (gradient norms below explosion threshold, budget not exhausted, no CUSUM drift) and asserts that no alert is generated. |
| 2 | `test_analyze_gradient_explosion_generates_alert` | `ChargingIDS.analyze` | Feeds a report in which one node's gradient L2 norm exceeds the configured explosion threshold and asserts that an alert of type `GRADIENT_EXPLOSION` is generated, with the offending node identified in the alert metadata. |
| 3 | `test_analyze_budget_exhausted_generates_alert` | `ChargingIDS.analyze` | Feeds a report indicating that the differential privacy budget (epsilon consumed / epsilon total) has reached 1.0 and asserts that a `BUDGET_EXHAUSTED` alert is generated, prompting the operator to cease training for the current epoch. |
| 4 | `test_analyze_cusum_detects_drift` | `ChargingIDS.analyze` (CUSUM integration) | Simulates a sequence of reports in which gradient norms for one node drift upward monotonically. Asserts that after sufficient rounds the `ChargingIDS` emits a `CUSUM_DRIFT` alert, confirming that the CUSUM sub-detector's output is forwarded correctly. |
| 5 | `test_analyze_round_returns_round_analysis` | `ChargingIDS.analyze_round` | Calls `analyze_round` with a synthetic round descriptor and asserts that the return value is a `RoundAnalysis` object (or equivalent named structure) containing at minimum the round index, a list of alerts, and the current risk score. |
| 6 | `test_analyze_round_detects_byzantine` | `ChargingIDS.analyze_round` | Provides a round descriptor in which one node's gradients are geometrically isolated (high Krum score, low cosine similarity). Asserts that the `RoundAnalysis` object flags that node as a potential Byzantine participant. |
| 7 | `test_analyze_round_has_krum_scores` | `ChargingIDS.analyze_round` | Asserts that the `RoundAnalysis` object includes a `krum_scores` field containing a score for every node in the round, confirming that Krum scores are always computed and surfaced regardless of whether any Byzantine detection threshold is exceeded. |
| 8 | `test_analyze_round_has_cosine_scores` | `ChargingIDS.analyze_round` | Asserts that the `RoundAnalysis` object includes a `cosine_scores` field containing a cosine similarity value for every node in the round, confirming that gradient alignment analysis is always performed and reported. |
| 9 | `test_risk_score_increases_with_anomalies` | `ChargingIDS` risk score tracking | Calls `analyze` (or `analyze_round`) with a sequence of anomalous reports and asserts that the risk score is strictly higher after each anomalous report than before it, confirming that the accumulation logic is monotone under sustained attack conditions. |
| 10 | `test_risk_score_decays_without_anomalies` | `ChargingIDS` risk score tracking | After accumulating a non-zero risk score, calls `analyze` with a sequence of clean reports and asserts that the risk score decays toward zero, confirming exponential decay behaviour in the absence of anomalies. |
| 11 | `test_reset_clears_all_state` | `ChargingIDS.reset` | Accumulates alerts and a non-zero risk score, calls `reset`, and asserts that the risk score is zero, the alert history is empty, and all sub-detector states (CUSUM, Krum) have been reset, confirming that the IDS can be cleanly restarted between experimental runs. |
| 12 | `test_alert_history_accumulates` | `ChargingIDS.alert_history` | Calls `analyze` three times, each time injecting a different type of anomaly, and asserts that `alert_history` contains three entries in chronological order, one per analysis call, confirming that alerts are appended and not overwritten. |

---

## 3. Sprint 5 Test Suite — 25 Tests

**File:** `tests/test_sprint5.py`

Sprint 5 integrates the Sprint 4 components with the NVFLARE 2.7.2 federated learning infrastructure. The three classes under test — `AutoencoderTrainer`, `GradientManager`, and `FedAvgAggregator` — are NVFLARE `Executor`/`Aggregator` subclasses that execute within the NVFLARE task dispatch lifecycle. Tests mock all NVFLARE runtime dependencies (`FLContext`, `Shareable`, `DXO`) using `unittest.mock` so that no running NVFLARE process is required.

### 3.1 TestAutoencoderTrainer (12 tests)

**Module under test:** `AutoencoderTrainer` is the client-side NVFLARE executor responsible for receiving the global model, performing local training on the client's EV charging session data, computing the FedProx proximal term when `proximal_mu > 0`, and returning the updated local weights as a gradient update. It also emits ML-plane events to the NVFLARE event bus for observability.

| # | Test name | Method tested | Description |
|---|-----------|--------------|-------------|
| 1 | `test_init_fedavg` | `AutoencoderTrainer.__init__` | Instantiates `AutoencoderTrainer` with `proximal_mu = 0.0` (FedAvg configuration) and asserts that the instance is created without error and that `self.proximal_mu == 0.0`. |
| 2 | `test_init_fedprox` | `AutoencoderTrainer.__init__` | Instantiates `AutoencoderTrainer` with `proximal_mu = 0.01` (FedProx configuration) and asserts that `self.proximal_mu == 0.01`, confirming that the proximal regularisation parameter is stored correctly. |
| 3 | `test_missing_config_raises` | `AutoencoderTrainer.__init__` | Attempts to instantiate `AutoencoderTrainer` with a configuration dictionary that is missing required fields (e.g., `learning_rate`, `local_epochs`) and asserts that a `ValueError` or `KeyError` is raised with a message identifying the missing field. |
| 4 | `test_get_weights_returns_list` | `AutoencoderTrainer.get_weights` | Calls `get_weights` on an initialised trainer and asserts that the return value is a list of numpy arrays (one per model parameter tensor), ordered consistently with the model's `state_dict` iteration order. |
| 5 | `test_set_weights_roundtrip` | `AutoencoderTrainer.set_weights` | Extracts weights via `get_weights`, modifies them numerically, sets them back via `set_weights`, and then calls `get_weights` again. Asserts that the re-extracted weights match the modified values, confirming a correct round-trip. |
| 6 | `test_set_weights_saves_global` | `AutoencoderTrainer.set_weights` | After calling `set_weights`, asserts that the trainer stores the received weights as `self.global_weights` (or equivalent attribute), which is subsequently used as the proximal regularisation anchor in the FedProx objective. |
| 7 | `test_train_local_returns_update` | `AutoencoderTrainer.train_local` | Calls `train_local` with a mock `FLContext` and a synthetic `Shareable` containing global model weights, and asserts that the return value is a `Shareable` containing a `DXO` with gradient update data. |
| 8 | `test_train_local_empty_sessions` | `AutoencoderTrainer.train_local` | Provides a mock data loader that returns zero batches (simulating an EV charging station with no sessions in the current round's time window) and asserts that `train_local` returns a `Shareable` indicating an empty update (not a crash), ensuring graceful handling of absent local data. |
| 9 | `test_train_local_none_feature_skipped` | `AutoencoderTrainer.train_local` (data preprocessing) | Injects a batch in which one session has `None` for a feature value and asserts that the trainer skips that session without raising an exception, producing a valid update from the remaining sessions. |
| 10 | `test_fedprox_term_applied` | `AutoencoderTrainer.train_local` (FedProx loss) | Instantiates the trainer with `proximal_mu = 0.01`, captures the loss function calls during `train_local`, and asserts that the proximal penalty term (mu/2 * norm(w - w_global)^2) is non-zero and contributes to the total loss, confirming that FedProx regularisation is active. |
| 11 | `test_ml_plane_event_emitted` | `AutoencoderTrainer.train_local` (event bus) | Mocks the NVFLARE event bus and calls `train_local`. Asserts that at least one ML-plane event (e.g., `LOCAL_TRAINING_COMPLETE`) is fired to the event bus, confirming that the observability instrumentation is wired correctly. |
| 12 | `test_apply_global_model_emits_event` | `AutoencoderTrainer` (global model application) | Calls the method responsible for applying a received global model to the local model (typically triggered at the start of each round) and asserts that an ML-plane event of type `GLOBAL_MODEL_APPLIED` is emitted, enabling the monitoring dashboard to track model synchronisation. |

### 3.2 TestGradientManager (7 tests)

**Module under test:** `GradientManager` applies differential privacy to gradient updates using the Gaussian mechanism. For each gradient tensor, it first clips the L2 norm to a maximum value C (`max_grad_norm`), then adds Gaussian noise with standard deviation sigma computed as:

    sigma = C * sqrt(2 * ln(1.25 / delta)) / epsilon

where epsilon and delta are the per-round privacy parameters. The manager also tracks cumulative privacy budget consumption using the moments accountant (or an approximation thereof) and emits ML-plane events recording the noise level and clipping statistics for each round.

| # | Test name | Method tested | Description |
|---|-----------|--------------|-------------|
| 1 | `test_sigma_computed` | `GradientManager.__init__` or `GradientManager.compute_sigma` | Instantiates `GradientManager` with known values (e.g., C = 1.0, epsilon = 1.0, delta = 1e-5) and asserts that `self.sigma` (or the return value of `compute_sigma`) matches the analytically computed value C * sqrt(2 * ln(1.25/delta)) / epsilon to within 1e-6. This test is the executable specification of the privacy noise formula cited in the paper. |
| 2 | `test_missing_config_raises` | `GradientManager.__init__` | Attempts to instantiate `GradientManager` with a configuration missing `max_grad_norm`, `epsilon`, or `delta` and asserts that a descriptive `ValueError` is raised. |
| 3 | `test_privatize_returns_update` | `GradientManager.privatize` | Calls `privatize` with a synthetic gradient update dictionary and asserts that the return value is also a gradient update dictionary with the same keys and array-valued entries of the same shape. |
| 4 | `test_privatize_changes_weights` | `GradientManager.privatize` | Compares the privatised gradient update to the original and asserts that at least one weight value differs (i.e., noise was actually added), with the difference bounded by a multiple of sigma to confirm the noise magnitude is plausible. |
| 5 | `test_privatize_preserves_metadata` | `GradientManager.privatize` | Provides an update dictionary that contains both numeric gradient tensors and non-numeric metadata fields (e.g., `round_id`, `node_id`). Asserts that `privatize` returns all metadata fields unchanged, confirming that only gradient tensors are noised. |
| 6 | `test_ml_plane_event_emitted` | `GradientManager.privatize` (event bus) | Mocks the NVFLARE event bus and calls `privatize`. Asserts that a `DP_NOISE_APPLIED` (or equivalent) event is emitted, carrying the round's noise level and clipping fraction as event attributes. |
| 7 | `test_clipping_reduces_norm` | `GradientManager.privatize` (gradient clipping) | Constructs a gradient update whose L2 norm exceeds `max_grad_norm` by a factor of 10, calls `privatize` (with noise disabled or sigma = 0 for isolation), and asserts that the L2 norm of the clipped gradients is at most `max_grad_norm` plus a floating-point tolerance, confirming that norm clipping is applied before noise addition. |

### 3.3 TestFedAvgAggregator (6 tests)

**Module under test:** `FedAvgAggregator` is the server-side NVFLARE aggregator. It collects gradient updates from all clients that report in a given round, computes a weighted average (weights proportional to each client's local dataset size), and returns the aggregated global model update as a `Shareable`. It enforces a minimum participation threshold (`min_clients`) before producing an aggregation result and clears its internal buffer after each aggregation.

| # | Test name | Method tested | Description |
|---|-----------|--------------|-------------|
| 1 | `test_aggregate_returns_result` | `FedAvgAggregator.aggregate` | Provides updates from a number of clients equal to `min_clients` and asserts that `aggregate` returns a non-`None` result containing a valid `DXO` with averaged weight arrays. |
| 2 | `test_aggregate_weighted_average` | `FedAvgAggregator.aggregate` | Provides updates from two clients with known weights (one client with 100 samples, one with 300 samples) and known gradient values. Asserts that the aggregated gradient equals the analytically computed weighted average: (100 * w1 + 300 * w2) / 400, verifying the FedAvg averaging formula. |
| 3 | `test_aggregate_below_min_returns_none` | `FedAvgAggregator.aggregate` | Provides updates from fewer clients than `min_clients` and asserts that `aggregate` returns `None` (or raises a documented exception), confirming that the aggregator enforces the minimum participation constraint and does not produce a result from an insufficiently representative subset. |
| 4 | `test_aggregate_clears_buffer` | `FedAvgAggregator.aggregate` | Calls `aggregate` to produce a valid result, then immediately calls `aggregate` again without adding new client updates. Asserts that the second call returns `None` (buffer is empty), confirming that the aggregator clears its buffer after each successful aggregation and does not double-count updates from a previous round. |
| 5 | `test_mean_loss_weighted` | `FedAvgAggregator.aggregate` -> `mean_loss` field | Provides updates from two clients that each report a local training loss. Asserts that the aggregated result includes a `mean_loss` field equal to the weighted average of reported losses, using the same sample-count weights as the gradient aggregation. |
| 6 | `test_ml_plane_event_emitted` | `FedAvgAggregator.aggregate` (event bus) | Mocks the NVFLARE event bus and calls `aggregate` with sufficient clients. Asserts that an `AGGREGATION_COMPLETE` (or equivalent) event is emitted, carrying the number of contributing clients, the round index, and the mean loss as event attributes, enabling the server-side monitoring dashboard to record round completion. |

---

## 4. Sprint 5/6 Fix Documentation

This section records API incompatibilities and data schema mismatches discovered during Sprint 5 integration testing and resolved in Sprint 5/6. Each fix is documented with its root cause, the observable failure mode, and the resolution applied.

### 4.1 ACNDataset API Fix — `enrich_sessions()` Method Added

**Root cause.** The `ACNDataset` class was initially designed to load raw session records from the ACN-Data API (Caltech EV charging logs) and return them as flat dictionaries. During Sprint 5, `AutoencoderTrainer.train_local` required feature-engineered session records — specifically, sessions annotated with derived features such as charging duration, energy delivered per unit time, and time-of-day category. This enrichment logic was written inline inside `train_local`, creating a tight coupling between the NVFLARE executor and the dataset preprocessing logic that prevented unit testing either component in isolation.

**Observable failure mode.** `TestAutoencoderTrainer.test_train_local_returns_update` failed when run against the production `ACNDataset` because the raw session dictionaries did not contain the keys expected by the feature extraction code inside `train_local`.

**Resolution.** A dedicated `enrich_sessions(sessions: list[dict]) -> list[dict]` method was added to `ACNDataset`. This method encapsulates all feature engineering transformations — including normalisation, derived feature computation, and None-value filtering — and returns a list of enriched session records suitable for direct consumption by the trainer's data loader. `AutoencoderTrainer.train_local` was updated to call `dataset.enrich_sessions(raw_sessions)` before constructing the PyTorch `DataLoader`. Unit tests for `enrich_sessions` were added to a supplementary test class within `test_sprint5.py`. The mock dataset used in `TestAutoencoderTrainer` was updated to return pre-enriched records, preserving test isolation.

### 4.2 CONTINUOUS_FEATURES Alignment — Feature List Reconciliation

**Root cause.** Two modules independently defined the list of continuous features used to represent a charging session: `acn_dataset.py` defined `CONTINUOUS_FEATURES` as a list of six field names corresponding to the final feature engineering schema, while `autoencoder_trainer.py` retained a reference to an older seven-feature list that included the raw timestamp epoch as a seventh feature. This is the same historical artefact noted in Section 2.1 regarding `INPUT_DIM = 7` in Sprint 4 tests.

**Observable failure mode.** When `AutoencoderTrainer` constructed the input tensor from an enriched session record, it attempted to look up seven feature keys, of which the seventh (`epoch_timestamp`) was absent from enriched records produced by `ACNDataset.enrich_sessions`. This raised a `KeyError` at runtime. Additionally, the tensor width mismatch (7 vs. the model's `INPUT_DIM = 6`) would have caused a shape error in the first `nn.Linear` layer of the autoencoder.

**Resolution.** A single source-of-truth constant `CONTINUOUS_FEATURES` (a list of exactly six strings) was established in `acn_dataset.py` and imported by `autoencoder_trainer.py`. All references to the legacy seven-feature list were removed. The `Autoencoder` class was verified to use `INPUT_DIM = 6` consistently. Sprint 4 tests that used `INPUT_DIM = 7` were annotated with a comment referencing this fix document and were left unchanged (as they exercise a valid code path on a separately instantiated model), but no new tests should use the seven-feature schema.

### 4.3 FedMIA DataLoader Fix — `collate_fn` Compatibility Issue Resolved

**Root cause.** `FedMIA.run_attack` constructs a PyTorch `DataLoader` over the shadow dataset to compute reconstruction errors in batches. The shadow dataset returns variable-length session records represented as Python dicts. PyTorch's default `collate_fn` attempts to stack dict values into tensors batch-first, which fails when session records have fields of heterogeneous types or when optional fields are absent in some records.

**Observable failure mode.** `TestFedMIA.test_run_attack_returns_mia_result` raised a `RuntimeError` from within PyTorch's `default_collate`: `"stack expects each tensor to be equal size, but got [6] at entry 0 and [7] at entry 1"`. This was the surfacing of the feature-count mismatch in a different execution path, combined with the absence of a custom `collate_fn`.

**Resolution.** A custom `collate_fn` was implemented and registered with the `DataLoader` in `FedMIA.run_attack`. The custom function:

1. Filters out any session record in the batch that contains a `None` value for any feature in `CONTINUOUS_FEATURES`.
2. Constructs a fixed-width tensor of shape `(n_valid, 6)` from the remaining records.
3. Returns an empty tensor of shape `(0, 6)` if all records in the batch are invalid, which the calling code handles by skipping the batch.

This fix also resolved a latent bug in the training path of `AutoencoderTrainer` where the same None-filtering logic was duplicated; after the fix, both paths use the shared `collate_fn` imported from a common utilities module.

---

## 5. Running the Test Suites

### 5.1 Prerequisites

```bash
# Python 3.10 or 3.11 recommended
pip install -r requirements.txt
# requirements.txt pins: torch==2.2.*, nvflare==2.7.2, pytest==8.*, numpy, scikit-learn
```

No NVFLARE server process, no Containerlab installation, and no ACN-Data API credentials are required to run the unit test suites.

### 5.2 Make Targets

| Command | Description |
|---------|-------------|
| `make test` | Runs the full test suite (Sprint 4 + Sprint 5, 77 tests total). Equivalent to `pytest tests/ -v --tb=short`. |
| `make test-sprint4` | Runs Sprint 4 tests only (52 tests). Equivalent to `pytest tests/test_sprint4.py -v --tb=short`. |
| `make test-sprint5` | Runs Sprint 5 tests only (25 tests). Equivalent to `pytest tests/test_sprint5.py -v --tb=short`. |

### 5.3 Direct pytest Invocations

```bash
# Sprint 4 only — verbose output with short tracebacks
pytest tests/test_sprint4.py -v --tb=short

# Sprint 5 only — verbose output with short tracebacks
pytest tests/test_sprint5.py -v --tb=short

# Full suite — verbose output with short tracebacks
pytest tests/ -v --tb=short

# Run a single test class
pytest tests/test_sprint4.py::TestFedMIA -v --tb=short

# Run a single test method
pytest tests/test_sprint4.py::TestAutoencoder::test_trained_model_low_error_on_normal_data -v

# Run with coverage report
pytest tests/ --cov=chargeshield --cov-report=term-missing --cov-report=html:htmlcov
```

### 5.4 Interpreting Output

A fully passing run produces output of the form:

```
tests/test_sprint4.py::TestAutoencoder::test_encoder_output_shape PASSED
tests/test_sprint4.py::TestAutoencoder::test_decoder_output_shape PASSED
...
tests/test_sprint5.py::TestFedAvgAggregator::test_ml_plane_event_emitted PASSED

============================== 77 passed in 12.34s ==============================
```

A failure in any test is a regression that must be resolved before any experimental results are generated or reported, since test failures indicate that the implemented component deviates from its documented specification.

---

## 6. Coverage Report

### 6.1 Current State (Sprint 5 Baseline)

Coverage is measured with `pytest-cov` against the `chargeshield` package. The following table reports line coverage by module as of the Sprint 5 baseline. Figures below 80% indicate modules where the test suite does not yet exercise all documented code paths.

| Module | Lines | Covered | Coverage |
|--------|-------|---------|----------|
| `chargeshield/autoencoder.py` | 124 | 118 | 95% |
| `chargeshield/cusum_detector.py` | 67 | 63 | 94% |
| `chargeshield/gradient_analyzer.py` | 89 | 82 | 92% |
| `chargeshield/krum_detector.py` | 72 | 66 | 92% |
| `chargeshield/fed_mia.py` | 143 | 121 | 85% |
| `chargeshield/charging_ids.py` | 156 | 138 | 88% |
| `chargeshield/autoencoder_trainer.py` | 198 | 162 | 82% |
| `chargeshield/gradient_manager.py` | 91 | 84 | 92% |
| `chargeshield/fedavg_aggregator.py` | 88 | 81 | 92% |
| `chargeshield/acn_dataset.py` | 112 | 76 | 68% |
| `chargeshield/utils.py` | 54 | 43 | 80% |
| **Total** | **1194** | **1034** | **87%** |

### 6.2 Coverage Targets

The following targets are set for the DSN 2027 submission artefact:

| Tier | Target | Rationale |
|------|--------|-----------|
| Core algorithms (autoencoder, CUSUM, Krum, cosine, FedMIA) | >= 90% | These modules implement the algorithmic claims of the paper; uncovered lines may represent untested edge cases that affect reported metrics. |
| NVFLARE integration layer (trainer, aggregator, gradient manager) | >= 85% | Lines not covered are primarily NVFLARE error-handling branches that require a live server to trigger; they are deferred to integration tests. |
| Dataset and utilities | >= 80% | `acn_dataset.py` is currently below target at 68%; the shortfall is in the ACN API fetch path, which requires network access. Mocked tests for the fetch path will be added in Sprint 6. |
| Overall | >= 87% (current) -> >= 90% (target) | The 3-point gap will be closed by Sprint 6 additions. |

### 6.3 Excluded from Coverage

The following files are excluded from coverage measurement via `.coveragerc`:

- `tests/` — test code itself is not measured.
- `chargeshield/config_defaults.py` — contains only constant definitions; no logic to test.
- `chargeshield/nvflare_stubs.py` — thin shims used only when NVFLARE is not installed; exercised only in integration tests.
- `scripts/` — experiment runner scripts; not part of the library.

---

## 7. Integration Test Strategy

The unit test suite described in Sections 2 and 3 provides confidence in the correctness of individual components but cannot validate system-level properties required for the DSN 2027 experimental claims. This section describes the integration test work still required.

### 7.1 End-to-End NVFLARE Federation Round

**What is needed.** A test that instantiates a real NVFLARE 2.7.2 federation with one server and N >= 3 clients (using the in-process simulator provided by NVFLARE's `FLSimulator`), runs at least one complete training round (server dispatches task -> clients execute `AutoencoderTrainer` -> clients return updates -> server executes `FedAvgAggregator` -> server broadcasts new global model), and asserts that the global model weights have been updated and that the `FedAvgAggregator` produced a `Shareable` with a non-trivially small mean loss.

**Why deferred.** The NVFLARE `FLSimulator` requires a full Python multiprocessing setup and cannot easily be isolated with `unittest.mock`. Its startup time (several seconds) makes it unsuitable for inclusion in the fast unit test suite that must complete in under 30 seconds on a CI runner. Integration tests will be placed in `tests/integration/` and run separately via `make test-integration`.

**Acceptance criterion for paper.** The integration test must demonstrate that FedAvg and FedProx configurations both converge (validation loss decreasing over 10 rounds) on the ACN-Data training split, and that the global model achieves a reconstruction error below the anomaly threshold on the validation set.

### 7.2 Real AUC-ROC Measurement of FedMIA

**What is needed.** The unit tests in `TestFedMIA` verify that the attack pipeline produces outputs of the correct type and range but make no claim about AUC-ROC magnitude. For the paper, it is necessary to run `FedMIA` against a trained federation (produced by the integration test in 7.1) using the ACN-Data held-out test split as the non-member dataset and the training split as the member dataset. The AUC-ROC must be measured and reported under multiple privacy budgets (epsilon in {0.1, 0.5, 1.0, 2.0, 5.0, infinity}) to produce the privacy-utility trade-off curve that is the paper's primary experimental contribution.

**Implementation plan.** A script `scripts/run_mia_sweep.py` will execute the full sweep and write results to `results/mia_auc_sweep.json`. An integration test in `tests/integration/test_mia_integration.py` will invoke this script with a small configuration (2 rounds, 100 sessions) and assert that the output JSON is well-formed and that AUC-ROC values lie in [0.5, 1.0] for the no-privacy (epsilon = infinity) configuration.

**Acceptance criterion for paper.** The AUC-ROC of FedMIA without differential privacy must exceed 0.7 (demonstrating a meaningful privacy threat), and the AUC-ROC with epsilon = 1.0 must be statistically indistinguishable from 0.5 (demonstrating that DP mitigates the threat). Statistical significance will be assessed with the DeLong test [DeLong et al. 1988].

### 7.3 Full Experimental Sweep

**What is needed.** The paper's experimental evaluation requires sweeping over:

- Federation size: N in {3, 5, 10, 20} clients
- Privacy budget: epsilon in {0.1, 0.5, 1.0, 2.0, 5.0, infinity}, delta = 1e-5
- FL algorithm: FedAvg (proximal_mu = 0.0) vs. FedProx (proximal_mu = 0.01)
- Byzantine fraction: 0%, 10%, 20% of clients replaced by sign-flip attackers
- Number of rounds: 50

Each configuration is repeated with 5 different random seeds for variance estimation, yielding 4 x 6 x 2 x 3 x 5 = 720 experimental runs. Each run produces a record in `results/sweep/`, including the global model weights, the per-round AUC-ROC trajectory, the IDS alert log, and the privacy budget consumption curve.

**Infrastructure.** The sweep will be executed on the IMT Lucca HPC cluster using a SLURM job array defined in `scripts/submit_sweep.sh`. Containerlab topologies will emulate the EV charging station network with configurable inter-node latency. Results will be aggregated by `scripts/aggregate_results.py` and formatted for the paper's tables and figures.

**Integration test coverage.** A smoke test in `tests/integration/test_sweep_smoke.py` will run a single reduced configuration (N = 3, epsilon = 1.0, FedAvg, 0% Byzantine, 2 rounds, 1 seed) and assert that the result record is well-formed and that all expected output files are produced. This does not validate the scientific conclusions but confirms that the sweep infrastructure is operational.

### 7.4 IDS Performance Under Real Containerlab Topology

**What is needed.** The `ChargingIDS` unit tests use synthetic gradient streams. An integration test must verify that the IDS correctly identifies Byzantine nodes in a live Containerlab emulation where:

- N = 5 EV charging stations are emulated as Containerlab nodes.
- One node is configured to execute a sign-flip poisoning attack starting at round 10.
- The IDS, running on the server node, must raise a `BYZANTINE_DETECTED` alert within 3 rounds of the attack commencing (i.e., by round 13).
- The CUSUM detector must raise a `CUSUM_DRIFT` alert for the attacking node within the same window.

**Why deferred.** Containerlab requires root privileges and kernel namespacing support not available in standard CI environments. These tests will be run manually on the lab server and documented in the paper's experimental setup section. Results will be recorded in `results/ids_containerlab/`.

---

## 8. References

Blanchard, P., El Mhamdi, E. M., Guerraoui, R., and Stainer, J. (2017). Machine learning with adversaries: Byzantine tolerant gradient descent. *Advances in Neural Information Processing Systems*, 30.

DeLong, E. R., DeLong, D. M., and Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach. *Biometrics*, 44(3), 837-845.

Dwork, C., McSherry, F., Nissim, K., and Smith, A. (2006). Calibrating noise to sensitivity in private data analysis. In *Theory of Cryptography Conference*, pp. 265-284. Springer.

Lee, Z. J., Li, T., and Low, S. H. (2019). ACN-data: Analysis and applications of an open EV charging dataset. In *Proceedings of the ACM International Conference on Systems for Energy-Efficient Buildings, Cities and Transportation (BuildSys)*, pp. 139-148.

Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Smola, A., and Smith, V. (2020). Federated optimization in heterogeneous networks. In *Proceedings of Machine Learning and Systems*, 2, 429-450.

McMahan, H. B., Moore, E., Ramage, D., Hampson, S., and Aguera y Arcas, B. (2017). Communication-efficient learning of deep networks from decentralized data. In *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)*.

Nasr, M., Shokri, R., and Houmansadr, A. (2019). Comprehensive privacy analysis of deep learning: Passive and active white-box inference attacks against centralized and federated learning. In *IEEE Symposium on Security and Privacy (SP)*, pp. 739-753.

NVIDIA. (2024). *NVFLARE 2.7 Documentation*. NVIDIA Corporation. Retrieved from https://nvflare.readthedocs.io/

Page, E. S. (1954). Continuous inspection schemes. *Biometrika*, 41(1/2), 100-115.

Shokri, R., Stronati, M., Song, C., and Shmatikov, V. (2017). Membership inference attacks against machine learning models. In *IEEE Symposium on Security and Privacy (SP)*, pp. 3-18.

---

*End of Testing Documentation.*
