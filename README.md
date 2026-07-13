# ChargeShield-FL

A research framework for evaluating Membership Inference Attacks and differential privacy defences in Federated Learning systems deployed across heterogeneous Electric Vehicle charging infrastructure.

---

## Abstract

ChargeShield-FL is an open research framework designed to empirically evaluate the privacy guarantees of Federated Learning (FL) in the context of Electric Vehicle (EV) charging networks, with a particular focus on Membership Inference Attacks (MIA) and the effectiveness of differential privacy (DP) as a countermeasure. As smart-grid deployments increasingly adopt FL to train shared models over distributed charging stations without centralising raw session data, the question of whether individual charging sessions can be re-identified from model updates becomes a critical safety and regulatory concern. The framework instantiates a realistic, heterogeneous topology of 12 nodes across four cluster types — Highway, Urban, Residential, and Corporate — each governed by distinct communication protocols (OCPP 1.6, OCPP 2.0.1, MQTT v5) and power profiles, trained on 13,073 real EV sessions drawn from the ACN-Data JPL dataset (2019–2020). MIA evaluation is performed via a loss-based per-round evaluator (Yeom et al. 2018) embedded in the experiment pipeline: at each FL round the global weights are loaded into the Autoencoder, membership scores are computed as −MSE, and AUC-ROC is measured per round via scikit-learn; summary statistics (mean, max, min AUC-ROC across rounds) are reported in the experiment JSON. A separate shadow-model-based FedMIA plugin (`src/plugins/attacks/fedmia.py`) is used by ChargingIDS for per-node IDS scoring and remains unchanged. Both mechanisms are integrated alongside CUSUM, Krum, and Cosine Similarity intrusion detection baselines, enabling controlled measurement of attack success across a full sweep of FL aggregation strategies (FedAvg, FedProx) and privacy budgets (ε ∈ {0.1, 0.5, 1.0, 2.0, 5.0}). Initial results from a 100-round experiment at ε = 1.0 yield mean AUC-ROC = 0.5172 across rounds, confirming that the Gaussian Mechanism at standard privacy budgets is effective at suppressing membership leakage to near-random-guess levels; ChargeShield-FL targets publication at the IEEE/IFIP International Conference on Dependable Systems and Networks (DSN) 2027.

---

## Why ChargeShield-FL?

### The Privacy Problem in EV Federated Learning

Electric Vehicle charging sessions encode remarkably fine-grained behavioural information. A single session record — encompassing the energy requested, the peak power draw, the time of arrival, and the duration of stay — is sufficient to infer a user's home address, workplace, daily routine, health status, and socioeconomic profile. As national charging networks scale to millions of sessions per day, centralising raw data for model training becomes both a regulatory liability (GDPR Art. 5, CCPA) and a security risk.

Federated Learning is widely proposed as the privacy-preserving alternative: clients train locally, and only model updates (gradients or weights) are aggregated centrally. However, a substantial body of research demonstrates that model updates are not informationally inert. Shokri et al. (2017) showed that gradient updates leak membership, and Carlini et al. (2022) confirmed that even aggregated models trained with FedAvg retain non-trivial membership signals. In the EV domain, this means an adversary with access to the aggregated FL model — including a semi-honest FL server or a compromised aggregation endpoint — can, in principle, determine whether a specific charging session was used in training.

### Why EV Infrastructure Specifically?

Three converging trends make this problem urgent:

1. **Scale and heterogeneity.** Modern charging networks span residential 7 kW AC sockets, urban 22 kW AC posts, highway 150 kW DC fast chargers, and corporate 50 kW DC installations, each governed by different communication protocols (OCPP 1.6, OCPP 2.0.1, MQTT v5). Protocol heterogeneity introduces non-IID data distributions across FL clients, which affects both model utility and privacy guarantees in ways that are not well characterised in the literature.

2. **Regulatory pressure.** The EU Alternative Fuels Infrastructure Regulation (AFIR, 2023) and the US National Electric Vehicle Infrastructure (NEVI) programme both mandate interoperability and data sharing between charging operators, increasing the attack surface for cross-operator MIA.

3. **Absence of rigorous benchmarks.** Despite a growing literature on FL privacy in healthcare and finance, no publicly reproducible framework exists for evaluating MIA risk in EV charging systems at realistic scale, with real session data, realistic network topologies, and integrated DP accounting.

ChargeShield-FL fills this gap by providing a fully reproducible, containerised experimental environment that can serve both as a research instrument and as a compliance evaluation tool for charging network operators.

---

## Architecture Overview

```
+-----------------------------------------------------------------------------+
|                          ChargeShield-FL Topology                           |
|                                                                             |
|                         +---------------------+                             |
|                         |    FL SERVER        |                             |
|                         |    (NVFLARE 2.7.2)  |                             |
|                         |  +---------------+  |                             |
|                         |  |  FedAvg /     |  |                             |
|                         |  |  FedProx      |  |                             |
|                         |  |  Aggregator   |  |                             |
|                         |  +---------------+  |                             |
|                         |  +---------------+  |                             |
|                         |  |  DP Layer     |  |                             |
|                         |  |  Gaussian Mech|  |                             |
|                         |  |  s = f(e, d)  |  |                             |
|                         |  +---------------+  |                             |
|                         +----------+----------+                             |
|                   mTLS / WireGuard |                                        |
|         +--------------+-----------+-----------+--------------+             |
|         |              |                       |              |             |
|  +------+-------+ +----+----------+ +----------+------+ +----+----------+  |
|  |  HIGHWAY     | |  URBAN        | |  RESIDENTIAL    | |  CORPORATE    |  |
|  |  CLUSTER     | |  CLUSTER      | |  CLUSTER        | |  CLUSTER      |  |
|  |  3 nodes     | |  3 nodes      | |  3 nodes        | |  3 nodes      |  |
|  |  OCPP 1.6    | |  OCPP 1.6     | |  MQTT v5        | |  OCPP 2.0.1   |  |
|  |  150 kW DC   | |  22 kW AC     | |  7 kW AC        | |  50 kW DC     |  |
|  |              | |               | |                 | |               |  |
|  | +----------+ | | +----------+  | | +-------------+ | | +----------+  |  |
|  | |Autoencoder| | | |Autoencoder| | | |Autoencoder  | | | |Autoencoder|  |  |
|  | |6->16->8->4| | | |6->16->8->4| | | |6->16->8->4  | | | |6->16->8->4|  |  |
|  | |+ Decoder  | | | |+ Decoder  | | | |+ Decoder    | | | |+ Decoder  |  |  |
|  | +----------+ | | +----------+  | | +-------------+ | | +----------+  |  |
|  +--------------+ +---------------+ +-----------------+ +---------------+  |
|                                                                             |
|         +-------------------------------------------------+                |
|         |               MONITORING PLANE                  |                |
|         |  +----------+  +----------+  +---------------+  |                |
|         |  |  CUSUM   |  |  Krum    |  |   Cosine      |  |                |
|         |  |  IDS     |  |  Filter  |  |  Similarity   |  |                |
|         |  +----------+  +----------+  +---------------+  |                |
|         +-------------------------------------------------+                |
|                                                                             |
|         +-------------------------------------------------+                |
|         |               ATTACKER PLANE (FedMIA)          |                |
|         |  +-------------------------------------------+  |                |
|         |  |  (a) FedMIA Plugin (IDS, per-node)        |  |                |
|         |  |      Shadow Model on public ACN split     |  |                |
|         |  |      Reconstruction Error -> Membership   |  |                |
|         |  |      Score; used by ChargingIDS           |  |                |
|         |  +-------------------------------------------+  |                |
|         |  +-------------------------------------------+  |                |
|         |  |  (b) FedMIA Evaluator (run_experiments)   |  |                |
|         |  |      Loss-based per-round (Yeom 2018)     |  |                |
|         |  |      Score = -MSE on global_weights       |  |                |
|         |  |      AUC-ROC measured each FL round       |  |                |
|         |  +-------------------------------------------+  |                |
|         +-------------------------------------------------+                |
+-----------------------------------------------------------------------------+
```

---

## Components

| Component | Module | Role |
|---|---|---|
| FL Aggregation Server | NVFLARE 2.7.2 | Coordinates federated rounds; applies FedAvg or FedProx aggregation; enforces DP clipping and noise injection |
| Local Autoencoder | PyTorch (6→16→8→4→8→16→6) | Per-client anomaly detector trained on local EV session features; produces reconstruction error as membership signal |
| Differential Privacy Layer | Gaussian Mechanism | Clips per-sample gradients to max_grad_norm; adds calibrated Gaussian noise with σ = max_grad_norm × √(2 ln(1.25/δ)) / ε before upload |
| FedMIA Plugin (`src/plugins/attacks/fedmia.py`) | Shadow model on public ACN split | Trains a reference autoencoder on held-out public data; uses reconstruction error gap between members and non-members to produce per-node membership scores; used by ChargingIDS for IDS scoring |
| FedMIA Evaluator — Loss-based (`scripts/run_experiments.py`) | Loss-based per-round evaluator (Yeom et al. 2018) | At each FL round loads global weights into the Autoencoder; computes membership score as −MSE; measures AUC-ROC via scikit-learn per round; JSON output includes `per_round[round]["auc_roc"]` and summary `mean_auc_roc`, `max_auc_roc`, `min_auc_roc` |
| FedMIA Evaluator — Shadow/Calibrated (`scripts/run_experiments.py`) | Calibrated shadow-model attack (Carlini et al. 2022) | Trains a local shadow autoencoder on 50% of the training set; for each FL round computes calibrated score = MSE(shadow, x) − MSE(target, x); controls for per-sample reconstruction difficulty; JSON output includes `shadow_auc_roc`, `shadow_score_gap` per round |
| CUSUM IDS Baseline | Sequential CUSUM statistic | Detects distributional drift in incoming gradient magnitudes; triggers alert when cumulative sum exceeds threshold |
| Krum IDS Baseline | Multi-Krum filter | Rejects client updates that are Euclidean outliers relative to the median neighbourhood; provides Byzantine resilience baseline |
| Cosine Similarity IDS | Pairwise cosine distance | Flags updates that deviate in direction from the running aggregate; identifies gradient inversion style anomalies |
| Network Fabric | Containerlab + Docker + OrbStack | Emulates the heterogeneous charging network topology; manages container lifecycle and inter-node routing |
| Transport Security | mTLS + WireGuard | Provides mutual authentication and encrypted tunnels between FL clients and server; eliminates passive eavesdropping from the threat model |
| Dataset Pipeline | ACN-Data JPL 2019+2020 | Preprocesses, splits, and distributes 13,073 real EV sessions across cluster clients according to cluster power profile |
| Experiment Orchestrator | GNU Make + Python | Drives round sweeps, DP budget sweeps, result logging, and AUC-ROC aggregation via Makefile targets |

---

## Threat Model

| Threat | Attacker Type | Defence | Metric |
|---|---|---|---|
| Membership Inference on training sessions | Honest-but-curious FL server; external adversary with model access | Gaussian Mechanism DP (ε ∈ {0.1–5.0}, δ = 1e-5) | Two parallel evaluators: (1) loss-based (Yeom 2018): mean AUC-ROC per round; (2) calibrated shadow attack (Carlini 2022): `shadow_auc_roc` and `shadow_score_gap` per round. Loss-based AUC ≈ 0.50 at all tested ε indicates the attack is too weak relative to the model generalisation level; shadow attack is the primary signal for DSN 2027 |
| Gradient Inversion (reconstruction of raw session data) | Active server-side attacker | DP noise injection; mTLS transport integrity | Reconstruction MSE on held-out sessions |
| Byzantine update poisoning | Malicious FL client submitting corrupted updates | Krum aggregation filter; Cosine Similarity anomaly detection | Attack detection rate; model accuracy degradation |
| Distributional shift / concept drift exploitation | Compromised client inflating local loss | CUSUM sequential monitoring | False positive rate; detection latency in rounds |
| Network-level eavesdropping | Passive adversary on inter-node links | WireGuard VPN tunnels; mTLS certificate pinning | N/A (eliminated by design) |
| Model extraction via repeated query | Black-box query adversary | Rate limiting (not yet implemented; Sprint 7 target) | Query efficiency bound |

---

## Dataset

**Source:** ACN-Data, Adaptive Charging Network, Caltech / JPL Campus  
**URL:** https://ev.caltech.edu/dataset  
**Coverage:** 2019 and 2020 calendar years  
**Sessions:** 13,073 real EV charging sessions  
**Licence:** Caltech ACN-Data research licence (non-commercial academic use)

### Feature Schema

| Feature | Unit | Description |
|---|---|---|
| `total_energy_kwh` | kWh | Total energy delivered in the session |
| `max_power_kw` | kW | Peak power draw during the session |
| `kwh_requested` | kWh | Energy requested by the vehicle at session initiation |
| `minutes_available` | min | Time the vehicle remained plugged in |
| `hour_of_day` | h (0–23) | Wall-clock hour at session start |
| `duration_hours` | h | Elapsed time from plug-in to plug-out |

### Distribution to Clusters

Sessions are partitioned across the four cluster types according to power profile compatibility: Highway nodes receive sessions with `max_power_kw` > 50; Corporate nodes receive sessions with `max_power_kw` in (20, 50]; Urban nodes receive sessions with `max_power_kw` in (10, 20]; Residential nodes receive sessions with `max_power_kw` <= 10. This produces a realistic non-IID distribution across FL clients, reflecting the structural heterogeneity of a real charging network.

---

## Infrastructure

### Containerisation and Network Emulation

The experimental topology is instantiated using **Containerlab** (https://containerlab.dev), which defines the 12-node network graph declaratively in YAML and provisions Docker containers as virtual charging nodes. Each container runs a NVFLARE FL client process alongside a simulated OCPP or MQTT endpoint. **OrbStack** is used on macOS development hosts as a high-performance Docker runtime with native Linux kernel support, reducing container startup latency.

### Transport Security

All FL client-to-server communication is protected by two layers:

- **mTLS (mutual TLS):** Every container presents a client certificate signed by the experiment's internal CA. The NVFLARE server rejects connections from uncertified clients, preventing spoofed participant injection.
- **WireGuard VPN:** An overlay VPN mesh encrypts all inter-container IP traffic, eliminating passive eavesdropping from the threat model even on shared Docker bridge networks.

### Protocol Endpoints

| Cluster | Protocol | Stack |
|---|---|---|
| Highway | OCPP 1.6 (WebSocket/JSON) | ocpp Python library, 150 kW DC profile |
| Urban | OCPP 1.6 (WebSocket/JSON) | ocpp Python library, 22 kW AC profile |
| Residential | MQTT v5 (TLS) | paho-mqtt, 7 kW AC profile |
| Corporate | OCPP 2.0.1 (WebSocket/JSON) | ocpp Python library, 50 kW DC profile |

---

## Key Results

### Experiment 1: Baseline DP Effectiveness (100 rounds, ε = 1.0)

The first completed experiment ran 100 federated rounds with FedAvg aggregation and Gaussian Mechanism DP at ε = 1.0, δ = 1e-5, across all 12 nodes. MIA evaluation used the loss-based per-round evaluator (Yeom et al. 2018): at each round the global weights were loaded into the Autoencoder, membership scores were computed as −MSE, and AUC-ROC was measured via scikit-learn. The reported AUC-ROC is the mean across all 100 rounds.

| Parameter | Value |
|---|---|
| FL algorithm | FedProx (proximal_mu = 0.01) |
| Rounds | 100 |
| Privacy budget ε | 1.0 |
| Privacy budget δ | 1e-5 |
| FedMIA mean AUC-ROC (per-round, Yeom 2018) | **0.5030** |
| Interpretation | Loss-based attack at noise floor; model generalises too well for raw-MSE signal |

A mean AUC-ROC of ~0.503 across all tested configurations (ε ∈ {0.1, 0.5, 1.0, 2.0, 5.0}, rounds ∈ {100, 200, 500, 1000}) — near the theoretical random-guess baseline of 0.50 — indicates that the loss-based MIA (Yeom 2018) is insufficient to extract a membership signal from this model. The Autoencoder on 6 normalised EV features with 4 clusters of ≈2,600 sessions each generalises well without overfitting individual records; the loss-based signal (−MSE) is too coarse to detect the small member/non-member gap that DP further suppresses. This is a known limitation of threshold attacks on well-regularised models (Carlini et al. 2022, §3). Accordingly, the primary MIA evaluator for DSN 2027 is the calibrated shadow attack (`run_fedmia_shadow()`), which controls for per-sample reconstruction difficulty and is expected to produce non-trivial AUC-ROC where the loss-based approach fails.

### Ongoing: Full Parameter Sweep

A systematic sweep across the following parameter grid is currently in progress:

| Axis | Values |
|---|---|
| Rounds | 100, 200, 500, 1000 |
| Privacy budget ε | 0.1, 0.5, 1.0, 2.0, 5.0 |
| FL algorithm | FedAvg, FedProx (proximal_mu = 0.01) |

This yields 20 experimental configurations (4 × 5). The primary hypothesis under evaluation is that AUC-ROC increases monotonically with ε (relaxed privacy) and with the number of training rounds (greater memorisation). FedProx regularisation (proximal_mu = 0.01) is active across all configurations; a FedAvg baseline (proximal_mu = 0) will be evaluated in Sprint 8. Results will be reported in the DSN 2027 submission.

---

## Quickstart

### Prerequisites

**Required for experiments (no Docker needed):**
- Python >= 3.11
- GNU Make
- pip

**Optional — for containerised topology emulation (Sprint 7+):**
- Docker (OrbStack recommended on macOS)
- Containerlab >= 0.54
- WireGuard tools (`wg`, `wg-quick`)

### Installation

```bash
git clone https://github.com/susetta85/ChargeShield-FL.git
cd ChargeShield-FL

# Install all Python dependencies (runtime + dev)
pip install -e ".[dev]" --break-system-packages
```

### Dataset

Download the ACN-Data JPL dataset (2019 and 2020) from https://ev.caltech.edu/dataset and place the JSON files at:

```
datasets/acn/jpl/acndata_sessions_2019.json
datasets/acn/jpl/acndata_sessions_2020.json
```

### Running Experiments

```bash
# Verify config and dataset without training (~5 seconds)
make experiment-dry

# Single experiment: 100 rounds, ε=1.0, FedProx μ=0.01
make experiment

# Full sweep: rounds ∈ {100,200,500,1000} × ε ∈ {0.1,0.5,1.0,2.0,5.0}
# Each run creates a new numbered directory experiments/exp{N}/
# Estimated runtime: ~36h on CPU (sequential; 20 configs × ~1.8h average)
nohup make experiment-full-sweep &

# Monitor progress in real time
tail -f experiments/exp1/sweep_log.txt
```

Results are written to `experiments/exp{N}/`:
- `experiment_{timestamp}.json` — per-config result (one file per completed config, timestamped)
- `exp{N}.xlsx` — 6-sheet Excel report (Raw Data, Heat Map, Per Rounds, Per Epsilon, Comparison, AUC Progression), regenerated after each config completes

Running `make experiment-full-sweep` a second time produces `experiments/exp2/` and so on; sweep directories are never mixed.

### Running Tests

```bash
make test
```

### Teardown (container topology only)

```bash
make destroy   # remove Containerlab containers
make clean     # remove __pycache__ and build artefacts
```

---

## Development Commands

| Makefile Target | Description |
|---|---|
| `make experiment-dry` | Dry run: loads config and dataset, prints summary, exits without training |
| `make experiment` | Single experiment with default parameters (100 rounds, ε=1.0, FedProx μ=0.01); result written to `experiments/experiment_{timestamp}.json` |
| `make experiment-full-sweep` | Full rounds × ε sweep (20 configs); creates `experiments/exp{N}/` with JSON files and `exp{N}.xlsx`; each new run auto-increments N |
| `make test` | Run the complete unit and integration test suite (pytest) |
| `make test-sprint4` | Sprint 4 tests only |
| `make test-sprint5` | Sprint 5 tests only |
| `make lint` | Static analysis via ruff |
| `make clean` | Remove `__pycache__`, `.pyc` files, and pytest artefacts |
| `make build` | Build Docker images for the containerised topology (Sprint 7+; not required for experiments) |
| `make provision` | Provision Containerlab topology and WireGuard mesh (container topology only) |
| `make deploy` | Start NVFLARE server and all 12 FL clients in containers (container topology only) |
| `make destroy` | Tear down the Containerlab topology |

### Experiment Result Scripts

| Script | Description |
|---|---|
| `scripts/run_experiments.py` | Orchestrates a single FL experiment; runs two parallel MIA evaluators per round — loss-based (Yeom 2018, `auc_roc`) and calibrated shadow attack (Carlini 2022, `shadow_auc_roc`); performs 80/20 hold-out split; writes per-round MIA scores, FL `mean_loss`, and IDS delta-weight analysis to `experiment_{timestamp}.json`; accepts `--sweep-dir experiments/exp{N}`; auto-regenerates `exp{N}.xlsx` at completion |
| `scripts/run_sweep.py` | Runs multiple experiments sequentially for a given `--rounds` and `--epsilon` grid; auto-detects next `exp{N}` directory if `--sweep-dir` not provided; logs progress per config; invokes `run_experiments.py` as a subprocess |
| `scripts/generate_excel_report.py` | Standalone tool: reads all `experiment_*.json` files from a given directory and generates a 6-sheet Excel workbook: **Raw Data** (one row per experiment), **Heat Map** (AUC-ROC matrix: rounds × ε), **Per Rounds** (aggregated by round count), **Per Epsilon** (aggregated by ε), **Comparison** (side-by-side metrics), **AUC Progression** (per-round AUC trajectory) |

### Engineering Fixes

Fixes applied during Sprint 7 (code review v6/v7, post-sweep):

- **IDS receives weight deltas, not absolute weights** (`scripts/run_experiments.py`): `run_ids()` now computes `delta_weights = update.weights − prev_global_weights` and passes deltas to both `PrivacyAuditor.audit()` and `ChargingIDS.analyze_round()`. Absolute weights (L2-norm >> max_grad_norm) caused systematic `GRADIENT_EXPLOSION` alerts and cosine similarity ≈ 0.0 for all nodes. Deltas are bounded by `max_grad_norm` by construction.
- **`GradientAnalyzer.flatten()` handles PyTorch tensors** (`src/ids/charging_ids.py`): added `hasattr(value, "flatten") and hasattr(value, "tolist")` branch with `.detach().cpu().flatten().tolist()`; without this, Krum and cosine similarity always received empty vectors and were functionally disabled.
- **`PrivacyAuditor._flatten_model_update()` handles tensors** (`src/auditor/privacy_auditor.py`): same fix as above, required for the audit of delta-weight dicts.
- **`FedAvgAggregator._weighted_average()` recomputes denominator** (`src/ml/fedavg_aggregator.py`): `total_samples` now computed from compatible updates only (same weight-vector length as the first update); updates with incompatible shapes are logged and excluded before the denominator is summed, preventing under-weighted aggregation.
- **`PrivacyAuditor` budget formula corrected** (`src/auditor/privacy_auditor.py`): `round_epsilon = (sensitivity / max_grad_norm) × ε_target` (Gaussian Mechanism approximation); total budget = `ε_target × total_rounds_budget` (basic composition, Dwork 2014); `PRIVACY_BUDGET_EXHAUSTED` alert now fires at `budget_ratio ≥ 1.0` instead of `cumulative ≥ ε_target` — eliminates false positives from round 2 onward.
- **Krum enabled with `byzantine_tolerance=0`** (`scripts/run_experiments.py`): with 4 clusters, `n=4 ≥ 2f+3=3` is satisfied only with f=0; documented as geometric outlier detector (not Byzantine-tolerant in Blanchard 2017 sense).
- **Cosine threshold lowered to 0.3** (`scripts/run_experiments.py`): threshold of 0.85 produced systematic false positives on homogeneous FL clusters.
- **Reproducibility seeds** (`scripts/run_experiments.py`, `src/ml/autoencoder_trainer.py`): added `torch.cuda.manual_seed_all`, `cudnn.deterministic=True`, `cudnn.benchmark=False`; `DataLoader` uses `torch.Generator` with `seed + round_num` for deterministic shuffle.
- **`GradientManager` delta validation** (`src/ml/gradient_manager.py`): `delta` range corrected to `(0, 1)` (was `(0, 1.25)`); metadata key renamed `noise_perturbation_applied` (was `dp_applied`) for terminological accuracy.
- **Double risk-score update removed** (`src/ids/charging_ids.py`): `analyze_round()` Step 5 now skips nodes already updated by `analyze()` in Step 1, preventing `_update_risk_score()` from being called twice per round.

Fixes applied during Sprint 5/6 development (pre-sweep):

- **BatchNorm buffers excluded from DP noise** (`src/ml/gradient_manager.py`): `_add_noise()` now skips `running_mean`, `running_var`, and `num_batches_tracked` when `weight_keys` are provided. With σ ≈ 48 at ε = 0.1, adding Gaussian noise to `running_var` (typically 0.1–1.0) rendered it negative, causing `sqrt(running_var + eps) → NaN` in BatchNorm1d eval mode, which propagated through the entire Autoencoder and made all MIA scores NaN. Root-cause fix: `GradientManager._add_noise()` now receives the state_dict key list from `AutoencoderTrainer.get_weight_keys()` and skips BN buffers. Defensive fix: `running_var` is also clamped to ≥ 1e-8 after `load_state_dict` in `run_fedmia()`.
- **`save_results()` called unconditionally**: wrapped `run_fedmia()` and `run_ids()` in independent `try/except` blocks in `main()`; `save_results()` is always reached and FL results are persisted even when MIA or IDS phases raise exceptions.
- **`per_round` union over all result sets**: `save_results()` now iterates `mia_results.keys() | fl_results.keys() | ids_results.keys()`, so FL round data is preserved in the JSON even when MIA produces an empty dict.
- **NaN/Inf filter before `roc_auc_score`**: `run_fedmia()` filters score arrays with `~np.isnan() & ~np.isinf()` before calling `roc_auc_score`; rounds with fewer than 10 valid scores fall back to AUC = 0.5 with a `nan_fraction` field logged.
- **`drop_last=True` in DataLoader** (`src/ml/autoencoder_trainer.py`): guard against empty `batch_losses` list prevents `ZeroDivisionError` on small clusters.
- **`_compute_sigma()` input validation**: enforces `epsilon > 0` and `0 < delta < 1.25`; warning emitted when `delta > 1e-2`.
- **`_parse_record()` error handling**: per-record `try/except` in `load()` and `load_multiple()`; malformed records skipped with warning instead of aborting the full dataset load. `doneChargingTime` parsing isolated with fallback to `disconnectTime`.
- **`PrivacyAuditor.audit()` now active** and receives `epsilon` from experiment config (`PrivacyAuditor(config_path=..., epsilon=cfg["experiment"]["epsilon"])`), overriding the YAML default.
- **Hold-out split**: sessions split 80/20 before `run_fl_rounds()`; hold-out set passed as `non_members` to `run_fedmia()`, ensuring AUC-ROC measures true membership inference.
- **`state_dict` / `load_state_dict`**: `get_weights()` and `set_weights()` use `model.state_dict()` to transfer BatchNorm running statistics alongside trainable parameters; `FedAvgAggregator._weighted_average()` accumulates in `float32` and restores original dtypes.
- **FedAvg loss denominator**: weighted mean loss computed only over nodes with `loss is not None`, using their sample counts as the denominator.
- **`roc_auc_score` guard**: skips AUC computation if either member or non-member score list is empty.
- **Sweep directory isolation** (`scripts/run_experiments.py`, `Makefile`): `--sweep-dir experiments/exp{N}` saves JSON and Excel to a numbered per-sweep directory; `make experiment-full-sweep` auto-increments N on each invocation.

---

## Sprint Roadmap

| Sprint | Status | Deliverables |
|---|---|---|
| Sprint 1 | Complete | Repository scaffold; Containerlab topology definition; Docker images for all 12 nodes; base NVFLARE integration |
| Sprint 2 | Complete | ACN-Data ingestion pipeline; feature extraction (6 features); non-IID cluster partitioning; data validation |
| Sprint 3 | Complete | PyTorch autoencoder (6→16→8→4 encoder + symmetric decoder); local training loop; MSE loss; per-client dataset loaders |
| Sprint 4 | Complete | FedAvg and FedProx aggregation via NVFLARE 2.7.2; proximal_mu configuration; multi-round orchestration |
| Sprint 5 | Complete | Gaussian Mechanism DP integration; gradient clipping; σ calibration; DP accounting (ε, δ tracking per round) |
| Sprint 6 | Complete | FedMIA loss-based evaluator (Yeom 2018); full 20-config sweep (rounds × ε); first results: AUC-ROC ≈ 0.503 all configs — loss-based attack below noise floor; 10 code review fixes applied (HIGH×3, MEDIUM×5, LOW×2) including IDS delta-weights, DP budget formula, FedAvg denominator, Krum config |
| Sprint 7 | In Progress | Calibrated shadow MIA attack (Carlini 2022) implemented: `run_fedmia_shadow()` trains a local shadow model on 50% of training data, computes per-sample calibrated score = MSE(shadow)−MSE(target); shadow `auc_roc` and `score_gap` added to all experiment JSONs; new sweep to validate attack strength |
| Sprint 8 | Planned | FedProx sweep (proximal_mu sweep); statistical analysis of shadow AUC-ROC vs. ε; confidence intervals across 5 seeds; comparison FedAvg vs FedProx; Byzantine injection experiments to validate Krum/cosine IDS |
| Sprint 9 | Planned | Protocol-level experiments: OCPP 1.6 vs. OCPP 2.0.1 vs. MQTT v5 leakage differential; non-IID severity analysis |
| Sprint 10 | Planned | DSN 2027 paper writing; results consolidation; reproducibility packaging; artefact evaluation preparation |

---

## References

1. R. Shokri, M. Stronati, C. Song, and V. Shmatikov, "Membership Inference Attacks Against Machine Learning Models," in *Proceedings of the 2017 IEEE Symposium on Security and Privacy (S&P)*, pp. 3–18, 2017. https://doi.org/10.1109/SP.2017.41

1a. S. Yeom, I. Giacomelli, M. Fredrikson, and S. Jha, "Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting," in *Proceedings of the 31st IEEE Computer Security Foundations Symposium (CSF)*, pp. 268–282, 2018. https://doi.org/10.1109/CSF.2018.00027

2. N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis, and F. Tramèr, "Membership Inference Attacks From First Principles," in *Proceedings of the 2022 IEEE Symposium on Security and Privacy (S&P)*, pp. 1897–1914, 2022. https://doi.org/10.1109/SP46214.2022.9833649

3. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, "Communication-Efficient Learning of Deep Networks from Decentralized Data," in *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 54, pp. 1273–1282, 2017.

4. T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Smola, and V. Smith, "Federated Optimization in Heterogeneous Networks," in *Proceedings of Machine Learning and Systems (MLSys)*, vol. 2, pp. 429–450, 2020.

5. M. Nasr, R. Shokri, and A. Houmansadr, "Comprehensive Privacy Analysis of Deep Learning: Passive and Active White-box Inference Attacks against Centralized and Federated Learning," in *Proceedings of the 2019 IEEE Symposium on Security and Privacy (S&P)*, pp. 739–753, 2019. https://doi.org/10.1109/SP.2019.00065

6. C. Dwork, F. McSherry, K. Nissim, and A. Smith, "Calibrating Noise to Sensitivity in Private Data Analysis," in *Theory of Cryptography Conference (TCC)*, LNCS 3876, pp. 265–284, 2006. https://doi.org/10.1007/11681878_14

7. M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, and L. Zhang, "Deep Learning with Differential Privacy," in *Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (CCS)*, pp. 308–318, 2016. https://doi.org/10.1145/2976749.2978318

8. E. Bagdasaryan, A. Veit, Y. Hua, D. Estrin, and V. Shmatikov, "How To Backdoor Federated Learning," in *Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 108, pp. 2938–2948, 2020.

9. P. Blanchard, E. M. El Mhamdi, R. Guerraoui, and J. Stainer, "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.

10. Z. J. Lee, D. Chang, Z. Hu, G. S. Taylor, and S. H. Low, "ACN-Data: Analysis and Applications of an Open EV Charging Dataset," in *Proceedings of the 10th ACM International Conference on Future Energy Systems (e-Energy)*, pp. 139–149, 2019. https://doi.org/10.1145/3307772.3328313

11. S. Truong, K. Sun, S. Moran, and P. Phung, "Privacy Preservation in Federated Learning: An Insightful Survey from the GDPR Perspective," *Computers & Security*, vol. 110, 2021. https://doi.org/10.1016/j.cose.2021.102402

12. L. Melis, C. Song, E. De Cristofaro, and V. Shmatikov, "Exploiting Unintended Feature Leakage in Collaborative Learning," in *Proceedings of the 2019 IEEE Symposium on Security and Privacy (S&P)*, pp. 691–706, 2019. https://doi.org/10.1109/SP.2019.00029

13. R. Bassily, A. Smith, and A. Thakurta, "Private Empirical Risk Minimization: Efficient Algorithms and Tight Error Bounds," in *Proceedings of the 55th Annual IEEE Symposium on Foundations of Computer Science (FOCS)*, pp. 464–473, 2014.

14. European Parliament and of the Council, "Regulation (EU) 2023/1804 on the deployment of alternative fuels infrastructure (AFIR)," *Official Journal of the European Union*, L 234, pp. 1–65, 2023.

15. Z. Wang, M. Song, Z. Zhang, Y. Song, Q. Wang, and H. Qi, "Beyond Inferring Class Representatives: User-Level Privacy Leakage From Federated Learning," in *Proceedings of IEEE INFOCOM 2019*, pp. 2512–2520, 2019. https://doi.org/10.1109/INFOCOM.2019.8737416

16. R. C. Geyer, T. Klein, and M. Nabi, "Differentially Private Federated Learning: A Client Level Perspective," *Workshop on Machine Learning in Private Setting*, NeurIPS 2017. https://arxiv.org/abs/1712.07557  
    *Foundational reference for the weight perturbation approach used in ChargeShield-FL: instead of per-sample gradient clipping (DP-SGD), Gaussian noise is added to the aggregated weight vector before upload, providing client-level DP. Distinguishes this approach from sample-level guarantees of Abadi et al. (ref 7).*

17. K. Wei, J. Li, M. Ding, C. Ma, H. H. Yang, F. Farokhi, S. Jin, T. Q. S. Quek, and H. V. Poor, "Federated Learning with Differential Privacy: Algorithms and Performance Analysis," *IEEE Transactions on Information Forensics and Security*, vol. 15, pp. 3454–3467, 2020. https://doi.org/10.1109/TIFS.2020.2988575  
    *Comprehensive analysis of DP mechanisms in FL, covering weight perturbation vs. gradient perturbation trade-offs, sigma calibration under composition across rounds, and convergence bounds under non-IID data — directly applicable to ChargeShield-FL's experimental setup.*

---

## License

MIT License

Copyright (c) 2026 ChargeShield-FL Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
