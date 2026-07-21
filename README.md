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
| FedMIA Evaluator — LiRA ★ PRIMARY (`scripts/run_experiments.py`) | LiRA (Carlini et al. 2022, IEEE S&P) | **Server-side attack**: intercepts `raw_updates` (local client weights) BEFORE FedProx aggregation and DP noise. Trains n_shadow local shadow models on random 50% subsets of training sessions. Score = log P(loss\|IN) − log P(loss\|OUT) via Gaussian log-likelihood ratio. Stronger than Yeom/Shadow because FedProx averaging destroys per-cluster memorisation in the global model. JSON output includes `lira_auc_roc`, `lira_member_score_mean`, `lira_non_member_score_mean`, `lira_score_gap` per round |
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
| Membership Inference on training sessions | Honest-but-curious FL server; external adversary with model access | Gaussian Mechanism DP (ε ∈ {0.1–5.0}, δ = 1e-5) | Three-tier attack hierarchy: (1) **Yeom 2018** (baseline, loss-based, global model); (2) **Shadow MIA** (calibrated, global model, Carlini 2022); (3) **LiRA ★ PRIMARY** (server-side, `raw_updates` PRE-aggregation, Carlini 2022). LiRA is the primary attack for DSN 2027: intercepts per-client local weights before FedProx merging — FedProx averaging destroys cluster-level memorisation in the global model, making LiRA on raw_updates the only reliable signal. Score = log P(loss\|IN) − log P(loss\|OUT) via Gaussian log-LR over n_shadow shadow models. AUC > 0.55 no-DP + AUC ≈ 0.50 with DP → DP suppresses LiRA → claim validated |
| Gradient Inversion (reconstruction of raw session data) | Active server-side attacker | DP noise injection; mTLS transport integrity | Reconstruction MSE on held-out sessions |
| Byzantine update poisoning | Malicious FL client submitting corrupted updates | Krum aggregation filter; Cosine Similarity anomaly detection | Attack detection rate; model accuracy degradation |
| Distributional shift / concept drift exploitation | Compromised client inflating local loss | CUSUM sequential monitoring | False positive rate; detection latency in rounds |
| Network-level eavesdropping | Passive adversary on inter-node links | WireGuard VPN tunnels; mTLS certificate pinning | N/A (eliminated by design) |
| Model extraction via repeated query | Black-box query adversary | Rate limiting (not yet implemented; Sprint 7 target) | Query efficiency bound |

---

## Dataset

**Source:** ACN-Data, Adaptive Charging Network, Caltech / JPL Campus  
**URL:** https://ev.caltech.edu/dataset  
**Primary training coverage:** 2019 and 2020 calendar years (default config)  
**Cross-dataset validation coverage:** 2018 and 2021 (available for reproducibility, enabled via `experiment.yaml`)  
**Sessions:** 13,073 real EV charging sessions (2019+2020); 2018 and 2021 datasets available for cross-year reproducibility validation  
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

> **Note (Sprint 9, 2026-07-16):** All results from exp1–exp6 have been invalidated and deleted. They were obtained with: (a) `epochs=3` (Scenario B — model does not memorise), (b) no LiRA attack, (c) IDS false positives that skewed the results. New experiments must be run with `epochs=50` and the Sprint 9 codebase. The sections below document the pre-Sprint-9 baseline for historical context only.

> **Note (2026-07-21):** `experiments/exp1`, `exp2`, `exp3` (no-DP multi-seed sweep attempts, seed=42/123/456) have been **invalidated and deleted**: `exp1` was interrupted before saving any JSON; `exp2` hit the IDS Long-tensor crash on 2/5 seeds (fixed same day, commit `301413f`) and was interrupted on the 3rd; `exp3` (seed=42 only, post cross-cluster-eval fix) completed but exposed a second, deeper bug — LiRA shadow models were trained on cross-cluster mixed samples while the attacked model (client raw update) is a per-cluster specialist, producing systematically inverted LiRA AUC (0.14–0.32 instead of ≈0.5). Fixed by training one shadow ensemble **per cluster** (see Engineering Fixes below). Re-run the full sweep only after re-validating with `make experiment-smoke`.

### Historical Baseline: Pre-Sprint-9 (epochs=3, no LiRA) — INVALIDATED

The first completed experiment ran 100 federated rounds at ε = 1.0 with `epochs=3`. Results are kept here only to document the Scenario B discovery.

| Parameter | Value |
|---|---|
| FL algorithm | FedProx (proximal_mu = 0.01) |
| Rounds | 100 |
| Local epochs | 3 (too few — model does not memorise) |
| Privacy budget ε | 1.0 |
| Yeom mean AUC-ROC | 0.5030 |
| Shadow mean AUC-ROC | 0.4970 |
| LiRA | Not implemented yet |
| Interpretation | **Scenario B** — model does not memorise with 15 total training epochs; DP has nothing to suppress |

**Scenario B explanation:** With only 3 local epochs × 5 rounds = 15 total training epochs per cluster, the autoencoder (570 parameters) learned a general EV pattern but not individual session details. Reconstruction error was nearly identical for members and non-members. Scenario B is not a valid test of DP effectiveness — it must be ruled out first via the no-DP baseline with `epochs=50`.

### Sprint 9 Target: Full Parameter Sweep (epochs=50, Yeom+Shadow+LiRA)

**Hypothesis:** With `epochs=50`, LiRA (raw_updates, server-side) should achieve AUC > 0.55 without DP, and AUC ≈ 0.50 with DP. This is the DSN 2027 core claim.

**Experimental sequence:**
1. `make experiment-smoke` — pipeline OK (5 round, no-DP, n_shadow=4)
2. `make experiment-nodp` — Scenario A confirmation (LiRA AUC > 0.55?)
3. `make experiment-dp EPS=1.0` — DP suppresses LiRA? (AUC → 0.50?)
4. `make experiment-full-sweep` — full rounds × ε grid for paper figures
5. Enable `jpl_2018` and `jpl_2021` in config → cross-dataset reproducibility

A systematic sweep across the following parameter grid is planned:

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

# ── Sequenza raccomandata Sprint 9 ────────────────────────────────────────────
# Ogni run esegue SEMPRE Yeom + Shadow + LiRA insieme — stesso JSON, stesse condizioni.

# Passo 1: verifica pipeline (5 round, no-DP, n_shadow=4, ~5 min CPU)
make experiment-smoke

# Passo 2: baseline no-DP (10 round, n_shadow=8) — LiRA AUC > 0.55?
make experiment-nodp

# Passo 3: con DP (10 round, ε=1.0) — LiRA AUC → 0.50?
make experiment-dp

# Override epsilon o n_shadow:
make experiment-dp EPS=0.5
make experiment-dp EPS=0.1 N_SHADOW=16

# Full sweep: rounds × ε (20 configs), tutti gli attacchi
caffeinate -s nohup make experiment-full-sweep > /tmp/sweep.log 2>&1 &
tail -f /tmp/sweep.log

# IDS validation (Byzantine sweep): 5 seeds × 5 ε = 25 run
caffeinate -s nohup make experiment-byzantine-sweep > /tmp/byz_sweep.log 2>&1 &
tail -f /tmp/byz_sweep.log
```

#### CLI arguments

`scripts/run_experiments.py` supports the following arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--config` | path | `config/experiment.yaml` | Config file path |
| `--epsilon` | float | from config | Override DP budget ε |
| `--rounds` | int | from config | Override FL rounds |
| `--seed` | int | from config (42) | Reproducibility seed — affects session shuffle, DataLoader, model init. Use values 42, 123, 456, 789, 1234 for multi-seed validation |
| `--byzantine` | flag | false | Enable Byzantine gradient scaling attack |
| `--byzantine-node` | str | from config (highway) | Attacking cluster name |
| `--scale-factor` | float | from config (10.0) | Attack intensity multiplier |
| `--sweep-dir` | path | None | Output directory for sweep isolation (e.g. `experiments/exp2`) |
| `--dry-run` | flag | false | Validate config and dataset; no training |
| `--no-dp` | flag | false | Disable DP (σ=0, no clipping) — baseline for Scenario A/B disambiguation |
| `--n-shadow` | int | from config (8) | Number of shadow models for LiRA (4=smoke, 8=default, 16=paper quality, 32+=high confidence) |
| `--skip-ids` | flag | false | Skip IDS analysis (faster, for MIA-only evaluation) |

**Example: manual multi-seed run**

```bash
# Baseline: 5 seeds, ε=1.0, 100 rounds (sequential — do NOT run in parallel)
for seed in 42 123 456 789 1234; do
  python3 scripts/run_experiments.py \
    --epsilon 1.0 --rounds 100 --seed $seed \
    --sweep-dir experiments/exp_baseline
done

# Byzantine: same seeds, with attack enabled
for seed in 42 123 456 789 1234; do
  python3 scripts/run_experiments.py \
    --epsilon 1.0 --rounds 100 --seed $seed \
    --byzantine --byzantine-node highway --scale-factor 10 \
    --sweep-dir experiments/exp_byzantine
done
```

> **Note on parallelism**: run experiments **sequentially**, not in parallel. Two concurrent FL experiments contend for CPU/memory and produce slower, less reproducible results on Apple Silicon. The Makefile targets are sequential by design.

> **Note on seed choice**: any integer works. The values 42, 123, 456, 789, 1234 are conventional and already embedded in the Makefile sweep targets. 5 seeds is the minimum for reporting `mean ± std`; 10 seeds for Wilcoxon significance testing.

Results are written to a directory prefixed by sweep type — `experiments/full-sweep{N}/`, `experiments/nodp-sweep{N}/`, `experiments/dp-sweep{N}/` (fix 2026-07-21: previously all three shared one generic `exp{N}` counter, which made it impossible to tell no-DP from DP results apart by name alone):
- `experiment_{timestamp}.json` — per-config result (one file per completed config, timestamped)
- `{dirname}.xlsx` — Excel report (Raw Data, Heat Map, Per Rounds, Per Epsilon, Comparison, AUC Progression, + per-attack sheets), regenerated after each config completes, named after its directory

Running `make experiment-full-sweep` a second time produces `experiments/full-sweep2/` and so on; sweep directories are never mixed, and each sweep type has its own independent counter.

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
| `make experiment` | Single experiment with default parameters (100 rounds, ε=1.0, FedProx μ=0.01) |
| `make experiment-full-sweep` | Full rounds × ε sweep (20 configs); creates `experiments/exp{N}/`; auto-increments N |
| `make experiment-byzantine-sweep` | Sprint 8: Byzantine gradient scaling sweep — 5 seeds × 5 ε (25 runs); highway cluster ×10; validates IDS Krum + cosine detection |
| `make test` | Run the complete unit and integration test suite (pytest) |
| `make test-sprint4` | Sprint 4 tests only |
| `make test-sprint5` | Sprint 5 tests only |
| `make lint` | Static analysis via ruff |
| `make clean` | Remove `__pycache__`, `.pyc` files, and pytest artefacts |
| `make build` | Build Docker images for the containerised topology (not required for experiments) |
| `make provision` | Provision Containerlab topology and WireGuard mesh (container topology only) |
| `make deploy` | Start NVFLARE server and all 12 FL clients in containers (container topology only) |
| `make destroy` | Tear down the Containerlab topology |

### Experiment Result Scripts

| Script | Description |
|---|---|
| `scripts/run_experiments.py` | Orchestrates a single FL experiment; runs two parallel MIA evaluators per round — loss-based (Yeom 2018, `auc_roc`) and calibrated shadow attack (Carlini 2022, `shadow_auc_roc`); performs 80/20 hold-out split; writes per-round MIA scores, FL `mean_loss`, and IDS delta-weight analysis to `experiment_{timestamp}.json`; accepts `--sweep-dir experiments/exp{N}`; auto-regenerates `exp{N}.xlsx` at completion |
| `scripts/run_sweep.py` | Runs multiple experiments sequentially for a given `--rounds` and `--epsilon` grid; auto-detects next `exp{N}` directory if `--sweep-dir` not provided; logs progress per config; invokes `run_experiments.py` as a subprocess |
| `scripts/generate_excel_report.py` | Standalone tool: reads all `experiment_*.json` files from a given directory and generates a 10-sheet Excel workbook: **Raw Data** (one row per experiment, Yeom+Shadow+LiRA columns), **Heat Map** (AUC-ROC matrix: rounds × ε), **Per Rounds** (aggregated by round count), **Per Epsilon** (aggregated by ε), **Comparison** (side-by-side metrics), **AUC Progression** (Yeom per-round trajectory), **Attack Comparison** (Yeom vs Shadow vs LiRA synthetic, with Δ column), **Yeom Per Round** (round-by-round AUC per experiment), **Shadow Per Round** (idem, Shadow MIA), **LiRA Per Round** (idem, LiRA — ★ PRIMARY) |

### Engineering Fixes

Fixes applied 2026-07-21 (LiRA shadow/target mismatch, three rounds — post-Sprint-9 code review):

- **(a) LiRA shadow models trained cross-cluster, target model is per-cluster** (`scripts/run_experiments.py`, `run_lira()`): shadow ensembles were trained by sampling randomly from the *entire* `train_sessions` (all 4 clusters mixed), while the attacked model is each client's own update, trained only on its own cluster's ~2600 sessions. Fixed by training one full shadow ensemble **per cluster**, sampled only from that cluster's own index range. Validated empirically on `nodp-sweep1`/`dp-sweep1` (5 seeds each): round 1 improved from 0.558 to 0.75, but rounds 2–10 stayed broken and flat (~0.26) — fix (a) alone was insufficient, see (b).
- **(b) Shadow/target TRAINING PROCEDURE mismatch**: from round 2 onward, a real client does *not* train from scratch — it starts from the previous round's shared global weights and does only `local_epochs` (50) more epochs. The shadow ensemble, however, was trained once, from random init, for a fixed 250-epoch budget — a different trajectory entirely. This is why round 1 (no shared init yet) looked healthy after fix (a) while rounds 2+ did not. Fixed by retraining every shadow **each round**, warm-started from `fl_results[round-1]["global_weights"]` (round 1 = random init, matching real clients), for exactly `local_epochs` epochs — mirroring the real per-round procedure. Shadow training compute roughly doubles (retrained every round instead of once) but each run trains for fewer epochs (50 vs 250) each time.
- **(c) DP noise never reached what LiRA attacked**: LiRA attacked `raw_updates`, captured in `run_fl_rounds()` *before* `gm.privatize()` is called; since `privatize()` always returns a new object, `raw_updates` never carries DP noise regardless of `--no-dp`/`--epsilon`. Verified: `lira_auc_roc` at round 1 was bit-for-bit identical between `nodp-sweep1` and `dp-sweep1` for matching seeds (e.g. seed=42 → 0.750451 in both). By construction, DP could never suppress LiRA — defeating the purpose of the no-DP vs DP comparison. Fixed: LiRA now attacks `updates` (the actual per-client update submitted for aggregation, post-privatize when DP is enabled), and shadow models are privatised with the same clip+noise procedure when DP is on, so target and shadow are calibrated under the same noise regime.
- **Conceptual finding, not yet acted on (terminology corrected 2026-07-21)**: the DP mechanism here (`gm.privatize()` called by the server, per client, immediately upon receiving that client's raw update, before aggregation) is architecturally **DP-FedAvg** [McMahan et al., 2017] — server-side, per-client clip+noise — not **local DP** (client noises on its own device, server never sees the raw value) and not textbook **central DP** (server receives all raw updates, adds one noise draw to the aggregate). "Distributed DP" was the wrong label for this: that term refers to cryptographic protocols (secure aggregation with secret-shared noise) where no party ever sees another's raw contribution, which is not what's implemented here. Under true central DP, individual client updates are *never* noised, so LiRA-style attacks on individual updates would show *zero* suppression at any ε — not a bug, but the standard argument for why central DP needs a trusted aggregator and why an untrusted one needs secure aggregation or local DP. See `docs/CaseStudies.md` §2.4.3 for the full three-way distinction and a proposed DP-FedAvg-vs-central-DP comparison experiment (CS4 candidate, not yet implemented).
- **Excel "Seed Aggregation" / `save_results()` "LOW risk" bucket doesn't distinguish "DP works" from "attack is broken"**: `privacy_risk` and the sheet's colour-coding classify any AUC ≤ 0.52 as green/"LOW risk", including the 0.14–0.32 (and now potentially lower, pre-fix-b/c) values above. Still open — add a sanity check/warning band for AUC significantly below 0.5 (e.g. < 0.4) distinct from the "AUC ≈ 0.5, DP is working" case.
- **Status**: none of (a)/(b)/(c) has been re-validated end-to-end — torch is unavailable in the review sandbox (network blocks `download.pytorch.org`, plain PyPI torch too large to fetch there). Re-run `make experiment-smoke` (and ideally re-run `nodp-sweep1`/`dp-sweep1` from scratch — the existing ones predate fixes b/c and must be treated as invalid) before trusting new LiRA numbers.

Fixes applied during Sprint 9 (LiRA + IDS false-positive elimination):

- **GRADIENT_EXPLOSION systematic false positives with 50 local epochs** (`scripts/run_experiments.py`): With 50 epochs per round, accumulated weight deltas always exceeded the absolute adaptive threshold (`max_grad_norm + 3σ ≈ 15.5`). Fixed with **peer-relative normalisation**: compute L2-norm of each client's delta; scale by `max_grad_norm / median_norm` before passing to the auditor. Guard: if `median_norm < 1e-4` (round 1, sparse deltas) use `scale=1.0` to avoid `inf` scaling. Median uses lower-middle index `(len-1)//2` to avoid inflating the reference when a Byzantine outlier is present. Krum analysis continues on non-normalised deltas (geometric, scale-independent).
- **Krum false positives with 50 local epochs** (`scripts/run_experiments.py`): With 50 epochs, legitimate cluster divergence produces Krum scores up to 3.27. Threshold raised from 1.5 to **3.5**. Validated: Byzantine score ≈ 4.0, max legitimate FP = 3.27 → gap = 0.73.
- **Budget exhausted alert with `--no-dp`** (`scripts/run_experiments.py`): `PrivacyAuditor` tracked `epsilon=cfg["epsilon"]` even when DP was disabled. Fixed by passing `epsilon=1000.0` when `no_dp=True` → budget ratio stays near 0 for any realistic number of rounds.
- **Three per-attack Excel sheets added** (`scripts/generate_excel_report.py`): `build_yeom_per_round()`, `build_shadow_per_round()`, `build_lira_per_round()` — each shows round-by-round AUC for one attack across all experiments, color-coded by risk level. Added via shared `_build_per_round_sheet()` helper. `main()` and `_update_excel_report()` both create all 10 sheets.
- **Excel merge crash on `build_raw_data()`** (`scripts/generate_excel_report.py`): old `ws.merge_cells("A1:K1")` left from 10-column layout conflicted with new `ws.merge_cells("A1:O1")` (15 columns). openpyxl 3.x raises `ValueError` when merging a range that includes already-merged cells. Fixed by removing the duplicate old call.

Fixes applied during Sprint 8 (post-exp1 verification):

- **IDS receives pre-DP raw weights** (`scripts/run_experiments.py`): `run_fl_rounds()` now stores `raw_updates` (pre-`gm.privatize()`) and `raw_global_weights` (FedAvg of raw, noise-free) alongside the privatised updates. `run_ids()` uses raw weights to compute deltas. Root cause: with ε=0.1, σ ≈ 48×`max_grad_norm`; post-DP weight L2-norm ≈ 1157 >> 1.0, causing `GRADIENT_EXPLOSION` and `BUDGET_EXHAUSTED` on every node every round. A "round 0" entry with the initial model weights eliminates the round-1 false positive. In a real FL deployment the server-side IDS sees raw client updates before DP noise is applied — this fix aligns the simulation with the real threat model.
- **Krum normalisation fixed** (`src/ids/charging_ids.py`): `KrumDetector.compute_scores()` now normalises by `mean(scores)` instead of `max(scores)`. With max-normalisation, all legitimate nodes had score ≈ 1.0 (equidistant) → all above threshold 0.8 → 400 false positives per 100-round experiment. With mean-normalisation: legitimate ≈ 1.0, Byzantine (×10 update) ≈ 2.7 → threshold 1.5 discriminates correctly.
- **Krum threshold raised to 1.5** (`scripts/run_experiments.py`): `ChargingIDS` now instantiated with `krum_threshold=1.5` (was 0.8 default). Validated by simulation: legitimate max score ≈ 1.01, Byzantine score ≈ 2.7 across 100 simulated rounds.
- **GRADIENT_EXPLOSION adaptive threshold** (`src/auditor/privacy_auditor.py`): `_detect_threats()` now uses `threshold = max_grad_norm + 3×σ` (Gaussian mechanism 3-sigma rule) instead of the fixed `max_grad_norm×10`. σ = `max_grad_norm × √(2ln(1.25/δ)) / ε`; examples at max_grad_norm=1.0: ε=5.0 → threshold≈3.9 (stringent), ε=1.0 → threshold≈15.5, ε=0.1 → threshold≈146 (permissive). Threshold is computed in `__init__()` and exposed as `explosion_threshold` in every `AuditReport.metadata` for inspection. With the raw_updates fix, legitimate raw deltas ≈ 0.036 are far below any reasonable threshold; GRADIENT_EXPLOSION can still fire when the raw_global baseline is contaminated by a Byzantine node (documented as expected behaviour — Krum is the primary Byzantine detector). Three new unit tests added (`test_gradient_explosion_threshold_adapts_to_epsilon`, `test_metadata_contains_explosion_threshold`).

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

## Experiment Types and Directory Structure

ChargeShield-FL uses two completely separate experiment tracks. **Never mix results between them.**

### Track 1 — Privacy Risk (MIA) Experiments

These are the primary experiments for DSN 2027. All runs are **clean**: no Byzantine attack active (`byzantine_attack.enabled: false`). Results go in `experiments/nodp-sweep{N}/`, `experiments/dp-sweep{N}/`, or `experiments/full-sweep{N}/` depending on target (auto-incremented per type, see note above).

| Target | Purpose |
|---|---|
| `make experiment-smoke` | Pipeline verification (5 round, no-DP, n_shadow=4, ~5 min) |
| `make experiment-nodp` | No-DP baseline (10 round, n_shadow=8) — Yeom+Shadow+LiRA; LiRA AUC>0.55 → Scenario A |
| `make experiment-dp` | With DP (10 round, ε=EPS, n_shadow=8) — Yeom+Shadow+LiRA; compare with no-DP |
| `make experiment-full-sweep` | Full sweep: rounds × ε (20 configs), all attacks → paper figures |
| `make experiment` | Single run with default params (100 rounds, ε=1.0) |

### Track 2 — IDS Validation Experiments

These runs have **Byzantine attack active** and measure IDS detection capability. Results go in `experiments/ids_validation/` — a fixed, dedicated directory that is never mixed into the `exp{N}` sequence.

| Target | Purpose |
|---|---|
| `make experiment-byzantine-sweep` | 5 seed × 5 ε (25 runs) — validates Krum detection |

The code enforces this separation: `--byzantine` and `--no-dp` cannot be used together (hard error); running with `--byzantine` without `--sweep-dir` emits a warning.

### IDS Validation Results

IDS correctness is validated via a controlled Byzantine gradient scaling attack (Sprint 8). The highway cluster multiplies its local model weights by a scale factor of 10 before sending them to the aggregator. This creates a geometrically anomalous update that Krum — the primary Byzantine detector — should identify.

**Result (5-round validation, seed=42, ε=1.0):** Krum score for the highway-01 Byzantine node ≈ 4.0; Krum scores for all three legitimate nodes (urban-01, residential-01, corporate-01) ≈ 1.0. With threshold 1.5, the IDS correctly issues a CRITICAL alert on highway-01 at every round with zero false positives. A `GRADIENT_EXPLOSION` alert fires on all nodes in Byzantine rounds — expected behaviour: the contaminated raw global weights shift the delta baseline for legitimate nodes, causing sensitivity to briefly exceed `max_grad_norm + 3σ`. This motivates Krum as the primary Byzantine detector over simple threshold-based alarms.

**IDS validation scope:** IDS validation is secondary to the privacy risk claim in DSN 2027. It appears as a subsection demonstrating that the system can detect gradient-based attacks while remaining unaffected by them in the clean MIA sweep.

---

## No-DP Baseline Experiment

### Motivation

The primary experimental claim of ChargeShield-FL is that Differential Privacy (specifically, the Gaussian Mechanism applied via weight perturbation) suppresses membership inference signals in federated EV charging models. This claim is operationalised as: *AUC-ROC of FedMIA attacks drops toward 0.5 (random-guess baseline) as ε decreases.*

However, AUC ≈ 0.5 is ambiguous without a control condition. It may arise from two distinct scenarios:

**Scenario A — DP works as intended.** The model learns a non-trivial representation of the training data (members have lower reconstruction error than non-members), but the Gaussian noise added per round obscures this signal. Without noise, a MIA attacker could recover meaningful membership information (AUC > 0.5); with noise, the signal is suppressed (AUC → 0.5). This is the desired outcome: DP provides genuine privacy protection.

**Scenario B — The model does not memorise.** The autoencoder generalises sufficiently well that reconstruction error is nearly identical for members and non-members regardless of DP. Even without any privacy noise (σ = 0), AUC ≈ 0.5. This can happen when the model is too small relative to the dataset size, when training converges to a flat minimum, or when the data distribution is too homogeneous. In this scenario, DP is providing no additional protection beyond what the model's natural generalisation already provides — the claim that DP *causes* AUC ≈ 0.5 would be incorrect.

Without a no-DP control, these two scenarios are indistinguishable from AUC results alone.

### Methodology

The no-DP baseline runs the identical FL pipeline (same dataset split, same seed, same model architecture, same FedAvg aggregation) with σ = 0: the `GradientManager.privatize()` step is bypassed and raw local weights are sent directly to the aggregator. Gradient clipping (`max_grad_norm`) is also skipped, removing the only source of regularisation introduced by the DP mechanism. All other hyperparameters (lr, epochs, batch size, proximal_mu) are unchanged.

The comparison is:

| Condition | σ | Expected AUC if Scenario A | Expected AUC if Scenario B |
|---|---|---|---|
| No-DP baseline | 0 | **> 0.55** (MIA signal visible) | ≈ 0.5 (model does not memorise) |
| DP ε = 5.0 | ≈ 0.97 | ≈ 0.5 (noise suppresses signal) | ≈ 0.5 |
| DP ε = 1.0 | ≈ 4.84 | ≈ 0.5 | ≈ 0.5 |
| DP ε = 0.1 | ≈ 48.4 | ≈ 0.5 | ≈ 0.5 |

A no-DP AUC significantly above 0.5 confirms Scenario A and validates the DP claim. A no-DP AUC ≈ 0.5 indicates Scenario B and requires either a larger/more expressive model, a richer feature set, or additional training rounds before the DP sweep is meaningful.

### First Run Result (epochs=3, 5 rounds) — Scenario B

The first no-DP baseline run with the original configuration (3 local epochs, 5 FL rounds = 15 total training epochs per cluster) produced:

```
Shadow MIA AUC rounds 1–5: 0.495, 0.495, 0.496, 0.496, 0.496
Mean AUC-ROC: 0.4970 — Privacy risk: LOW
```

**Interpretation — Scenario B confirmed:** AUC ≈ 0.5 even without DP. The model never memorised the training data. With only 15 total training epochs on ~2,615 sessions per cluster, the autoencoder (570 parameters) learned a general EV charging pattern but not individual session details. Reconstruction error was nearly identical for members and non-members. DP is providing no measurable privacy protection because there is no membership signal to suppress.

### Fix: Increasing Local Epochs (3 → 50)

To establish Scenario A — a prerequisite for demonstrating that DP suppresses MIA — local training epochs are increased from 3 to 50. This gives each FL round 50 gradient epochs on the local dataset, producing 10 × 50 = 500 total training epochs per cluster in the no-DP validation run. At this training depth, the autoencoder is expected to overfit: training reconstruction error drops significantly below hold-out error, creating a measurable membership signal (AUC > 0.5 without DP).

Only after confirming AUC > 0.5 without DP does the full DP sweep become scientifically meaningful. The shadow model cap (`shadow_epochs = min(50 × fl_rounds, 500)`) is unchanged.

### Running the Experiment

```bash
# No-DP baseline with corrected epochs (10 rounds × 50 epochs = 500 total per cluster)
python3 scripts/run_experiments.py --config config/experiment.yaml --no-dp --rounds 10

# Or via Makefile
make experiment-nodp
```

The JSON output is written to `experiments/` with `"name": "..._nodp_baseline"` and `"no_dp": true` in the config section, making it unambiguously distinguishable from DP runs.

### Implications for DSN 2027

If Scenario A is confirmed, the paper can make the strong claim: *"The Gaussian Mechanism at ε = 1.0 reduces FedMIA AUC from X to 0.50, demonstrating that DP suppresses membership signals in EV FL models."* The ε vs AUC curve becomes the central figure.

If Scenario B is confirmed, the claim must be reframed: the current autoencoder architecture (570 parameters, 6 input features) does not memorise individual sessions. The paper should either (a) increase model expressiveness (larger hidden layers, more features), or (b) reframe the contribution as a negative result — demonstrating that standard EV charging autoencoders are naturally resistant to MIA, with DP providing defence-in-depth against stronger future attacks.

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
| Sprint 7 | Complete | Calibrated shadow MIA attack (Carlini 2022): `run_fedmia_shadow()` computes per-sample calibrated score = MSE(shadow)−MSE(target); shadow AUC-ROC ≈ 0.499 across all ε; IDS false-positive bugs fixed (raw_updates, Krum normalisation) |
| Sprint 8 | Complete | Byzantine gradient scaling IDS validation: highway ×10 → Krum score 4.0 vs 1.0 legitimate; zero false positives; GRADIENT_EXPLOSION adaptive threshold (max_grad_norm + 3σ); `--seed`, `--byzantine`, `--scale-factor`, `--no-dp` CLI args; exp1 regenerated with corrected code |
| Sprint 9 | Complete | **LiRA attack** (Carlini et al. 2022): server-side MIA on `raw_updates` PRE-FedProx aggregation — strongest attack; **attack hierarchy** Yeom (weak) → Shadow (medium) → LiRA (★ primary); **IDS fixes**: peer-relative normalisation for GRADIENT_EXPLOSION FPs, Krum threshold 3.5 for 50-epoch training, DP budget guard for no-DP runs; **Excel**: 10-sheet report (Attack Comparison + 3 per-attack sheets); **epochs 3→50** forces memorisation (Scenario A); **datasets 2018+2021** added for cross-dataset reproducibility validation; `--n-shadow` CLI arg; all old results (exp1–exp6, pre-LiRA) invalidated and deleted; Makefile unified: `experiment-smoke`, `experiment-nodp`, `experiment-dp` each run Yeom+Shadow+LiRA together |
| Sprint 9e | Complete (code) | **LiRA shadow/target mismatch, three rounds of fixes**: (a) per-cluster shadow ensembles (was cross-cluster), (b) per-round warm-started shadow retraining matching the real client procedure (was one-shot 250-epoch training from scratch), (c) LiRA now attacks post-DP `updates` instead of pre-DP `raw_updates`, with shadows privatised to match. Also surfaced a conceptual finding: the DP mechanism is **DP-FedAvg** (server-side, per-client), not local DP and not central DP — see `docs/CaseStudies.md` §2.4.3 for the full distinction. `make experiment-smoke` re-run 2026-07-21 confirms the pipeline completes without errors post-fix; `nodp-sweep1`/`dp-sweep1` predate fixes (b)/(c) and must be re-run from scratch before trusting the numbers. |
| Sprint 9f | Audit only | **Containerlab/NVFLARE infra gap found**: `scripts/run_experiments.py` (source of every number in this README/CaseStudies.md) never touches the Containerlab/Docker/NVFLARE infra also in this repo — it's a single-process simulation of exactly 4 clients, not the "100+ nodes" claimed in `ChargeShield-FL_Decisions.txt`. `src/flare/flare_connector.py` is still the Sprint-3 placeholder (never imports `nvflare`, simulates gradients with `random.gauss()`); no NVFLARE job/app exists; `docker/*/Dockerfile` are orphaned with broken `CMD`s; two PKI trees were never reconciled. Wiring the real privacy pipeline into the containerised topology is a multi-week effort (real NVFLARE Executor, per-client raw-update extraction for LiRA/Shadow, Dockerfile fixes, PKI reconciliation) — see `docs/CaseStudies.md` §2.4.3 for full detail. Decision pending: pursue this, or document current 4-client single-process simulation as an explicit paper limitation. |
| Sprint 10 | Planned | **Cross-dataset validation**: enable jpl_2018 and jpl_2021 in config; run `experiment-nodp` and `experiment-dp` on all four datasets; verify LiRA AUC reproducibility across years |
| Sprint 11 | Planned | **Gradient inversion attack** (after LiRA validated): reconstruct raw EV session features from weight updates; measures reconstruction quality on held-out sessions |
| Sprint 12 | Planned | **Interactive demo GUI** (Streamlit): real-time FL training visualisation, per-round AUC curve for all three attacks, IDS alert timeline, DP noise/utility tradeoff slider; artefact for DSN 2027 evaluation |
| Sprint 13 | Planned | DSN 2027 paper writing; results consolidation; reproducibility packaging; artefact evaluation preparation |

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
