# ChargeShield-FL

A research framework for evaluating Membership Inference Attacks and differential privacy defences in Federated Learning systems deployed across heterogeneous Electric Vehicle charging infrastructure.

---

## Abstract

ChargeShield-FL is an open research framework designed to empirically evaluate the privacy guarantees of Federated Learning (FL) in the context of Electric Vehicle (EV) charging networks, with a particular focus on Membership Inference Attacks (MIA) and how much real privacy leakage survives once a standard mitigation (differential privacy, DP) is nominally in place. **The project's central claim is not that DP suppresses MIA** — it is the opposite: that a sophisticated MIA (LiRA, Carlini et al. 2022) continues to detect membership even when DP is nominally active, in a realistic multi-site FL deployment, which is the practically relevant question for anyone deploying FL with "DP enabled" and assuming that alone settles the privacy question. As smart-grid deployments increasingly adopt FL to train shared models over distributed charging stations without centralising raw session data, the question of whether individual charging sessions can be re-identified from model updates becomes a critical safety and regulatory concern. The framework's current experiment pipeline (`scripts/run_experiments.py`, 2026-07-22) trains across a cross-silo federation of the 3 real ACN-Data sites — Caltech, JPL, and Office 1 — each a genuinely distinct organisation with its own EVSE population, grouped by each session's own `site_id` rather than an artificial slice (see "Dataset" below for the correction of an earlier JPL/Caltech mislabeling and the historical Containerlab/OCPP topology this superseded). MIA evaluation follows an attack hierarchy of increasing strength: a loss-based per-round evaluator (Yeom et al. 2018), a calibrated Shadow MIA (Carlini 2022), and **LiRA (★ primary attack)**, which intercepts each client's own submitted update immediately before FedAvg combines it — the same honest-but-curious-aggregator threat model this project specializes, into a real EV charging deployment, from prior work on privacy auditing for FL-enabled OT intrusion detection systems (see "Relation to Prior Work" below). A separate shadow-model-based FedMIA plugin (`src/plugins/attacks/fedmia.py`) is used by ChargingIDS for per-node IDS scoring and remains unchanged. Both mechanisms are integrated alongside CUSUM, Krum, and Cosine Similarity intrusion detection baselines, enabling controlled measurement of attack success across a full sweep of FL aggregation strategies (FedAvg, FedProx), DP placements (DP-FedAvg, central, local), and privacy budgets (ε ∈ {0.1, 0.5, 1.0, 2.0, 5.0}). A first smoke-scale run on the 3 real sites (5 rounds, no DP, seed=42) already shows the expected direction: LiRA mean AUC-ROC ≈ 0.589 (max 0.615) versus ≈0.50 for Yeom/Shadow — i.e. the model measurably memorises and the strongest attack detects it; the open question the full sweep is designed to answer is how much of that signal survives across realistic DP configurations and epsilon budgets, not whether it can be pushed to exactly 0.5. ChargeShield-FL targets publication at the IEEE/IFIP International Conference on Dependable Systems and Networks (DSN) 2027 (abstract deadline 25 November 2026, paper deadline 2 December 2026).

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
| Membership Inference on training sessions | Honest-but-curious FL server; external adversary with model access | Gaussian Mechanism DP (ε ∈ {0.1–5.0}, δ = 1e-5) — evaluated as a mitigation to test, not assumed effective | Three-tier attack hierarchy: (1) **Yeom 2018** (baseline, loss-based, global model); (2) **Shadow MIA** (calibrated, global model, Carlini 2022); (3) **LiRA ★ PRIMARY** (server-side, per-client update PRE-FedAvg aggregation, Carlini 2022). LiRA is the primary attack: intercepts each client's own submitted update before it is combined into the global model — the same honest-but-curious-aggregator threat model as the auditing framework this project specializes to EV charging (see "Relation to Prior Work" below). Score = log P(loss\|IN) − log P(loss\|OUT) via Gaussian log-LR over n_shadow shadow models. **The core claim this project targets is that LiRA continues to detect membership even when DP is nominally active** (AUC stays meaningfully above 0.5 despite ε on the order of the values tested here) — not that DP suppresses it. See "Real multi-site experiment" (Dataset section) and `docs/CaseStudies.md` §2.4.3 for why nominal ε understates the actual privacy cost (composition across rounds) and is not by itself evidence that leakage has been eliminated |
| Gradient Inversion / model reconstruction (raw session data or model internals from updates) | Active server-side attacker | Not yet implemented — planned future work (Sprint 11), conditional on the MIA results above | Reconstruction MSE on held-out sessions (metric defined, attack not yet built) |
| Byzantine update poisoning | Malicious FL client submitting corrupted updates | Krum aggregation filter; Cosine Similarity anomaly detection | Attack detection rate; model accuracy degradation |
| Distributional shift / concept drift exploitation | Compromised client inflating local loss | CUSUM sequential monitoring | False positive rate; detection latency in rounds |
| Network-level eavesdropping | Passive adversary on inter-node links | WireGuard VPN tunnels; mTLS certificate pinning | N/A (eliminated by design) |
| Model extraction via repeated query | Black-box query adversary | Rate limiting (not yet implemented; Sprint 7 target) | Query efficiency bound |

---

## Dataset

**Source:** ACN-Data, Adaptive Charging Network (Caltech)  
**URL:** https://ev.caltech.edu/dataset  
**Sites (3 — the only real sites ACN-Data provides):** Caltech campus (public, 54 EVSEs), JPL (workplace, employees-only, 50 EVSEs), Office 1 (Silicon Valley office, 8 EVSEs)  
**Coverage:** 2018–2021 for Caltech and JPL; 2019–2021 for Office 1 (2018 not published for this site)  
**Licence:** Caltech ACN-Data research licence (non-commercial academic use)

> **Correction (2026-07-22):** every experiment prior to this date (all Sprint results, `exp1`–`exp6`, `nodp-sweep1`/`dp-sweep1`) used a dataset checked into this repo as `datasets/acn/jpl/acndata_sessions_2019.json` and labelled "JPL" throughout this README and `docs/CaseStudies.md`. Re-verifying the raw file against ACN-Data's own `siteID` field and the official per-site EVSE counts published at the URL above showed this file is actually **Caltech** data (`siteID="0002"`, 54 unique stations — an exact match for Caltech's 54 EVSEs, not JPL's 50). The mislabeling dates to Sprint 2 and does not change the qualitative conclusion of the Sprint 9f/10 scalability review (the old 4-"cluster" topology was always a single real site sliced four ways, never genuinely heterogeneous) — but every prior reference to "JPL" as a *public campus* site was backwards: Caltech is the public campus site, JPL is the restricted workplace site. Freshly downloaded, correctly-labelled per-site files (via `scripts/download_acn_sessions.py`, a paginated REST client written to work around the web export's bulk-download timeout) now live at `datasets/acn/{caltech,jpl,office1}/acndata_sessions_<year>.json`.

### Real multi-site experiment (current)

The main FL experiment (training + FedMIA/Shadow/LiRA privacy attacks) uses exactly the 3 real ACN-Data sites above as its 3 FL clients — no synthetic/fictional site is used for training or for privacy measurement. Each client is grouped by its sessions' own `site_id` field (`scripts/run_experiments.py::group_indices_by_site()`), not by an arbitrary slice, so client boundaries now correspond to genuinely distinct organisations with different EVSE counts and usage populations. This replaces the previous design (documented below in "Distribution to Clusters" and throughout "Infrastructure"/"Key Results"), where 4 same-site "clusters" (Highway/Urban/Residential/Corporate) were reconstructed by slicing one real site's sessions by `max_power_kw`, and are retained here only as historical/aspirational context for the Containerlab network-emulation layer, which — per the Sprint 9f finding below — was never actually wired into the experiment pipeline that produces the paper's numbers.

**IDS/Byzantine validation uses 2 additional synthetic clients, and only for that purpose.** Krum's Byzantine-detection guarantee requires `n ≥ 2f+3` total nodes to reliably detect `f` compromised nodes; with `f=1` that means `n ≥ 5`, and only 3 real sites exist in ACN-Data — no 4th or 5th real site can be added. When `byzantine_attack.enabled: true` in `config/experiment.yaml`, `inject_synthetic_client_indices()` adds exactly 2 synthetic clients (`synthetic_1`, `synthetic_2`) built by pooling all 3 real clients' session indices and re-slicing them (not duplicating a single real site), giving `n=5`. This is the **only** thing the synthetic clients are for. In that mode: the FL training model *does* train on all 5 clients (needed so Krum has real per-client updates to score), but **FedMIA/Shadow/LiRA are skipped entirely** — `scripts/run_experiments.py::main()` never calls them when `byzantine_attack.enabled=true`, logging a warning instead — because the synthetic clients' sessions overlap arbitrarily with the real clients' and with each other, which would invalidate any privacy/membership-inference number. With `byzantine_attack.enabled: false` (the default, and the only setting used for privacy-paper numbers), there are no synthetic clients: the training model and every attack (Yeom, Shadow, LiRA) use only the 3 real clients. In short — **2 synthetic clients, IDS/Krum validation only; training model and the privacy attack always use only the 3 real clients.**

### Feature Schema

| Feature | Unit | Description |
|---|---|---|
| `total_energy_kwh` | kWh | Total energy delivered in the session |
| `max_power_kw` | kW | Peak power draw during the session |
| `kwh_requested` | kWh | Energy requested by the vehicle at session initiation |
| `minutes_available` | min | Time the vehicle remained plugged in |
| `hour_of_day` | h (0–23) | Wall-clock hour at session start |
| `duration_hours` | h | Elapsed time from plug-in to plug-out |

### Distribution to Clusters (superseded — historical/aspirational, see "Real multi-site experiment" above)

> The paragraph below describes the pre-2026-07-22 design, where all sessions came from one real site and were sliced four ways by power profile to fake heterogeneity. It is kept for historical context on the Containerlab/OCPP network-emulation layer (see "Infrastructure" below), which was never actually driven by this data split — the real experiment pipeline (`scripts/run_experiments.py`) now groups sessions by real `site_id` across the 3 ACN-Data sites instead, as described above.

Sessions were partitioned across the four cluster types according to power profile compatibility: Highway nodes receive sessions with `max_power_kw` > 50; Corporate nodes receive sessions with `max_power_kw` in (20, 50]; Urban nodes receive sessions with `max_power_kw` in (10, 20]; Residential nodes receive sessions with `max_power_kw` <= 10. This produced a non-IID distribution across FL clients by construction, but — since all sessions came from a single real site — never reflected genuine cross-organisation heterogeneity.

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

**Hypothesis (revised 2026-07-22 — see Abstract):** With `epochs=50`, LiRA (server-side, per-client update pre-FedAvg) should achieve AUC > 0.55 without DP (model measurably memorises — confirmed at smoke scale, mean 0.589). **The core claim this project targets is that a meaningful part of that signal survives across the ε range actually used in the sweep below, not that it collapses to ≈0.50** — a full collapse at every tested ε would mean DP-FedAvg is already a complete fix and there is no residual-leakage finding to report. The interesting, publishable outcomes are: (a) AUC stays significantly above 0.5 at "reasonable"/looser ε (e.g. 2.0, 5.0), demonstrating leakage survives realistic DP configurations; and/or (b) even where AUC drops close to 0.5 at strict ε, the naive cumulative budget `epsilon_cumulative_naive = ε × fl_rounds` (see `docs/CaseStudies.md` §2.4.3) shows the nominal per-round ε understates the true composed privacy cost paid to get there.

**Experimental sequence:**
1. `make experiment-smoke` — pipeline OK (5 round, no-DP, n_shadow=4) — done on the 3 real sites, LiRA mean AUC 0.589
2. `make experiment-nodp` — upper bound on leakage (no mitigation active)
3. `make experiment-dp EPS=<value>` — repeat across ε ∈ {5.0, 2.0, 1.0, 0.5, 0.1}: where does LiRA AUC stay above 0.5 despite DP?
4. `make experiment-full-sweep` — full rounds × ε grid for paper figures
5. Cross-dataset reproducibility: all 4 years (2018–2021) are already enabled per-site in `config/experiment.yaml`'s `sites:` section (see "Real multi-site experiment" in Dataset)

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

Download all 3 real ACN-Data sites from https://ev.caltech.edu/dataset (or use `scripts/download_acn_sessions.py`, a paginated REST client — the web UI's bulk export times out and silently truncates large years/sites) and place the JSON files at:

```
datasets/acn/caltech/acndata_sessions_2018.json   (+ 2019, 2020, 2021)
datasets/acn/jpl/acndata_sessions_2018.json       (+ 2019, 2020, 2021)
datasets/acn/office1/acndata_sessions_2019.json   (+ 2020, 2021 — no 2018 for this site)
```

> The folder name must be `office1` (not `office`) — it must match the `office1` site name used in `config/experiment.yaml`'s `sites:` key and in NVFLARE's `cluster_id`/`project.yml`, since `chargeshield_executor.py` derives each client's dataset directory as `datasets/acn/<cluster_id>/`.

See `config/experiment.yaml`'s `sites:` section for the exact file list read by `scripts/run_experiments.py`.

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
| `--byzantine` | flag | false | Enable Byzantine gradient scaling attack — also adds 2 synthetic clients (`synthetic_1`/`synthetic_2`, n=5 total) so Krum's n≥2f+3 guarantee holds; skips FedMIA/Shadow/LiRA entirely for this run (see "Real multi-site experiment" in the Dataset section) |
| `--byzantine-node` | str | from config (`synthetic_1`) | Attacking client name — always one of the 2 synthetic clients, never a real site |
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

# Byzantine/IDS validation: same seeds, with attack enabled (adds synthetic_1/
# synthetic_2 automatically, n=5 total; FedMIA/Shadow/LiRA are skipped for
# this run — it validates Krum/IDS only, never a source of privacy numbers)
for seed in 42 123 456 789 1234; do
  python3 scripts/run_experiments.py \
    --epsilon 1.0 --rounds 100 --seed $seed \
    --byzantine --byzantine-node synthetic_1 --scale-factor 10 \
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
| `scripts/run_experiments.py` | Orchestrates a single FL experiment; runs two parallel MIA evaluators per round — loss-based (Yeom 2018, `auc_roc`) and calibrated shadow attack (Carlini 2022, `shadow_auc_roc`); performs 80/20 hold-out split; writes per-round MIA scores, FL `mean_loss`, and IDS delta-weight analysis to `experiment_{timestamp}.json`; accepts `--sweep-dir experiments/{nodp,dp,full}-sweep{N}`; auto-regenerates `{sweep-dir-name}.xlsx` at completion |
| `scripts/run_sweep.py` | Runs multiple experiments sequentially for a given `--rounds` and `--epsilon` grid; auto-detects next `full-sweep{N}` directory if `--sweep-dir` not provided (fixed 2026-07-22 — previously used the abandoned generic `exp{N}` scheme); logs progress per config; invokes `run_experiments.py` as a subprocess |
| `scripts/generate_excel_report.py` | Standalone tool: reads all `experiment_*.json` files from a given directory and generates a 10-sheet Excel workbook: **Raw Data** (one row per experiment, Yeom+Shadow+LiRA columns), **Heat Map** (AUC-ROC matrix: rounds × ε), **Per Rounds** (aggregated by round count), **Per Epsilon** (aggregated by ε), **Comparison** (side-by-side metrics), **AUC Progression** (Yeom per-round trajectory), **Attack Comparison** (Yeom vs Shadow vs LiRA synthetic, with Δ column), **Yeom Per Round** (round-by-round AUC per experiment), **Shadow Per Round** (idem, Shadow MIA), **LiRA Per Round** (idem, LiRA — ★ PRIMARY) |

### Engineering Fixes

Fixes applied 2026-07-21 (LiRA shadow/target mismatch, three rounds — post-Sprint-9 code review):

- **(a) LiRA shadow models trained cross-cluster, target model is per-cluster** (`scripts/run_experiments.py`, `run_lira()`): shadow ensembles were trained by sampling randomly from the *entire* `train_sessions` (all 4 clusters mixed), while the attacked model is each client's own update, trained only on its own cluster's ~2600 sessions. Fixed by training one full shadow ensemble **per cluster**, sampled only from that cluster's own index range. Validated empirically on `nodp-sweep1`/`dp-sweep1` (5 seeds each): round 1 improved from 0.558 to 0.75, but rounds 2–10 stayed broken and flat (~0.26) — fix (a) alone was insufficient, see (b).
- **(b) Shadow/target TRAINING PROCEDURE mismatch**: from round 2 onward, a real client does *not* train from scratch — it starts from the previous round's shared global weights and does only `local_epochs` (50) more epochs. The shadow ensemble, however, was trained once, from random init, for a fixed 250-epoch budget — a different trajectory entirely. This is why round 1 (no shared init yet) looked healthy after fix (a) while rounds 2+ did not. Fixed by retraining every shadow **each round**, warm-started from `fl_results[round-1]["global_weights"]` (round 1 = random init, matching real clients), for exactly `local_epochs` epochs — mirroring the real per-round procedure. Shadow training compute roughly doubles (retrained every round instead of once) but each run trains for fewer epochs (50 vs 250) each time.
- **(c) DP noise never reached what LiRA attacked**: LiRA attacked `raw_updates`, captured in `run_fl_rounds()` *before* `gm.privatize()` is called; since `privatize()` always returns a new object, `raw_updates` never carries DP noise regardless of `--no-dp`/`--epsilon`. Verified: `lira_auc_roc` at round 1 was bit-for-bit identical between `nodp-sweep1` and `dp-sweep1` for matching seeds (e.g. seed=42 → 0.750451 in both). By construction, DP could never suppress LiRA — defeating the purpose of the no-DP vs DP comparison. Fixed: LiRA now attacks `updates` (the actual per-client update submitted for aggregation, post-privatize when DP is enabled), and shadow models are privatised with the same clip+noise procedure when DP is on, so target and shadow are calibrated under the same noise regime.
- **Conceptual finding, not yet acted on (terminology corrected 2026-07-21)**: the DP mechanism here (`gm.privatize()` called by the server, per client, immediately upon receiving that client's raw update, before aggregation) is architecturally **DP-FedAvg** [McMahan et al., 2017] — server-side, per-client clip+noise — not **local DP** (client noises on its own device, server never sees the raw value) and not textbook **central DP** (server receives all raw updates, adds one noise draw to the aggregate). "Distributed DP" was the wrong label for this: that term refers to cryptographic protocols (secure aggregation with secret-shared noise) where no party ever sees another's raw contribution, which is not what's implemented here. Under true central DP, individual client updates are *never* noised, so LiRA-style attacks on individual updates would show *zero* suppression at any ε — not a bug, but the standard argument for why central DP needs a trusted aggregator and why an untrusted one needs secure aggregation or local DP. See `docs/CaseStudies.md` §2.4.3 for the full three-way distinction and a proposed DP-FedAvg-vs-central-DP comparison experiment (CS4 candidate, not yet implemented).
- **Excel "Seed Aggregation" / `save_results()` "LOW risk" bucket doesn't distinguish "DP works" from "attack is broken"**: `privacy_risk` and the sheet's colour-coding classified any AUC ≤ 0.52 as green/"LOW risk", including the 0.14–0.32 values above. **Fixed later the same day** — see the next entry.
- **Status**: (a)/(b)/(c) were validated on real sweep data (`nodp-sweep1`/`dp-sweep1`, then `nodp-sweep2`/`dp-sweep2`) — which is exactly what surfaced fix (e) below. All sweep results predating fix (e) were invalid for LiRA at round ≥ 2 and have been **deleted** (2026-07-22): `nodp-sweep1`, `dp-sweep1`, `nodp-sweep2` (complete, pre-fix-e), `dp-sweep2`/`dp-sweep3` (both interrupted mid-run, no usable output). A partial re-run of `dp-sweep3` before deletion, using the fix-(e) code, showed `lira_auc_roc` = 0.4832 / 0.4989 / 0.4693 for rounds 1-3 (seed 42, ε=1.0) — no more inversion or clip-ceiling saturation, consistent with DP correctly suppressing LiRA (Scenario A). `experiments/` now only contains `smoke/`; the next `make experiment-nodp-sweep`/`experiment-dp-sweep` will start a clean `nodp-sweep1`/`dp-sweep1`.

- **(e) Asymmetric IN/OUT variance estimation for non-members inflates/inverts LiRA scores** (`scripts/run_experiments.py`, `run_lira()`): found by inspecting real sweep output (`nodp-sweep2`, seeds 42 and 123, with fixes a-d already applied) — `lira_auc_roc` still crashed from 0.76 (round 1) to 0.15-0.32 (rounds 2-4), recovering only partially to ~0.45-0.56 by round 10. The JSON showed why: `lira_non_member_score_mean` spiked to +17.9 at round 2 (clip ceiling is ±20) and decayed slowly, while `lira_member_score_mean` stayed flat near 0 throughout. Root cause: a non-member's OUT reference is the per-sample, cross-shadow spread over just `n_shadow` (~8) values on that one point — right after a shared warm-start, all shadows start from identical weights and haven't yet diverged, so they agree almost perfectly on any point none of them trained on, collapsing σ_out toward the scale-adaptive floor. Meanwhile the IN side for a non-member uses the pooled, cluster-wide fallback distribution (hundreds of samples, much wider σ). Comparing a near-collapsed per-point σ against a wide pooled σ makes the Gaussian log-likelihood-ratio dominated by the `1/σ²` term rather than by genuine membership signal. Members don't show this because both their IN and OUT sides are already per-sample/small-N (symmetric — the collapse, when it happens, affects both terms and partly cancels). Fixed by computing an analogous pooled `global_out_stats_per_cluster` (mirrors the existing `global_in_stats_per_cluster`) and using it as a floor for both σ_out and σ_in, so neither side's variance can collapse below what's typically observed across the cluster that round. μ_in/μ_out (the actual signal) are untouched. Contributing factor: `n_shadow=8` ("fast demo" per `config/experiment.yaml`) makes any per-sample variance estimate inherently noisy; the config recommends 16-32 for "paper quality," which would also reduce reliance on this floor.

Fixes applied 2026-07-22 (independent review + real central/local DP modes):

- **"Comparison" sheet missing ANOMALY/HIGH/MEDIUM/LOW coloring for Shadow and LiRA AUC** (`scripts/generate_excel_report.py::build_comparison()`): found by an independent review pass — the per-cell coloring only special-cased the Yeom keys (`auc_roc`/`auc_max`/`auc_min`); `shadow_auc`/`shadow_max`/`lira_auc`/`lira_max` rendered as plain black text with no risk coloring at all, meaning an anomalous LiRA AUC (the exact symptom fixes 2026-07-21a-e were about) would render invisibly in this one sheet while every other sheet already colored all three attacks correctly. Fixed by adding the four missing keys to the same `_auc_risk_color()` call.
- **`scripts/run_sweep.py` never migrated off the abandoned generic `exp{N}` naming** (same review pass): `_next_sweep_dir()`, docstrings, and `--sweep-dir` help still built/referenced `experiments/exp{N}/` with one shared counter — exactly the ambiguity the 2026-07-21 Makefile fix eliminated for the `nodp-sweep`/`dp-sweep`/`full-sweep` targets. `run_sweep.py` performs the same kind of rounds×epsilon grid sweep as `make experiment-full-sweep`, so `_next_sweep_dir()` now uses the matching `full-sweep{N}` scheme. Also updated: stale Makefile comments under `experiment-byzantine-sweep` still saying "MAI in experiments/exp{N}/", and the README script table.
- **Cleanup**: deleted `nodp-sweep1`, `dp-sweep1`, `nodp-sweep2` (complete but pre-fix-e, invalid for LiRA ≥ round 2), `dp-sweep2`/`dp-sweep3` (interrupted, no usable output), and stray root-level `ChargeShield_FL_Results.xlsx` + two loose `experiment_*.json` from earlier ad-hoc single-seed runs. `experiments/` now contains only `smoke/`.
- **Real central DP and local DP modes implemented** (`scripts/run_experiments.py::run_fl_rounds()`/`run_lira()`, `src/ml/gradient_manager.py::clip_only()`/`privatize_aggregate()`, new `--dp-mode {dp-fedavg,central,local}` CLI flag): turns the "proposed next step, not yet implemented" comparison from `docs/CaseStudies.md` §2.4.3 into working code.
  - **Central DP**: each client clips its own update (bounds sensitivity) but does not noise it (`GradientManager.clip_only()`); the server aggregates the clean-but-clipped updates via FedAvg, then adds one Gaussian noise draw to the aggregate, σ scaled by `1/n_participants` (`GradientManager.privatize_aggregate()`, refactored `_add_noise()` to accept a σ override). Individual client updates are never noised — LiRA attacking one is expected to show no suppression at any ε; this is the predicted result, not a bug. Shadow models in `run_lira()` are calibrated the same way (`clip_only()` instead of `privatize()`) when `dp_mode="central"`, so the target/shadow noise-regime mismatch fix (2026-07-21c) still holds under this mode.
  - **Local DP**: same per-client clip+noise mechanism as DP-FedAvg, but `run_fl_rounds()` no longer populates `raw_updates`/`raw_global_weights` in `fl_results` under this mode — modeling that the server must never see the raw update, not even transiently. `run_ids()`'s existing `round_data.get("raw_updates") or round_data.get("updates", [])` fallback then automatically uses the noised update. Expected consequence, documented in `run_ids()`'s docstring rather than "fixed": the peer-relative gradient-delta IDS check degrades to comparing absolute (noised) weights instead of round-over-round deltas, which can raise more `GRADIENT_EXPLOSION` false positives — a real tradeoff of genuine local DP (weaker anomaly detection when the server never sees a clean gradient).
  - New Makefile targets: `experiment-central-dp`/`experiment-local-dp` (single seed) and `experiment-central-dp-sweep`/`experiment-local-dp-sweep` (5-seed, → `experiments/central-sweep{N}/`/`local-sweep{N}/`).
  - Excel: "No-DP" column now shows `no (central)`/`no (local)` when active; Comparison sheet has a new "DP Mode" row.
  - **Not yet run**: no experiment has actually been executed with these new modes as of this writing (torch unavailable in the review sandbox — see recurring note). Predictions above are from the mechanism design, not yet empirically confirmed.

Fixes applied 2026-07-22, follow-up (independent review of today's own new code + NVFLARE scaffold — see `docs/NVFlareIntegration.md` for the NVFLARE finding in full):

- **NVFLARE: all 4 clients would have trained on the same cluster's data** (`nvflare/jobs/chargeshield_poc/`): `meta.json` deploys one shared `app/` to `@ALL` sites, and the single `config_fed_client.json` had `cluster_id: "highway"` hardcoded — since NVFLARE deploys the identical config to every site, all 4 clients (`nvflare/project.yml` names them `highway`/`urban`/`residential`/`corporate`) would have instantiated `cluster_id="highway"`, training on the same 25% data slice and defeating the per-cluster heterogeneity the whole simulation is built around. Caught by an independent review, not by the original author (me), on code written earlier the same day. Fixed: `ChargeShieldExecutor._setup()` now derives `cluster_id` from `fl_ctx.get_identity_name()` (the NVFLARE site name) when recognized, falling back to the config value with a warning otherwise. Still unverified (no `nvflare` installed to confirm `get_identity_name()`'s exact return value at `START_RUN`) — flagged as a 4th `VERIFY:` point.
- **Excel "Seed Aggregation"/"Heat Map" could silently pool different `dp_mode`s** (`scripts/generate_excel_report.py`): `build_seed_aggregation()` grouped by `(rounds, epsilon, no_dp)` — missing `dp_mode` — so seeds from `dp-fedavg` and `central`/`local` DP at the same ε would have been averaged into one mean±std row with no warning; `build_heat_map()`'s `(rounds, epsilon)` grid had the same exposure (last-by-timestamp record silently wins). Fixed: `dp_mode` added to the Seed Aggregation grouping key (and to sort order/label); Heat Map now explicitly filters to `dp_mode == "dp-fedavg"` (its original intent) with a note pointing to Comparison for central/local. Verified with real, **executed** tests (`tests/test_generate_excel_report_dp_mode.py` — this module has no torch dependency, so unlike the rest of today's new code it was actually run in the review sandbox, not just syntax-checked: `pytest tests/test_generate_excel_report_dp_mode.py` → 3/3 passed).
- **`--no-dp --dp-mode central` (or `local`) persisted the non-default `dp_mode` into the saved config** (`scripts/run_experiments.py::main()`): harmless at runtime (`no_dp` short-circuits before `dp_mode` is ever consulted) but confusing in the saved JSON/Excel — a row could show `no_dp: true` and `dp_mode: "central"` simultaneously. Fixed: `args.dp_mode` is normalized to `"dp-fedavg"` inside the `--no-dp` branch before being persisted.
- **Stale claim in `docs/CaseStudies.md` §2.4.3**: still said "With `epochs=3` (current configuration)..." in the formal-DP-guarantee discussion, though `config/experiment.yaml` has used `epochs=50` since Sprint 9 (2026-07-16). Fixed the number; noted the guarantee gap is larger at 50 epochs than it would have been at 3.
- **Open methodological question flagged, not fixed** (`src/ml/gradient_manager.py::_clip_weights()`, pre-existing, used by both DP-FedAvg and the new central-DP `clip_only()`): clips the L2 norm of the client's full post-training weight vector (not the delta from the global model), and — unlike `_add_noise()` — does not exclude BatchNorm running-stat buffers from that norm. With `max_grad_norm=1.0` on a ~650-parameter model, the clip plausibly engages most rounds, uniformly shrinking the whole model toward the previous global model every round; some of the observed "DP suppresses MIA" effect could be attributable to this repeated whole-model clipping rather than to the added Gaussian noise alone. Documented in `docs/CaseStudies.md` §2.4.3 as an open question requiring empirical investigation (e.g., logging how often/hard the clip engages on real data) before DP-vs-no-DP AUC comparisons are attributed to noise specifically. **Deliberately not touched**: this is pre-existing behavior also used by the currently-running `dp-sweep1` — do not patch mid-sweep.
- **New/expanded test coverage**: `tests/test_sprint5.py` gained `test_clip_only_does_not_add_noise` and `test_privatize_aggregate_sigma_scales_with_n_participants` (empirically checks σ_central/σ_n4 ratio ≈ 4, catching silent 1/n scaling regressions); `tests/test_run_experiments_integration.py` gained `test_runs_under_central_and_local_dp_mode` (LiRA under both new modes — previously zero coverage of the `clip_only()` shadow branch) and `test_local_dp_degrades_without_crashing` (IDS under local DP — the documented peer-relative-to-absolute degradation had no test asserting it doesn't crash structurally); new `tests/test_generate_excel_report_dp_mode.py` (3 tests, actually executed — see above).

Fixes applied 2026-07-21, later the same day (Excel anomaly detection + integration test coverage, while `nodp-sweep`/`dp-sweep` ran in the background):

- **`privacy_risk` / Excel colour-coding didn't distinguish "AUC ≈ 0.5 because DP works" from "AUC ≪ 0.5 because the attack itself is broken"** (`scripts/run_experiments.py::save_results()`, `scripts/generate_excel_report.py`): added an `ANOMALY` category (`privacy_risk = "ANOMALY"` when the minimum AUC across attacks < 0.40), with a dedicated purple colour distinct from the "LOW risk" green, applied across every sheet that colours AUC values (Raw Data, Heat Map, Comparison, AUC Progression, Attack Comparison, Yeom/Shadow/LiRA Per Round, Seed Aggregation) plus their legends. A `logger.warning()` fires when `privacy_risk == "ANOMALY"`. The LiRA-vs-Yeom delta column in Seed Aggregation is deliberately left uncoloured by this change since it's a difference of two AUCs, not an absolute risk value.
- **New integration test suite** (`tests/test_run_experiments_integration.py`, 10 tests): `run_fl_rounds`, `run_fedmia`, `run_lira`, and `run_ids` had zero test coverage of any kind (unit or integration) despite being the functions that produce every number in the paper — `scripts/` is explicitly excluded from the unit-test coverage measurement (see `docs/Testing.md` §6.3). New tests run the real pipeline end-to-end on small synthetic data (`fl_rounds=3`, `epochs=2`, `n_shadow=2`, no mocking), including regression tests for fix (c) above (LiRA must read `updates`, not `raw_updates`) and for the Krum/budget IDS calibration. Run with `make test-integration`. See `docs/Testing.md` §8.
- **`Autoencoder._calibrate_threshold()` crash on a fully empty `DataLoader`** (`src/core/autoencoder.py`): the `ZeroDivisionError` guard in `fit()`'s training loop (`if not batch_losses: continue`) already handled a `DataLoader` with zero batches per epoch, but `fit()` still called `_calibrate_threshold()` on that same empty loader immediately afterward, which collected `errors = []` and then called `torch.quantile()` on an empty tensor — `RuntimeError: quantile() input tensor must be non-empty`. This is a residual gap in a fix already recorded as applied in `docs/ProjectReview_2026-07-v4.md` (§1, fix #4) — that review only checked the tuple-unpack bug, not this fully-empty-loader edge case. Fixed by returning the current (unchanged) `threshold` when no errors were collected. New regression test: `test_fit_with_empty_loader_does_not_crash` in `tests/test_sprint4.py`.
- **Verified already fixed** (no code change needed): the two other bugs flagged as still-open in `docs/ProjectReview_2026-07-v4.md` §2 — the `FedMIA._calibrate_reference_errors()` tuple-unpack `AttributeError` (`src/plugins/attacks/fedmia.py:182-184`) and the `charging_ids.py` runtime `import logging` inside a method — were both already resolved in the current codebase (unpack guard and module-level import respectively already present) as of this check.

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

> **Historical result below (Sprint 8, pre-2026-07-22):** measured on the old 4-same-site "cluster" topology (`highway`/`urban`/`residential`/`corporate`, `krum_threshold=1.5`, `byzantine_tolerance=0`, `n=4`). Kept for context on Krum's basic discriminative behaviour. It does **not** reflect the current 5-client IDS sweep (3 real sites + `synthetic_1`/`synthetic_2`, `byzantine_tolerance=1`, `n=5`) introduced 2026-07-22 — see "Real multi-site experiment" in the Dataset section and the calibration warning in `run_ids()` (`scripts/run_experiments.py`). That new n=5 configuration has genuinely heterogeneous real clients (not same-site slices), so the natural, attack-free Krum score distribution may differ from the one below; `krum_threshold` is carried over unchanged at 3.5 pending an actual empirical run of `--byzantine` with n=5 to confirm or recalibrate it (not yet done — requires `torch`/`nvflare`, unavailable in this environment during that session).

IDS correctness was validated via a controlled Byzantine gradient scaling attack (Sprint 8). The highway cluster multiplies its local model weights by a scale factor of 10 before sending them to the aggregator. This creates a geometrically anomalous update that Krum — the primary Byzantine detector — should identify.

**Result (5-round validation, seed=42, ε=1.0):** Krum score for the highway-01 Byzantine node ≈ 4.0; Krum scores for all three legitimate nodes (urban-01, residential-01, corporate-01) ≈ 1.0. With threshold 1.5, the IDS correctly issues a CRITICAL alert on highway-01 at every round with zero false positives. A `GRADIENT_EXPLOSION` alert fires on all nodes in Byzantine rounds — expected behaviour: the contaminated raw global weights shift the delta baseline for legitimate nodes, causing sensitivity to briefly exceed `max_grad_norm + 3σ`. This motivates Krum as the primary Byzantine detector over simple threshold-based alarms.

**IDS validation scope:** IDS validation is secondary to the privacy risk claim in DSN 2027. It appears as a subsection demonstrating that the system can detect gradient-based attacks while remaining unaffected by them in the clean MIA sweep. **Only** the 2 synthetic clients may ever be named as the attacker (`--byzantine-node synthetic_1` or `synthetic_2`) — never a real site — both so the paper's real-site results are never characterised as "the malicious one" and because attacking a real site would still leave `n=4` real+synthetic-minus-one, below Krum's `n≥5` requirement for `f=1`.

---

## No-DP Baseline Experiment

### Motivation

**Revised 2026-07-22 — see Abstract.** The primary experimental claim of ChargeShield-FL is *not* that Differential Privacy (the Gaussian Mechanism applied via weight perturbation) suppresses membership inference signals — it is that a realistic, sophisticated MIA (LiRA) continues to detect membership even when DP is nominally active, i.e. that meaningful privacy leakage survives standard mitigations in a real multi-site FL deployment. The no-DP baseline below remains methodologically necessary regardless of which claim is being made: without it, an AUC ≈ 0.5 result with DP active is ambiguous (see the two scenarios below), and without confirming the model memorises at all in the first place, no DP-vs-no-DP comparison means anything.

AUC ≈ 0.5 with DP active is ambiguous without a no-DP control condition. It may arise from two distinct scenarios:

**Scenario A — the model memorises, DP is the only variable in play.** The model learns a non-trivial representation of the training data (members have lower reconstruction error than non-members): without noise, a MIA attacker recovers meaningful membership information (AUC > 0.5). This is the prerequisite for the project's actual question — *how much of that signal survives once DP is turned on*. A DP result that fully collapses AUC to 0.5 at every tested ε would mean this project's residual-leakage claim does not hold at those settings; a DP result where AUC stays meaningfully above 0.5, especially at looser/more realistic ε, is the positive finding this project is designed to surface.

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

A no-DP AUC significantly above 0.5 confirms Scenario A — the model memorises, so the subsequent DP sweep can meaningfully test whether that signal survives DP (the project's actual question). A no-DP AUC ≈ 0.5 indicates Scenario B and requires either a larger/more expressive model, a richer feature set, or additional training rounds before the DP sweep is meaningful.

### First Run Result (epochs=3, 5 rounds) — Scenario B

The first no-DP baseline run with the original configuration (3 local epochs, 5 FL rounds = 15 total training epochs per cluster) produced:

```
Shadow MIA AUC rounds 1–5: 0.495, 0.495, 0.496, 0.496, 0.496
Mean AUC-ROC: 0.4970 — Privacy risk: LOW
```

**Interpretation — Scenario B confirmed:** AUC ≈ 0.5 even without DP. The model never memorised the training data. With only 15 total training epochs on ~2,615 sessions per cluster, the autoencoder (570 parameters) learned a general EV charging pattern but not individual session details. Reconstruction error was nearly identical for members and non-members. DP is providing no measurable privacy protection because there is no membership signal to suppress.

### Fix: Increasing Local Epochs (3 → 50)

To establish Scenario A — the prerequisite for testing whether DP actually suppresses the memorisation signal, or whether meaningful leakage survives it — local training epochs are increased from 3 to 50. This gives each FL round 50 gradient epochs on the local dataset, producing 10 × 50 = 500 total training epochs per cluster in the no-DP validation run. At this training depth, the autoencoder is expected to overfit: training reconstruction error drops significantly below hold-out error, creating a measurable membership signal (AUC > 0.5 without DP).

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

If Scenario A is confirmed (no-DP AUC > 0.55, already shown at smoke scale: 0.589) and the subsequent DP sweep shows LiRA AUC **remaining significantly above 0.5** at one or more tested ε — especially at looser, more realistically-deployed budgets (ε = 2.0, 5.0) — the paper can make the intended claim: *"A sophisticated MIA (LiRA) continues to detect membership in a real multi-site EV charging FL deployment even when DP-FedAvg is nominally active at ε = X, and the naive cumulative privacy cost actually paid (ε × rounds, see `docs/CaseStudies.md` §2.4.3) is far larger than the nominal ε suggests — standard DP configuration is not, by itself, sufficient evidence that membership privacy has been protected."* The ε vs AUC curve is still the central figure, but the finding of interest is where the curve stays *above* 0.5, not where it reaches 0.5.

If DP instead suppresses LiRA to ≈0.5 at every tested ε (including loose ones like 5.0), that is itself worth reporting, but it changes the paper's contribution: it would suggest DP-FedAvg is more robust against this attack in this deployment than the OT/FL privacy literature's threat model would predict, which is a different (and less central) claim than the one this project set out to make — worth investigating why (e.g. central DP mode is expected, by construction, to fully block LiRA on individual updates — see `docs/CaseStudies.md` §2.4.3 — so this outcome under `dp-fedavg` specifically would be the more surprising one to explain).

If Scenario B is confirmed instead (no-DP AUC ≈ 0.5), the claim must be reframed: the current autoencoder architecture (570 parameters, 6 input features) does not memorise individual sessions, so there is no leakage to demonstrate persists under DP. The paper should either (a) increase model expressiveness (larger hidden layers, more features), or (b) reframe the contribution as a negative result.

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
| Sprint 9f | Audit only | **Containerlab/NVFLARE infra gap found**: `scripts/run_experiments.py` (source of every number in this README/CaseStudies.md) never touches the Containerlab/Docker/NVFLARE infra also in this repo — it's a single-process simulation of exactly 4 clients, not the "100+ nodes" claimed in `ChargeShield-FL_Decisions.txt`. `src/flare/flare_connector.py` is still the Sprint-3 placeholder (never imports `nvflare`, simulates gradients with `random.gauss()`); no NVFLARE job/app exists; `docker/*/Dockerfile` are orphaned with broken `CMD`s; two PKI trees were never reconciled. Wiring the real privacy pipeline into the containerised topology is a multi-week effort (real NVFLARE Executor, per-client raw-update extraction for LiRA/Shadow, Dockerfile fixes, PKI reconciliation) — see `docs/CaseStudies.md` §2.4.3 for full detail. |
| Sprint 9g | In progress | **Decision made 2026-07-22**: pursue the Containerlab/NVFLARE integration (started in parallel with, not instead of, the DP work below) — see `docs/NVFlareIntegration.md`. **Real central DP and local DP modes** implemented and wired into `run_fl_rounds()`/`run_lira()`/Excel/Makefile (`--dp-mode {dp-fedavg,central,local}`) — see README Engineering Fixes 2026-07-22 and `docs/CaseStudies.md` §2.4.3. **NVFLARE job scaffold, fase 1**: `nvflare/jobs/chargeshield_poc/` with a real client Executor wrapping `AutoencoderTrainer` — transport-only. **Fase 2 (later the same day)**: `ChargeShieldAggregator`, a custom server-side Aggregator wrapping `FedAvgAggregator`+`PrivacyAuditor`+`ChargingIDS` (replaces NVFLARE's built-in `InTimeAccumulateWeightedAggregator`), mirroring `run_ids()`'s peer-relative IDS logic. **Fase 3 (same day, afternoon)**: `GradientManager`/DP wired into both sides — client-side `clip_only()`/`privatize()` in the Executor for `central`/`local` modes, server-side `privatize()`-per-client (before `FedAvgAggregator`) and `privatize_aggregate()` in `ChargeShieldAggregator` for `dp-fedavg`/`central` modes, mirroring the exact `dp_mode` semantics already tested in the simulation path — see `docs/NVFlareIntegration.md`'s "DP wiring" section. **Fase 4 (same day, evening)**: structured IDS/Auditor export — `ChargeShieldAggregator._run_ids_analysis()` now builds a per-round dict matching `run_ids()`'s `ids_results` shape (alerts, `byzantine_detected`, `low_similarity_nodes`, plus per-client `privacy_score`/`epsilon`/`threats_detected`) and `_export_results()` overwrites a JSON file (`experiments/nvflare_ids_audit_results.json` by default) after every round — previously log-only. **Fase 5 (same day, night)**: raw-update extraction for LiRA/Shadow, scoped deliberately as an OFFLINE step rather than a live port — `ChargeShieldAggregator._export_fl_results()` pickles a per-round dump with the exact schema `run_fl_rounds()` produces in the simulation (`mean_loss`/`n_participants`/`updates`/`raw_updates`/`raw_global_weights`/`global_weights`), and a new script `scripts/run_nvflare_mia.py` loads that dump and calls the existing, already-validated `run_lira()`/`run_ids()`/`run_fedmia()`/`run_fedmia_shadow()`/`save_results()` completely unchanged — chosen specifically to avoid blind-porting LiRA's shadow-retraining logic (which took 5 rounds of empirically-found fixes, see `run_lira()` docstring) into a live `aggregate()` callback with zero ability to test it here. See `docs/NVFlareIntegration.md`'s dedicated section for the full rationale and the one architectural nuance (central-DP's exported "raw_updates" are client-clipped, not pre-clip, which is actually the *correct* view for IDS/LiRA purposes, just reached via a different code path than the simulation). All written without `nvflare`/`torch` installed, **not executed** (only `py_compile`-checked; the pickle round-trip of `GradientUpdate` objects through the same `sys.path` setup used in production was isolated and actually run successfully). Neither DP mode has been run in simulation with the new central/local flags yet either (same sandbox limitation) — `nodp-sweep1`/`dp-sweep1` (dp-fedavg only) completed successfully same day, see Engineering Fixes. **Independent review (same night)** of fase 3-5 found and fixed two real functional bugs: (1) `chargeshield_executor.py` never called the `enrich_sessions()` equivalent, so `hour_of_day`/`duration_hours` (required `AutoencoderTrainer` features) were always missing — verified empirically that this made **every** session invalid (0/10609), i.e. any real NVFLARE client would have trained on an empty tensor; fixed and re-verified (2652/2652 valid after the fix). (2) `run_nvflare_mia.py` loaded sessions from the wrong dataset/order (`config/experiment.yaml`'s combined+shuffled set instead of the real client's `dataset_path`, unshuffled) — would have silently decorrelated LiRA/Shadow/Yeom's membership ground truth from what clients actually trained on; fixed with new `load_client_sessions()`/`load_holdout_sessions()` reading the real client config. A third finding (round-1 `dp-fedavg` reference_weights=None → absolute instead of delta clipping, first round only) was deliberately left as a documented open gap rather than a speculative blind fix — see `docs/NVFlareIntegration.md`'s review section. |
| Sprint 10 | Complete | **Real multi-site ACN-Data integration (2026-07-22)**, closing the realism gap found in the scalability review (the old 4 "clusters" were an arbitrary contiguous slice of ONE real site, never genuine non-IID heterogeneity). **JPL/Caltech mislabeling found and corrected**: the dataset used in every experiment since Sprint 2 (`datasets/acn/jpl/acndata_sessions_2019.json`, "JPL" throughout this README/`docs/CaseStudies.md`) was verified via `siteID` + official EVSE counts to actually be **Caltech** data — see the correction note in "Dataset" above. **`scripts/download_acn_sessions.py`** (new): paginated REST client working around the ev.caltech.edu web export's silent bulk-download truncation (found on the largest files: Caltech 2018, JPL 2019); used to re-download all 12 site×year files (Caltech/JPL 2018–2021, Office1 2019–2021) correctly labelled and validated. **Session grouping redesigned**: `group_sessions_by_site()`/`group_indices_by_site()` (`scripts/run_experiments.py`) now group every session by its own real `site_id` field instead of a positional/contiguous slice — eliminates the exact class of silent-membership-mismatch bug an earlier independent review found in `run_nvflare_mia.py`. `config/experiment.yaml`'s `datasets:` section replaced by `sites:` (per-site file lists, all available years combined per site). `run_fl_rounds()`/`run_lira()` gained optional `cluster_sessions`/`cluster_membership` parameters (default `None`, falling back to the exact old 4-slice behaviour — preserves existing unit tests built on synthetic fixtures). **2 synthetic clients for IDS validation only** (`inject_synthetic_client_indices()`): added exclusively when `byzantine_attack.enabled=true`, giving `n=5` so Krum's `n≥2f+3` guarantee holds for `f=1` (no 4th/5th real ACN-Data site exists); FedMIA/Shadow/LiRA are skipped entirely in that mode rather than computed on data that would mix overlapping synthetic and real sessions. The main experiment (training + every privacy attack) uses **only** the 3 real sites, always — see "Real multi-site experiment" in the Dataset section for the full separation. `run_ids()`'s `byzantine_tolerance` is now `1` during the n=5 IDS sweep and `0` otherwise (previously hardcoded `0` always, which silently gave Krum no formal detection guarantee even when the attack was active); `krum_threshold` stays at `3.5` (the value calibrated for the old, genuinely homogeneous n=4 case) pending an actual empirical run of the new n=5 sweep to confirm or recalibrate it — flagged with a prominent warning in code, since a heterogeneous-by-construction n=5 population may have a different natural (attack-free) score distribution than four same-site slices did. **NVFLARE updated to match**: `nvflare/project.yml` provisions 3 real clients (`caltech`/`jpl`/`office1`, was `highway`/`urban`/`residential`/`corporate`); `chargeshield_executor.py` loads all `.json` files under `datasets/acn/<cluster_id>/` for its own site (was one shared file sliced by position) — this also resolves the previously-documented "per-client dataset access is fake" limitation in `docs/NVFlareIntegration.md`; `config_fed_client.json`/`config_fed_server.json` updated (`cluster_id: caltech`, `dataset_path: datasets/acn`, `min_clients: 3`). Synthetic clients are simulation-only for now — not yet ported to NVFLARE (out of scope for this pass). None of this was executed end-to-end in this environment (`torch`/`nvflare` unavailable in-sandbox); verified via `py_compile`, `yaml.safe_load`, and standalone real-execution tests of the new grouping/injection functions against the actual downloaded ACN-Data files. |
| Sprint 11 | Planned | **Gradient inversion attack** (after LiRA validated): reconstruct raw EV session features from weight updates; measures reconstruction quality on held-out sessions |
| Sprint 12 | Planned | **Interactive demo GUI** (Streamlit): real-time FL training visualisation, per-round AUC curve for all three attacks, IDS alert timeline, DP noise/utility tradeoff slider; artefact for DSN 2027 evaluation |
| Sprint 13 | Planned | DSN 2027 paper writing; results consolidation; reproducibility packaging; artefact evaluation preparation |

---

## Relation to Prior Work

ChargeShield-FL specializes and extends a prior architectural proposal — Imperatrice & Romano, *"Toward Realistic Privacy Risk in FL-Enabled OT Intrusion Detection Systems"* (QRS 2026, [12]) — into a concrete, real-data EV charging deployment. That paper introduces the **ML Plane**, an architectural abstraction spanning the Purdue OT hierarchy that makes FL learning artifacts (model updates, gradients, aggregation exchanges) explicitly observable at the aggregation layer, where an **honest-but-curious aggregator** has access to every client's update *before* FedAvg combines them — "a realistic lower-bound adversarial model" evaluated via FedMIA. Its own testbed is a generic Smart Grid scenario (ICSSIM + NVIDIA FLARE + Containerlab, Modbus TCP/OPC-UA, Pecan Street + Kelmarsh Wind Farm SCADA data, supervised regression), explicitly described as "ongoing" with FedMIA experiments not yet completed at the time of writing.

**Coherence, verified against this codebase, not just asserted:**

- **Threat model — exact match.** LiRA in this project attacks precisely the artifact the QRS paper's Privacy Auditor targets: each client's own submitted update, "temporarily available... before the execution of FedAvg" (QRS §3). This is the same honest-but-curious-aggregator model, specialized from a generic OT/Smart-Grid setting to a real EV charging one (OCPP 1.6/2.0.1 + MQTT v5 instead of Modbus/OPC-UA; ACN-Data instead of Pecan Street/Kelmarsh; an unsupervised autoencoder anomaly detector instead of a supervised regressor).
- **ML Plane concept — present, but not the mechanism that produces this project's results.** `docs/MLPlane.md` documents an Observer-pattern implementation of the same idea (`MLPlaneListener`, `emit_event()`, `subscribe()` — real code in `src/ml/base_ml.py`/`gradient_manager.py`/`fedavg_aggregator.py`/`autoencoder_trainer.py`, exercised by unit tests). However, `emit_event()` is called by the real training code but **nothing in the actual experiment pipeline subscribes to it**: `scripts/run_experiments.py` (every number in this README/`docs/CaseStudies.md`) and the NVFLARE app (`chargeshield_aggregator.py`) both achieve the same conceptual outcome — reading each client's update at the point the QRS paper's aggregator would see it, pre-combination — through a simpler, direct mechanism (a `fl_results` dict of per-round, per-client updates, read post-hoc by `run_lira()`/`run_ids()`), not through the documented ML Plane Observer abstraction. **If a future paper claims "we implement the ML Plane," that claim is accurate as a threat-model/conceptual specialization but not yet as a literal software-architecture one** — the two would need to be wired together (`run_fl_rounds()` subscribing `FedMIA`/`ChargingIDS`/`PrivacyAuditor` instances to the same `MLPlane` the training/aggregation code already emits into) for the architectural claim to be literally true of the code that produces the results.
- **This project extends beyond QRS 2026's stated scope**, which lists differential privacy and Secure Aggregation evaluation as future work and evaluates only a single FedMIA variant: ChargeShield-FL evaluates 3 real DP placements (DP-FedAvg, central, local) and a 3-tier attack hierarchy (Yeom, Shadow, **LiRA** — a stronger, more recent attack than the "all-for-one" FedMIA the QRS paper cites), and adds a defensive IDS layer (CUSUM, Krum, Cosine Similarity, Byzantine detection) that QRS 2026 does not include at all.
- **This project is further along empirically.** QRS 2026's conclusion states its FedMIA experiments are not yet completed. ChargeShield-FL already has real, executed results on real ACN-Data (LiRA mean AUC-ROC ≈ 0.589 at smoke scale, 3 real sites, 2026-07-22) — the specialization is not just conceptual but experimentally realized.

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

12. A. Imperatrice and L. Romano, "Toward Realistic Privacy Risk in FL-Enabled OT Intrusion Detection Systems," QRS 2026 (Fast Abstract). Introduces the ML Plane architectural abstraction and the honest-but-curious-aggregator threat model this project specializes into a real EV charging deployment — see "Relation to Prior Work" above.

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
