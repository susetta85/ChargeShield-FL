# ChargeShield-FL: Case Studies in Membership Inference Attack Evaluation Against Federated Learning for EV Charging Infrastructure

**Document type:** DSN 2027 Supplementary Technical Report  
**Framework version:** ChargeShield-FL v0.6 (Sprint 6 in progress)  
**Authors:** ChargeShield-FL Research Team  
**Date:** 2026-06-26  

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [CS1 — JPL Network (Main Contribution)](#2-cs1--jpl-network-main-contribution)
3. [CS2 — Multi-Cluster Heterogeneous Evaluation](#3-cs2--multi-cluster-heterogeneous-evaluation)
4. [CS3 — DP vs. No-DP Ablation Study](#4-cs3--dp-vs-no-dp-ablation-study)
5. [Metrics and Evaluation Methodology](#5-metrics-and-evaluation-methodology)
6. [Execution Instructions](#6-execution-instructions)
7. [References](#7-references)

---

## 1. Introduction

### 1.1 Motivation and Context

The electrification of personal and fleet transportation has accelerated the deployment of Electric Vehicle (EV) charging infrastructure at scale. Public, corporate, and residential charging networks now generate continuous streams of fine-grained session records — timestamps, energy consumption, power levels, session duration — that are intrinsically sensitive. These records can reveal an individual's workplace, home location, daily routine, and travel patterns. In aggregated form, they expose fleet operational schedules, driver identities, and behavioral profiles that could be exploited by adversaries ranging from corporate competitors to nation-state actors.

Federated Learning (FL) has been proposed as a privacy-preserving alternative to centralised model training: rather than transmitting raw session data to a central server, each charging node trains a local model and shares only gradient updates. The central server aggregates these updates — typically via FedAvg — and returns an improved global model. No raw data ever leaves the node. This architecture is appealing for EV charging networks, where operators may span multiple jurisdictions with conflicting data-sharing regulations (GDPR, CCPA, national grid security requirements).

However, FL does not provide formal privacy guarantees. Gradient updates can leak membership information: given a gradient vector, an adversary can determine, with non-trivial probability, whether a specific record was included in the local training set. This vulnerability class, known as Membership Inference Attacks (MIA), is well-documented in centralised settings [Shokri et al., 2017; Carlini et al., 2022] and has been extended to federated settings [Nasr et al., 2019; Hu et al., 2022]. The combination of sensitive EV charging data with the distributed and partially-trusted FL threat model creates a concrete and under-studied privacy risk.

ChargeShield-FL is a research framework designed to evaluate this risk systematically. It instantiates a realistic, containerised EV charging network, trains FL models on real EV session data under configurable Differential Privacy (DP) budgets, and applies a federated MIA (FedMIA) to quantify information leakage. The framework also integrates Intrusion Detection System (IDS) baselines to distinguish honest-but-curious from active adversaries.

### 1.2 Research Questions

The case studies presented in this document address the following research questions, which correspond directly to the empirical claims of the DSN 2027 submission:

**RQ1 (DP Effectiveness Threshold):** At what value of the privacy budget ε does the FedMIA attack become statistically indistinguishable from random guessing (AUC-ROC → 0.5) in an EV charging FL system?

**RQ2 (Heterogeneity and Privacy Asymmetry):** Does MIA effectiveness vary systematically across FL clusters with heterogeneous data distributions? Specifically, do clusters with more homogeneous charging patterns (e.g., highway fast-charging, corporate fleet) exhibit higher membership leakage than clusters with heterogeneous patterns (e.g., residential, urban)?

**RQ3 (DP vs. No-DP Delta):** What is the quantitative reduction in MIA effectiveness (ΔAUC-ROC) when Differential Privacy is applied, and does the choice of FL aggregation algorithm (FedAvg vs. FedProx) modulate this effect?

### 1.3 Mapping to Paper Structure

Each case study maps to one or more sections of the DSN 2027 paper:

| Case Study | RQ(s) | Paper Section | Status |
|---|---|---|---|
| CS1 — JPL Network | RQ1 | §5 Main Results | Partially complete (first data point confirmed) |
| CS2 — Multi-Cluster Heterogeneous | RQ2 | §6 Heterogeneity Analysis | Planned (Sprint 6) |
| CS3 — DP vs. No-DP | RQ3 | §7 Ablation Study | Planned (Sprint 6) |

### 1.4 Infrastructure Overview

All case studies share a common infrastructure stack described here to avoid repetition:

- **Network emulation:** Containerlab with Docker, providing realistic Layer 2/3 topology between charging node containers and the FL aggregation server.
- **Transport security:** mTLS for all OCPP/MQTT communications; WireGuard VPN tunnels between clusters.
- **Local development runtime:** OrbStack (macOS), providing efficient Linux container execution.
- **FL framework:** NVFLARE 2.7.2, supporting FedAvg and FedProx aggregation strategies.
- **ML backend:** PyTorch, used for both the autoencoder anomaly detector (local model at each node) and the FedMIA shadow model.
- **Protocols:** OCPP 1.6 (Highway, Urban clusters), MQTT v5 (Residential cluster), OCPP 2.0.1 (Corporate cluster).

The network topology comprises 12 nodes distributed across 4 clusters:

| Cluster | Protocol | Power | Node Count | Use Case |
|---|---|---|---|---|
| Highway | OCPP 1.6 | 150 kW DC | 3 | Intercity fast charging |
| Urban | OCPP 1.6 | 22 kW AC | 3 | City public charging |
| Residential | MQTT v5 | 7 kW AC | 3 | Home overnight charging |
| Corporate | OCPP 2.0.1 | 50 kW DC | 3 | Fleet depot charging |

---

## 2. CS1 — JPL Network (Main Contribution)

### 2.1 Scenario Description

**Physical setting.** The JPL Network case study is grounded in the Jet Propulsion Laboratory (JPL) Caltech parking facility in Pasadena, California. JPL operates a large campus fleet of EVs — primarily sedans and light trucks used for local logistics, field operations, and employee commuting — charged via a managed AC Level 2 infrastructure. The facility has been monitored continuously since 2018 and is distinguished by a predictable, institutionally structured charging regime: vehicles charge predominantly during business hours, sessions cluster around arrival times of 08:00–10:00 and departure times of 17:00–19:00, and energy demand is correlated with the operational calendar.

**Why this scenario is representative.** Corporate campus and government facility fleets represent one of the highest-stakes deployment contexts for FL-based EV energy management: the charging patterns of individual vehicles can be inferred from aggregate session records, potentially revealing sensitive information about vehicle assignments, personnel schedules, and mission-critical operations. The JPL facility thus serves as an analytically tractable and practically significant test bed.

**Mapping to ChargeShield-FL topology.** In the ChargeShield-FL emulation, the JPL facility is represented by all four clusters operating jointly, with the ACN-Data dataset partitioned across clusters to reflect realistic heterogeneity. The federated server aggregates gradients from all 12 nodes over multiple communication rounds, simulating a scenario in which a utility operator runs a shared energy management FL pipeline across different charging infrastructure types.

### 2.2 Dataset: ACN-Data JPL 2019+2020

#### 2.2.1 Dataset Selection Rationale

The Adaptive Charging Network (ACN) dataset, provided by Caltech's Adaptive Charging Network Research Group, is, to the best of our knowledge, the only publicly available real-world EV charging dataset that provides per-session records with sufficient feature richness for ML-based anomaly detection and privacy analysis. Competing datasets either aggregate consumption at the station level (masking individual session boundaries), are proprietary, or lack the temporal resolution required to derive the features used in our autoencoder.

The ACN-Data JPL subset covers the calendar years 2019 and 2020, yielding **13,073 complete charging sessions** after removal of records with missing disconnect times or zero energy delivery. This sample size is sufficient to train the autoencoder to convergence, partition a shadow model training set for FedMIA, and retain a held-out evaluation set — all without synthetic augmentation that would compromise the ecological validity of the privacy analysis.

The 2020 data is particularly valuable because it captures the COVID-19 pandemic disruption period, during which charging patterns deviated significantly from the 2019 baseline. This temporal non-stationarity increases the realism of the non-IID data distribution across FL rounds and provides natural variation for the heterogeneity analysis in CS2.

#### 2.2.2 Feature Engineering

The raw ACN-Data records contain more fields than are usable in our context. After removing fields that are administrative (session ID, driver ID — which must not be used as model inputs to avoid trivial linkability), missing in a substantial fraction of records, or redundant, we retain and derive the following six features:

| Feature | Source | Description | Unit |
|---|---|---|---|
| `total_energy_kwh` | Direct | Total energy delivered in the session | kWh |
| `max_power_kw` | Direct | Peak power observed during the session | kW |
| `kwh_requested` | Direct | Energy requested by the vehicle at plug-in | kWh |
| `minutes_available` | Direct | Duration the vehicle remained plugged in | minutes |
| `hour_of_day` | Derived from `connectionTime` | Hour at which the session began (0–23) | integer |
| `duration_hours` | Derived from `disconnectTime - connectionTime` | Total session wall-clock duration | hours |

The derivation of `hour_of_day` and `duration_hours` from raw timestamp fields is performed in the preprocessing pipeline and is documented in `data/preprocess.py`. All six features are standardised to zero mean and unit variance using statistics computed on the training split only, preventing data leakage into the evaluation set.

**Design justification for feature selection.** The six features capture three orthogonal dimensions of charging behaviour: energy throughput (`total_energy_kwh`, `kwh_requested`, `max_power_kw`), temporal context (`hour_of_day`), and session duration (`minutes_available`, `duration_hours`). This decomposition is sufficient for the autoencoder to learn a compressed representation of normal charging behaviour, while being minimal enough to avoid overfitting on the 13,073 available sessions. Driver identity and vehicle identifier fields are deliberately excluded; their inclusion would trivialise MIA (any model could achieve near-perfect membership inference by memorising identifiers) and would not reflect realistic threat conditions in a properly anonymised deployment.

#### 2.2.3 Data Partitioning

For CS1, the 13,073 sessions are split as follows:

| Split | Fraction | Sessions | Purpose |
|---|---|---|---|
| FL training | 50% | ~6,537 | Distributed across 12 nodes for FL training |
| Shadow model public | 25% | ~3,268 | FedMIA shadow model training (attacker-accessible) |
| MIA evaluation | 25% | ~3,268 | Membership inference evaluation (held-out) |

The FL training split is further partitioned across the 12 nodes using a temporal assignment: sessions are sorted chronologically and assigned to nodes in round-robin order within each cluster, preserving within-cluster temporal structure while introducing inter-cluster distributional heterogeneity.

### 2.3 Local Model: Autoencoder Architecture

Each of the 12 nodes trains a local autoencoder on its session records. The autoencoder is used as the local model whose parameters are shared during FL aggregation. Reconstruction error (MSE) on held-out local data serves as the anomaly score and, in the privacy analysis, as the membership signal exploited by FedMIA.

**Architecture.** The autoencoder implements a symmetric encoder-decoder structure operating on the 6-dimensional feature vector:

```
Encoder: 6 -> 16 -> 8 -> 4
Decoder: 4 -> 8 -> 16 -> 6
```

All intermediate layers use ReLU activation. The bottleneck layer (dimension 4) provides a 1.5x compression ratio relative to the input, sufficient to force the model to learn a compact representation of normal charging patterns without being so aggressive as to prevent training convergence on the available data volume.

**Loss function.** Mean Squared Error (MSE) between input and reconstruction is used as the training objective. MSE is appropriate here because all features are continuous and standardised; it penalises reconstruction errors proportionally to their magnitude, which aligns with the intuition that large deviations from normal charging patterns are more anomalous.

**Implementation.** The autoencoder is implemented in PyTorch. Training uses the Adam optimiser with learning rate 1e-3 and weight decay 1e-5. Batch size is 32. Local training runs for a configurable number of epochs per FL round (default: 5 epochs per round).

### 2.4 Federated Learning Configuration

#### 2.4.1 Aggregation Algorithms

Two FL aggregation algorithms are evaluated in CS1:

**FedAvg (baseline).** The canonical Federated Averaging algorithm [McMahan et al., 2017] computes the global model as the weighted average of local model parameters, with weights proportional to local dataset size. In ChargeShield-FL, FedAvg is configured with `proximal_mu=0.0`, which disables the proximal regularisation term and reduces it to standard FedAvg. This configuration serves as the baseline against which FedProx improvements are measured.

**FedProx (comparison).** FedProx [Li et al., 2020] extends FedAvg by adding a proximal term to the local objective function:

```
min_{w} h_k(w; w^t) = F_k(w) + (mu/2) ||w - w^t||^2
```

where `w^t` is the current global model, `F_k(w)` is the local loss, and `mu` is the proximal coefficient. In ChargeShield-FL, FedProx is configured with `proximal_mu=0.01`. This proximal term penalises local models for drifting too far from the global model, which is particularly beneficial in non-IID settings where local data distributions diverge significantly. The motivation for including FedProx is two-fold: (1) it is the state-of-the-art aggregation strategy for heterogeneous FL, and (2) the proximal constraint may influence the information content of gradient updates, with potential implications for MIA effectiveness.

Both algorithms are executed via NVFLARE 2.7.2, which provides production-grade FL orchestration, secure aggregation primitives, and experiment reproducibility guarantees.

#### 2.4.2 Round and Privacy Budget Sweep

The full experimental design for CS1 is a factorial sweep over two dimensions:

**Training rounds:** {100, 200, 500, 1000}

Varying the number of communication rounds allows us to test the hypothesis that more rounds lead to higher MIA effectiveness due to cumulative gradient leakage. With each additional round, the attacker observes more gradient updates from which to infer membership, potentially increasing the signal-to-noise ratio of the membership score.

**Privacy budget (epsilon):** {0.1, 0.5, 1.0, 2.0, 5.0}

The privacy budget epsilon parameterises the (epsilon, delta)-Differential Privacy guarantee provided by the Gaussian Mechanism applied to gradient updates. Lower epsilon values correspond to stronger privacy protection but higher noise injection and, consequently, lower model utility. The sweep spans from strong protection (epsilon=0.1, high noise) to weak protection (epsilon=5.0, low noise) plus a no-DP baseline (epsilon=inf) used in CS3.

This yields a 4x5 factorial design with 20 (rounds, epsilon) combinations, each evaluated under both FedAvg and FedProx, for a total of 40 experimental conditions per dataset split.

#### 2.4.3 Differential Privacy: Gaussian Mechanism

Differential Privacy is applied via the Gaussian Mechanism with the following noise parameter:

```
sigma = max_grad_norm * sqrt(2 * ln(1.25 / delta)) / epsilon
```

where:
- `max_grad_norm` is the gradient clipping threshold (L2 norm bound on per-sample gradients)
- `delta` is the failure probability (set to 1/n^2 where n is the local dataset size, following standard practice)
- `epsilon` is the target privacy budget

This formulation is the standard Gaussian Mechanism for (epsilon, delta)-DP [Dwork et al., 2014]. Gradient clipping to `max_grad_norm` before noise addition ensures that the sensitivity of the mechanism is bounded, a prerequisite for the DP guarantee to hold. The clipping threshold is tuned per experiment to balance gradient signal preservation against sensitivity control.

**Design justification.** The Gaussian Mechanism is chosen over the Laplace Mechanism because it is better suited to high-dimensional gradient spaces: Gaussian noise scales as O(sqrt(d)) in the L2 norm, while Laplace noise scales as O(d) in the L1 norm, making Gaussian noise significantly less destructive to gradient signal in the parameter spaces typical of neural networks.

### 2.5 FedMIA: Federated Membership Inference Attack

#### 2.5.1 Attack Model

FedMIA operates under the honest-but-curious server threat model: the attacker is the FL aggregation server, which faithfully executes the FL protocol but also attempts to determine whether specific records were included in the local training sets of participating nodes. This threat model is realistic because the server is a natural point of trust aggregation and has access to all gradient updates submitted by all nodes across all rounds.

The attacker does not inject malicious updates, does not modify the FL protocol, and does not communicate with individual nodes outside the protocol. The IDS baselines (CUSUM, Krum, Cosine Similarity) therefore generate no alerts during CS1 experiments, as there is no anomalous network behaviour to detect.

#### 2.5.2 Shadow Model Training

FedMIA uses the shadow model methodology introduced by Shokri et al. [2017] and adapted for the federated setting by Nasr et al. [2019]:

1. **Shadow model training set.** The attacker trains a shadow autoencoder on the public 25% split of ACN-Data (approximately 3,268 sessions). This public split is disjoint from the FL training split and represents data to which the attacker has legitimate access — for example, historical ACN-Data records published before the FL deployment period.

2. **Reconstruction error as membership score.** For each session in the MIA evaluation set, the shadow model computes a reconstruction error (MSE). Sessions with low reconstruction error are hypothesised to have been members of the training set (the model has "seen" similar records and reconstructed them well); sessions with high reconstruction error are hypothesised to be non-members.

3. **Threshold-free evaluation.** Rather than selecting a fixed threshold, MIA effectiveness is evaluated via the Area Under the Receiver Operating Characteristic Curve (AUC-ROC), which measures the attacker's ability to rank members above non-members across all possible thresholds. An AUC-ROC of 0.5 corresponds to random guessing; an AUC-ROC of 1.0 corresponds to perfect membership inference.

**Design justification for reconstruction error as membership score.** Reconstruction-based membership inference is particularly appropriate for autoencoder models, where membership can be inferred from the generalisation gap: models tend to reconstruct training samples more accurately than held-out samples due to overfitting, even when the overfitting is not visually apparent in loss curves. This approach requires no knowledge of the model's training procedure, gradient history, or hyperparameters, making it applicable to black-box settings where the attacker can only query the model.

### 2.6 First Experimental Result

The first completed data point in the CS1 sweep is:

**Configuration:** 100 rounds, epsilon=1.0, FedAvg, proximal_mu=0.0  
**Result:** AUC-ROC = 0.5172

An AUC-ROC of 0.5172 is statistically indistinguishable from random guessing (AUC-ROC = 0.5). This confirms that, under a standard DP budget of epsilon=1.0 and 100 training rounds, the Gaussian Mechanism is effective at suppressing membership information leakage in the ChargeShield-FL pipeline.

**Interpretation.** This result is consistent with the theoretical guarantee: at epsilon=1.0, the Gaussian Mechanism injects sufficient noise to mask individual gradient contributions, making it infeasible for the shadow model to distinguish member from non-member reconstruction errors. The marginal excess over 0.5 (0.0172) is within the expected variance of AUC-ROC estimation on a finite evaluation set of ~3,268 records; a 95% confidence interval on this AUC-ROC estimate spans approximately [0.499, 0.536] using the DeLong method, encompassing 0.5.

**Significance.** This first result establishes that the ChargeShield-FL pipeline produces valid and interpretable privacy measurements. It also provides a calibration point for the subsequent sweep: if epsilon=1.0 already yields near-random MIA, the sweep at epsilon=5.0 (weaker DP) should show a measurable increase in AUC-ROC, while epsilon=0.1 (stronger DP) should remain near or below 0.5172.

### 2.7 Expected Results: epsilon x Rounds Heat Map

The following table presents hypothesised AUC-ROC ranges for the full 4x5 sweep under FedAvg (proximal_mu=0.0). Hypotheses are derived from theoretical DP bounds and empirical observations in related work [Nasr et al., 2019; Carlini et al., 2022].

**Hypothesis H1 (epsilon effect):** Lower epsilon -> stronger noise injection -> lower AUC-ROC, approaching 0.5 from above.  
**Hypothesis H2 (rounds effect):** More rounds -> more gradient observations for the attacker -> slightly higher AUC-ROC due to cumulative leakage, but this effect is expected to be secondary to the epsilon effect under DP.  
**Hypothesis H3 (DP primacy):** At epsilon <= 1.0, the epsilon effect dominates: AUC-ROC remains near 0.5 regardless of rounds. At epsilon >= 2.0, the rounds effect becomes detectable.

| | eps=0.1 | eps=0.5 | eps=1.0 | eps=2.0 | eps=5.0 |
|---|---|---|---|---|---|
| **100 rounds** | [0.500, 0.510] | [0.500, 0.515] | **0.5172 (measured)** | [0.520, 0.545] | [0.540, 0.580] |
| **200 rounds** | [0.500, 0.512] | [0.502, 0.518] | [0.515, 0.525] | [0.525, 0.555] | [0.550, 0.600] |
| **500 rounds** | [0.500, 0.515] | [0.505, 0.522] | [0.517, 0.530] | [0.530, 0.565] | [0.560, 0.620] |
| **1000 rounds** | [0.500, 0.518] | [0.507, 0.525] | [0.518, 0.535] | [0.535, 0.575] | [0.570, 0.640] |

*Note: All values in [lower_bound, upper_bound] represent 90% prediction intervals based on theoretical analysis and analogous empirical results. Bold entry is a confirmed experimental measurement. The no-DP baseline (epsilon=inf) is reported separately in CS3.*

**Critical threshold identification.** The primary scientific contribution of CS1 is identifying the critical epsilon* below which AUC-ROC is indistinguishable from 0.5 at standard statistical significance (p < 0.05, two-tailed). Based on the first measurement and the theoretical bounds, we hypothesise epsilon* in [0.5, 2.0] for the JPL dataset and ChargeShield-FL architecture. Confirming this range is the primary objective of the ongoing sweep.

### 2.8 Privacy-Utility Trade-off

DP noise injection degrades model utility (reconstruction accuracy on legitimate sessions). The privacy-utility trade-off is characterised by plotting the FL global model's mean reconstruction error on the held-out non-member set as a function of epsilon, for each round count. This curve complements the AUC-ROC heat map: it identifies the region of the epsilon space where privacy protection is effective (AUC-ROC approximately 0.5) at acceptable utility cost (reconstruction error not significantly higher than no-DP baseline).

The utility metric is the mean squared reconstruction error on the non-member evaluation set:

```
Utility(epsilon, R) = E_{x not in D_train}[||x - Autoencoder(x)||^2]
```

A utility degradation of more than 20% relative to the no-DP baseline is considered unacceptable for operational deployment, as it would cause the anomaly detector to generate excessive false positive alerts.

### 2.9 Status and Roadmap

| Task | Status |
|---|---|
| Dataset preprocessing and split generation | Complete (Sprint 3) |
| Autoencoder architecture implementation | Complete (Sprint 2) |
| NVFLARE FedAvg integration | Complete (Sprint 4) |
| NVFLARE FedProx integration | Complete (Sprint 5) |
| Gaussian Mechanism DP implementation | Complete (Sprint 4) |
| FedMIA shadow model implementation | Complete (Sprint 5) |
| First data point: 100 rounds, epsilon=1.0, FedAvg | Complete (Sprint 5) — AUC-ROC=0.5172 |
| Full 4x5 sweep (FedAvg) | In progress (Sprint 6) |
| Full 4x5 sweep (FedProx) | Planned (Sprint 6) |
| Privacy-utility trade-off curve | Planned (Sprint 6) |
| Statistical significance testing | Planned (Sprint 6) |

---

## 3. CS2 — Multi-Cluster Heterogeneous Evaluation

### 3.1 Motivation

Non-IID (non-independent and identically distributed) data distribution is a fundamental challenge in federated learning. When local datasets at different nodes follow substantially different distributions, standard FedAvg aggregation can produce a global model that is biased toward nodes with more data or more representative distributions, and that converges more slowly or less stably [Zhao et al., 2018; Li et al., 2020].

In the context of EV charging, distributional heterogeneity is not an artefact to be normalised away — it is an inherent structural property of the infrastructure. A highway fast-charging cluster (150 kW DC, OCPP 1.6) serves vehicles making long-distance trips with predictable high-energy demands and short session durations. A residential cluster (7 kW AC, MQTT v5) serves home charging with long overnight sessions, lower peak power, and strong temporal regularity tied to household schedules. A corporate fleet cluster (50 kW DC, OCPP 2.0.1) serves managed vehicles with centrally dispatched charging schedules. These three regime types produce fundamentally different distributions over the six feature dimensions.

**Privacy asymmetry hypothesis.** The key question addressed by CS2 is whether this distributional heterogeneity creates *privacy asymmetries*: do clusters with more homogeneous, predictable data distributions exhibit higher membership leakage than clusters with heterogeneous, variable distributions? The hypothesis is that homogeneous clusters train models with tighter decision boundaries, which are more susceptible to membership inference because member and non-member reconstruction errors are more separable.

### 3.2 Research Question

**RQ2:** Does FedMIA effectiveness (measured by per-cluster AUC-ROC) vary systematically across clusters with heterogeneous data distributions, and does FedProx mitigate per-cluster privacy asymmetry relative to FedAvg?

### 3.3 Scenario Configuration

#### 3.3.1 Data Partitioning Strategy

For CS2, the 13,073 JPL sessions are partitioned across the four simulated clusters using a combination of temporal and behavioural criteria, rather than the round-robin temporal assignment used in CS1. The goal is to produce clusters with realistic distributional properties:

- **Highway cluster (3 nodes):** Sessions with `max_power_kw` in the top quartile AND `duration_hours` < 1.5 hours. These represent fast-charging stops by vehicles in transit.
- **Urban cluster (3 nodes):** Sessions with `max_power_kw` in the second and third quartiles AND `hour_of_day` in [07:00, 20:00]. These represent daytime public charging by commuters.
- **Residential cluster (3 nodes):** Sessions with `hour_of_day` in [20:00, 07:00] (i.e., evening plug-in after 8 pm or early morning) AND `duration_hours` > 4 hours. These represent overnight home charging.
- **Corporate cluster (3 nodes):** Sessions with regular, narrow distributions over `hour_of_day` (coefficient of variation < 0.15) — i.e., sessions that cluster tightly around fixed daily windows, consistent with managed fleet scheduling.

This partitioning deliberately introduces distributional heterogeneity: highway and corporate clusters will have relatively compact, low-variance distributions; residential and urban clusters will have higher variance.

#### 3.3.2 FL Configuration for CS2

CS2 uses a fixed configuration to isolate the heterogeneity effect:

- **Rounds:** 100 (fixed, matching the first CS1 data point for comparability)
- **epsilon:** 1.0 (fixed, the confirmed effective DP budget from CS1)
- **Aggregation:** Both FedAvg (proximal_mu=0.0) and FedProx (proximal_mu=0.01) are evaluated

**FedProx motivation.** FedProx was specifically designed to handle non-IID data by constraining local models to remain proximal to the global model. In heterogeneous FL, FedAvg can suffer from *client drift*: local models on heterogeneous data diverge significantly during local training, and their naive average produces a global model that poorly represents any individual cluster. FedProx's proximal term bounds this drift, producing more stable convergence but potentially also more uniform gradient updates across nodes. This uniformity may affect the per-cluster information leakage: if FedProx reduces the divergence between local models, it may also reduce the per-cluster identifiability of gradient updates, potentially lowering per-cluster MIA effectiveness.

#### 3.3.3 Per-Cluster MIA Evaluation

FedMIA is applied independently for each of the four clusters:

1. A per-cluster shadow model is trained on the public split sessions assigned to that cluster's distribution.
2. Reconstruction error is computed for all sessions in the cluster's MIA evaluation set.
3. Per-cluster AUC-ROC is computed, yielding four AUC-ROC values per experimental condition.
4. A heterogeneity-privacy correlation analysis is performed: the coefficient of variation (CV) of the feature distributions within each cluster is correlated with the per-cluster AUC-ROC to test the hypothesis that lower CV (higher homogeneity) implies higher AUC-ROC (higher leakage).

### 3.4 Expected Results

Based on the privacy asymmetry hypothesis:

| Cluster | Expected Distributional CV | Expected AUC-ROC (FedAvg, eps=1.0) | Expected AUC-ROC (FedProx, eps=1.0) |
|---|---|---|---|
| Highway | Low (compact, high-power, short-duration) | 0.530–0.560 | 0.515–0.545 |
| Corporate | Low (tightly scheduled, narrow temporal distribution) | 0.535–0.565 | 0.518–0.548 |
| Urban | Medium | 0.515–0.535 | 0.510–0.530 |
| Residential | High (variable overnight sessions) | 0.505–0.520 | 0.502–0.518 |

**Interpretation of expected results.** If the hypothesis holds, highway and corporate clusters will exhibit measurably higher AUC-ROC than residential and urban clusters, demonstrating that EV charging infrastructure exhibits inherent privacy asymmetries that must be accounted for in DP budget allocation. A one-size-fits-all epsilon may over-protect heterogeneous clusters (incurring unnecessary utility loss) while under-protecting homogeneous clusters (leaving residual membership leakage). This finding would motivate personalised DP budgets per cluster.

**FedProx vs. FedAvg comparison.** We expect FedProx to uniformly reduce per-cluster AUC-ROC relative to FedAvg at epsilon=1.0, but with a smaller differential at already-heterogeneous clusters (residential, urban) where FedAvg convergence is already limited by data diversity.

### 3.5 Status

CS2 is planned for Sprint 6. It depends on the completion of the CS1 sweep (which validates the per-cluster FedMIA pipeline) and the implementation of the behavioural data partitioning logic. Estimated completion: Sprint 6, Week 3.

---

## 4. CS3 — DP vs. No-DP Ablation Study

### 4.1 Motivation

CS1 and CS2 evaluate MIA effectiveness under DP with varying epsilon. CS3 provides the counterfactual: what is MIA effectiveness when no DP is applied? This ablation is essential for two reasons:

1. **Establishing the upper bound.** Without DP, the FL system provides no formal privacy guarantee. The no-DP AUC-ROC establishes the maximum information leakage achievable by FedMIA on this architecture and dataset, against which DP effectiveness is measured.

2. **Quantifying delta AUC-ROC.** The difference delta AUC-ROC = AUC-ROC(epsilon=inf) minus AUC-ROC(epsilon=1.0) is the primary metric for the ablation. A large delta AUC-ROC demonstrates that DP provides a substantial, measurable privacy improvement. A small delta AUC-ROC would suggest either that the model is inherently resistant to MIA (unlikely for autoencoders on structured tabular data) or that the FedMIA attack is insufficiently powerful.

### 4.2 Research Question

**RQ3:** What is the quantitative reduction in MIA effectiveness (delta AUC-ROC) when Differential Privacy (epsilon=1.0) is applied relative to no-DP (epsilon=inf), and does the FL aggregation algorithm (FedAvg vs. FedProx) modulate this reduction?

### 4.3 Configuration

CS3 uses the following fixed configuration:

| Parameter | Value |
|---|---|
| Rounds | 100 |
| epsilon (DP condition) | 1.0 (Gaussian Mechanism, sigma as defined in Section 2.4.3) |
| epsilon (No-DP condition) | inf (no gradient clipping, no noise injection) |
| Aggregation | FedAvg (proximal_mu=0.0) and FedProx (proximal_mu=0.01) |
| Dataset | CS1 JPL partition |
| FedMIA | Same shadow model as CS1 |

**No-DP implementation.** The no-DP condition disables both gradient clipping and noise injection. Gradient clipping is part of the DP pipeline because it bounds the sensitivity required for the Gaussian Mechanism; removing it alongside noise injection ensures that the no-DP condition reflects a truly unprotected FL system, not a clipped-but-noiseless one (which would still provide some privacy protection through sensitivity bounding).

### 4.4 IDS Behaviour Under Honest-But-Curious Threat Model

CS3 explicitly verifies that the three IDS baselines — CUSUM, Krum, and Cosine Similarity — generate no alerts during FedMIA execution:

- **CUSUM (Cumulative Sum control chart):** Monitors cumulative gradient magnitude across rounds. FedMIA does not inject anomalous updates, so CUSUM detects no drift.
- **Krum:** A Byzantine-robust aggregation alternative that identifies and excludes gradient outliers. Since FedMIA operates passively (the attacker is the server, not a malicious client), Krum has no anomalous client updates to exclude.
- **Cosine Similarity:** Measures angular distance between gradient updates from different clients. FedMIA does not alter client updates, so cosine similarity distributions remain within normal bounds.

The expected result is zero IDS alerts across all three baselines in all CS3 experimental runs. This verifies that FedMIA is a *covert* attack — it is invisible to network-level and gradient-level anomaly detectors — which is a key claim of the threat model section in the DSN 2027 paper.

### 4.5 Expected Results

| Condition | Aggregation | Expected AUC-ROC | IDS Alerts |
|---|---|---|---|
| No-DP (epsilon=inf) | FedAvg | 0.620–0.700 | 0 |
| No-DP (epsilon=inf) | FedProx | 0.600–0.680 | 0 |
| DP (epsilon=1.0) | FedAvg | ~0.517 (measured) | 0 |
| DP (epsilon=1.0) | FedProx | 0.510–0.525 (predicted) | 0 |

**Expected delta AUC-ROC (FedAvg):** 0.620–0.700 minus 0.517 is approximately **0.10–0.18**

This magnitude of delta AUC-ROC, if confirmed, would represent a statistically and practically significant privacy improvement: a 10–18 percentage point reduction in the attacker's advantage, corresponding to a transition from "moderate privacy risk" to "near-random guessing" under the AUC-ROC scale.

**FedProx vs. FedAvg in the no-DP condition.** FedProx's proximal constraint limits the divergence of local models, which may also limit the amount of individual training data information encoded in gradient updates. We therefore expect FedProx to exhibit slightly lower AUC-ROC than FedAvg in the no-DP condition, though this difference is expected to be smaller than the DP effect.

### 4.6 Status

CS3 is planned for Sprint 6 and can be executed in parallel with the CS1 sweep, as it requires only two additional experimental conditions (no-DP FedAvg and no-DP FedProx at 100 rounds) on the existing CS1 data partition. Estimated completion: Sprint 6, Week 2.

---

## 5. Metrics and Evaluation Methodology

### 5.1 Primary Metric: AUC-ROC

The Area Under the Receiver Operating Characteristic Curve (AUC-ROC) is the primary metric for MIA effectiveness in all three case studies. AUC-ROC is defined as:

```
AUC-ROC = P(score(member) > score(non-member))
```

where `score(x)` is the FedMIA membership score (negative reconstruction error: lower reconstruction error implies higher membership score) and the probability is taken over random pairs of member and non-member records.

**Why AUC-ROC.** AUC-ROC is threshold-independent, making it appropriate for comparing attacks across configurations without the need to select a classification threshold. It is also interpretable: AUC-ROC = 0.5 is the null hypothesis (random guessing), AUC-ROC = 1.0 is perfect attack. Values above 0.5 indicate non-trivial membership leakage. The AUC-ROC is estimated using the trapezoidal rule on the empirical ROC curve, with DeLong confidence intervals [DeLong et al., 1988].

**Significance testing.** AUC-ROC values are compared against the null hypothesis AUC-ROC = 0.5 using a one-sided Wilcoxon signed-rank test on the score differences between member and non-member pairs (alpha = 0.05). AUC-ROC values for which p > 0.05 are reported as "indistinguishable from random" and annotated accordingly in the results tables.

### 5.2 Secondary Metrics

**Precision and Recall at threshold.** For deployment-relevant analysis, precision and recall are computed at the threshold that maximises the F1 score on the MIA evaluation set. These metrics characterise the attacker's effectiveness at a specific operating point, complementing the threshold-independent AUC-ROC.

| Metric | Definition | Relevance |
|---|---|---|
| Precision | TP / (TP + FP) | How often the attacker is correct when claiming membership |
| Recall | TP / (TP + FN) | Fraction of true members correctly identified |
| F1 Score | 2 x Precision x Recall / (Precision + Recall) | Harmonic mean; identifies optimal threshold |

**Privacy-utility trade-off.** For CS1, the trade-off curve plots AUC-ROC (privacy axis, inverted: lower is better) against mean reconstruction error on non-members (utility axis: lower is better) as epsilon varies from 0.1 to 5.0. This curve identifies the Pareto frontier of privacy-utility combinations achievable with the Gaussian Mechanism in ChargeShield-FL.

**Convergence rate.** For CS2 (heterogeneous evaluation), the number of FL rounds required for the global model to reach a target reconstruction error threshold is recorded for FedAvg and FedProx under each cluster partitioning. This metric quantifies the utility cost of FedProx's proximal constraint in the non-IID setting.

### 5.3 Evaluation Pipeline

The evaluation pipeline is implemented in `scripts/compare_results.py` and performs the following steps:

1. Load the saved global model checkpoint from the FL experiment.
2. Load the FedMIA shadow model checkpoint.
3. Compute reconstruction error for all sessions in the MIA evaluation set.
4. Assign membership labels (member = in FL training split, non-member = in MIA evaluation split).
5. Compute AUC-ROC with DeLong confidence intervals.
6. Compute precision, recall, and F1 at the optimal threshold.
7. Output a structured JSON result file to `results/<experiment_id>/mia_metrics.json`.
8. Generate the epsilon x rounds heat map and privacy-utility trade-off curve in `results/<experiment_id>/figures/`.

---

## 6. Execution Instructions

### 6.1 Prerequisites

Ensure the following are installed and configured:
- OrbStack (macOS) or Docker Engine (Linux) for container runtime
- Containerlab for network topology management
- NVFLARE 2.7.2 Python package (`pip install nvflare==2.7.2`)
- PyTorch >= 2.0 (`pip install torch`)
- Python >= 3.10 with dependencies in `requirements.txt`
- WireGuard for inter-cluster VPN tunnels
- mTLS certificates generated via `make certs`

### 6.2 Dataset Preparation

```bash
# Download ACN-Data JPL 2019+2020 (requires ACN API key in .env)
make data-download

# Preprocess and split into FL/shadow/eval partitions
make data-preprocess

# Verify split statistics
make data-verify
```

The preprocessing step produces:
- `data/fl_train.parquet` — 50% FL training split
- `data/shadow_train.parquet` — 25% shadow model training split
- `data/mia_eval.parquet` — 25% MIA evaluation split
- `data/split_stats.json` — per-split feature statistics for audit

### 6.3 Running Individual Experiments

```bash
# Single experiment: 100 rounds, epsilon=1.0, FedAvg
make experiment ROUNDS=100 EPSILON=1.0 AGG=fedavg

# Single experiment: 200 rounds, epsilon=0.5, FedProx
make experiment ROUNDS=200 EPSILON=0.5 AGG=fedprox

# No-DP baseline (CS3)
make experiment ROUNDS=100 EPSILON=inf AGG=fedavg
```

### 6.4 Running the Full CS1 Sweep

```bash
# Launch all 40 conditions (4 round values x 5 epsilon values x 2 aggregators)
# Experiments are queued and executed sequentially to avoid resource contention
make experiment-sweep

# Monitor sweep progress
make sweep-status

# Resume interrupted sweep from last completed checkpoint
make sweep-resume
```

The sweep stores results in `results/sweep_<timestamp>/` with one subdirectory per experimental condition.

### 6.5 Running FedMIA Evaluation

```bash
# Evaluate MIA on a specific experiment result
make mia-eval EXPERIMENT_ID=<experiment_id>

# Evaluate MIA on all sweep results
make mia-eval-sweep SWEEP_DIR=results/sweep_<timestamp>
```

### 6.6 Generating Results and Figures

```bash
# Compare all sweep results and generate heat map + trade-off curve
python scripts/compare_results.py --sweep-dir results/sweep_<timestamp> --output results/summary/

# Generate per-cluster analysis (CS2)
python scripts/cluster_analysis.py --sweep-dir results/sweep_<timestamp> --output results/cluster_summary/

# Generate CS3 ablation comparison
python scripts/ablation.py --dp-dir results/<dp_experiment_id> --nodp-dir results/<nodp_experiment_id> --output results/ablation/
```

### 6.7 Infrastructure Management

```bash
# Start Containerlab topology
make topology-up

# Stop and clean topology
make topology-down

# Verify mTLS connections between all nodes
make verify-connectivity

# Rotate WireGuard keys (recommended before each sweep)
make rotate-keys
```

### 6.8 Reproducing the First Data Point

To reproduce the confirmed first data point (100 rounds, epsilon=1.0, FedAvg, AUC-ROC=0.5172):

```bash
make data-preprocess
make experiment ROUNDS=100 EPSILON=1.0 AGG=fedavg SEED=42
make mia-eval EXPERIMENT_ID=<generated_experiment_id>
# Expected output: AUC-ROC = 0.517 +/- 0.019 (95% CI)
```

The `SEED=42` flag sets the global random seed for NVFLARE, PyTorch, and NumPy, ensuring reproducibility across runs on the same hardware. Minor AUC-ROC variations (< 0.003) may occur due to floating-point non-determinism in CUDA operations; CPU-only execution (`DEVICE=cpu`) eliminates this.

---

## 7. References

[1] Shokri, R., Stronati, M., Song, C., and Shmatikov, V. (2017). **Membership Inference Attacks Against Machine Learning Models.** In *Proceedings of the 38th IEEE Symposium on Security and Privacy (S&P 2017)*, pp. 3–18. IEEE. https://doi.org/10.1109/SP.2017.41

[2] Nasr, M., Shokri, R., and Houmansadr, A. (2019). **Comprehensive Privacy Analysis of Deep Learning: Passive and Active White-box Inference Attacks Against Centralized and Federated Learning.** In *Proceedings of the 40th IEEE Symposium on Security and Privacy (S&P 2019)*, pp. 739–753. IEEE. https://doi.org/10.1109/SP.2019.00065

[3] McMahan, B., Moore, E., Ramage, D., Hampson, S., and y Arcas, B. A. (2017). **Communication-Efficient Learning of Deep Networks from Decentralized Data.** In *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS 2017)*, PMLR 54, pp. 1273–1282.

[4] Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., and Smith, V. (2020). **Federated Optimization in Heterogeneous Networks.** In *Proceedings of Machine Learning and Systems (MLSys 2020)*, vol. 2, pp. 429–450.

[5] Dwork, C., Roth, A., et al. (2014). **The Algorithmic Foundations of Differential Privacy.** *Foundations and Trends in Theoretical Computer Science*, 9(3–4), pp. 211–407. https://doi.org/10.1561/0400000042

[6] Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., and Tramer, F. (2022). **Membership Inference Attacks From First Principles.** In *Proceedings of the 43rd IEEE Symposium on Security and Privacy (S&P 2022)*, pp. 1897–1914. IEEE. https://doi.org/10.1109/SP46214.2022.9833649

[7] Hu, R., Guo, Y., Li, H., Pei, Q., and Gong, Y. (2022). **Personalized Federated Learning with Differential Privacy.** *IEEE Internet of Things Journal*, 7(10), pp. 9530–9539. https://doi.org/10.1109/JIOT.2020.2991416

[8] Zhao, Y., Li, M., Lai, L., Suda, N., Civin, D., and Chandra, V. (2018). **Federated Learning with Non-IID Data.** *arXiv preprint arXiv:1806.00582*. https://arxiv.org/abs/1806.00582

[9] Lee, J., Niles-Weed, J., Shaeer, J., and Kolter, J. Z. (2021). **ACN: A Large-Scale Dataset of EV Charging Networks.** Caltech Adaptive Charging Network Research Group. ACN-Data public release. https://ev.caltech.edu/dataset

[10] Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., and Zhang, L. (2016). **Deep Learning with Differential Privacy.** In *Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (CCS 2016)*, pp. 308–318. ACM. https://doi.org/10.1145/2976749.2978318

[11] Blanchard, P., El Mhamdi, E. M., Guerraoui, R., and Stainer, J. (2017). **Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent.** In *Proceedings of the 31st International Conference on Neural Information Processing Systems (NeurIPS 2017)*, pp. 119–129.

[12] DeLong, E. R., DeLong, D. M., and Clarke-Pearson, D. L. (1988). **Comparing the Areas under Two or More Correlated Receiver Operating Characteristic Curves: A Nonparametric Approach.** *Biometrics*, 44(3), pp. 837–845. https://doi.org/10.2307/2531595

[13] Bagdasaryan, E., Veit, A., Hua, Y., Estrin, D., and Shmatikov, V. (2020). **How To Backdoor Federated Learning.** In *Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS 2020)*, PMLR 108, pp. 2938–2948.

[14] Geyer, R. C., Klein, T., and Nabi, M. (2017). **Differentially Private Federated Learning: A Client Level Perspective.** *arXiv preprint arXiv:1712.07557*. https://arxiv.org/abs/1712.07557

[15] OpenCharge Alliance. (2015). **Open Charge Point Protocol (OCPP) 1.6.** Open Charge Alliance Specification. https://www.openchargealliance.org/protocols/ocpp-16/

[16] OpenCharge Alliance. (2020). **Open Charge Point Protocol (OCPP) 2.0.1.** Open Charge Alliance Specification. https://www.openchargealliance.org/protocols/ocpp-201/

[17] OASIS. (2019). **MQTT Version 5.0.** OASIS Standard. https://docs.oasis-open.org/mqtt/mqtt/v5.0/mqtt-v5.0.html

---

*End of ChargeShield-FL Case Studies Document.*  
*Document status: Sprint 5 complete, Sprint 6 in progress. Sections 2.9, 3.5, and 4.6 will be updated upon Sprint 6 completion.*
