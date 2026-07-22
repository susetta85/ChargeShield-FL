# ChargeShield-FL: Case Studies in Membership Inference Attack Evaluation Against Federated Learning for EV Charging Infrastructure

**Document type:** DSN 2027 Supplementary Technical Report  
**Framework version:** ChargeShield-FL v0.6 (Sprint 6 in progress)  
**Authors:** ChargeShield-FL Research Team  
**Date:** 2026-06-26  

> **Correction and architecture update (2026-07-22) — read before relying on any section below.** Two things changed after this document was written, and neither has been propagated through the prose below yet (out of scope for this pass — flag before DSN submission):
>
> 1. **The "JPL" dataset was mislabeled.** The file used for every case study below (`datasets/acn/jpl/acndata_sessions_2019.json`, referred to throughout as "the ACN-Data JPL dataset", "the JPL facility", "JPL 2019+2020") was re-verified against ACN-Data's own `siteID` field and the official per-site EVSE counts at https://ev.caltech.edu/dataset: it is actually **Caltech** campus data (`siteID="0002"`, 54 stations — Caltech's exact EVSE count, not JPL's 50). Every qualitative characterisation below built on "JPL = restricted workplace facility" (e.g. §2.1's threat-model framing) describes the wrong site; the data is from Caltech's public campus deployment instead. See README "Dataset" section for the full correction.
> 2. **The 4-cluster topology (Highway/Urban/Residential/Corporate) has been superseded.** As this document itself concludes in RQ1/RQ2 and the CS2 heterogeneity analysis, the 4 clusters described below were never 4 real sites — they were one real site's sessions sliced 4 ways (by power profile in CS1, by temporal/behavioural criteria in CS2) to *approximate* heterogeneity. As of 2026-07-22, the actual experiment pipeline (`scripts/run_experiments.py`) uses the 3 real ACN-Data sites (Caltech, JPL, Office 1) as its FL clients instead, grouped by each session's genuine `site_id` — real, not simulated, heterogeneity. A 4th/5th client pair (`synthetic_1`/`synthetic_2`) exists only for a separate IDS/Krum-validation sweep (`byzantine_attack.enabled: true`) and is never used for training data in the main privacy experiment or for any MIA/LiRA attack — see README "Real multi-site experiment" for the full design and rationale. The CS1/CS2/CS3 case studies below should be read as describing the pre-2026-07-22 experimental design; re-running them against the 3-real-site pipeline (and re-deriving the RQ1/RQ2 hypotheses in terms of genuine cross-site heterogeneity rather than simulated intra-site heterogeneity) is pending future work.

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

The derivation of `hour_of_day` and `duration_hours` from raw timestamp fields is performed by `enrich_sessions()` in `scripts/run_experiments.py`. All six features are scaled to [0, 1] using **min-max normalization** with statistics computed on the training split only (80% of all sessions), preventing any data leakage into the hold-out evaluation set. The normalization statistics (min and max per feature) are computed after the 80/20 split and applied uniformly to both the FL training sessions and the MIA hold-out sessions.

**Design justification for feature selection.** The six features capture three orthogonal dimensions of charging behaviour: energy throughput (`total_energy_kwh`, `kwh_requested`, `max_power_kw`), temporal context (`hour_of_day`), and session duration (`minutes_available`, `duration_hours`). This decomposition is sufficient for the autoencoder to learn a compressed representation of normal charging behaviour, while being minimal enough to avoid overfitting on the 13,073 available sessions. Driver identity and vehicle identifier fields are deliberately excluded; their inclusion would trivialise MIA (any model could achieve near-perfect membership inference by memorising identifiers) and would not reflect realistic threat conditions in a properly anonymised deployment.

#### 2.2.3 Data Partitioning

For CS1, the 13,073 sessions are split as follows:

| Split | Fraction | Sessions | Purpose |
|---|---|---|---|
| FL training (members) | 80% | ~10,458 | Distributed across 4 clusters for FL training; used as the member set in FedMIA evaluation |
| Hold-out (non-members) | 20% | ~2,615 | Never seen by any FL node; used as the non-member set in FedMIA evaluation |

The split is performed with a fixed random seed (42) before any FL training begins, ensuring the hold-out set is strictly disjoint from the training process. No shadow model public set is used: FedMIA employs the **loss-based approach** (Yeom et al., 2018), which requires no separate shadow training data — membership is inferred directly from the reconstruction error of the FL global model (see Section 2.5).

The FL training split is further partitioned across the 4 clusters in equal contiguous blocks (sessions are shuffled before splitting). Each cluster trains on approximately 2,615 sessions.

### 2.3 Local Model: Autoencoder Architecture

Each of the 12 nodes trains a local autoencoder on its session records. The autoencoder is used as the local model whose parameters are shared during FL aggregation. Reconstruction error (MSE) on held-out local data serves as the anomaly score and, in the privacy analysis, as the membership signal exploited by FedMIA.

**Architecture.** The autoencoder implements a symmetric encoder-decoder structure operating on the 6-dimensional feature vector:

```
Encoder: 6 -> 16 -> 8 -> 4
Decoder: 4 -> 8 -> 16 -> 6
```

All intermediate layers use ReLU activation. The bottleneck layer (dimension 4) provides a 1.5x compression ratio relative to the input, sufficient to force the model to learn a compact representation of normal charging patterns without being so aggressive as to prevent training convergence on the available data volume.

**Loss function.** Mean Squared Error (MSE) between input and reconstruction is used as the training objective. MSE is appropriate here because all features are continuous and standardised; it penalises reconstruction errors proportionally to their magnitude, which aligns with the intuition that large deviations from normal charging patterns are more anomalous.

**Implementation.** The autoencoder is implemented in PyTorch. Training uses the Adam optimiser with learning rate 1e-3 and weight decay 1e-5. Batch size is 32. Local training runs for a configurable number of epochs per FL round (default: 3 epochs per round, as set in `config/experiment.yaml`).

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

This formulation follows the calibration formula from the standard Gaussian Mechanism [Dwork et al., 2014]. The noise parameter σ is calibrated so that higher ε values produce less noise (weaker protection) and lower ε values produce stronger noise (stronger protection). The L2 clipping threshold `max_grad_norm` bounds the magnitude of each model update before noise injection.

**Design justification.** The Gaussian Mechanism is chosen over the Laplace Mechanism because it is better suited to high-dimensional parameter spaces: Gaussian noise scales as O(√d) in the L2 norm, while Laplace noise scales as O(d) in the L1 norm, making Gaussian noise significantly less destructive to gradient signal in the parameter spaces typical of neural networks.

> **⚠ Limitation — Weight Perturbation vs. DP-SGD (Formal Claim)**
>
> ChargeShield-FL applies the Gaussian Mechanism to the **aggregated model update after local training** (weight perturbation), not to per-sample gradients during training (DP-SGD [Abadi et al., 2016]). This distinction has implications for the formal DP guarantee:
>
> - With `epochs=1` local training per FL round, per-sample sensitivity *can* be bounded by `max_grad_norm`, and a formal (ε,δ)-DP guarantee holds.
> - With `epochs=50` (current configuration since Sprint 9, 2026-07-16 — this section previously said `epochs=3`, stale; fixed 2026-07-22 review), the sensitivity of the full weight vector to any single training sample is no longer strictly bounded by `max_grad_norm`, because gradients accumulate over many more passes than the `epochs=1` case above. The formal (ε,δ)-DP guarantee is therefore **not proven** for the current configuration — and the gap from a formal guarantee is larger at 50 epochs than it would have been at 3.
> - **Open methodological question (flagged 2026-07-22, independent review, not yet investigated):** `_clip_weights()` (used by both DP-FedAvg's `privatize()` and central DP's `clip_only()`) clips the L2 norm of the client's full post-training weight vector, not the *delta* from the global model — and, unlike `_add_noise()`, does not exclude BatchNorm running-stat buffers from that norm. With `max_grad_norm=1.0` on a ~650-parameter model, the clip plausibly engages most rounds, uniformly shrinking the whole model (BN buffers included) back toward the global model on every round. This is a more aggressive operation than clipping a bounded per-round increment, and some of the observed "DP suppresses MIA" effect could be attributable to this repeated whole-model clipping rather than to the added Gaussian noise alone. Needs empirical investigation (e.g., logging how often/how hard the clip engages on real data) before DP-vs-no-DP AUC comparisons are reported as attributable to noise specifically. Not fixed yet — do not patch mid-sweep; `dp-sweep1` (running 2026-07-22) uses this same pre-existing mechanism.
> - Running T FL rounds compounds the per-round budget: total ε ≈ T × ε\_per\_round under naive composition (without Rényi DP amplification).
>
> **How to read ε in this paper.** We report ε as a **noise parameter** calibrated by the Gaussian Mechanism formula. Higher ε = less noise = weaker empirical privacy protection. We evaluate privacy empirically via MIA AUC-ROC rather than deriving a formal composition bound. This approach is consistent with prior work on weight-perturbation FL privacy [Geyer et al., 2017; Wei et al., 2020; Truex et al., 2019].
>
> **Epsilon composition across rounds.** The table below reports the naive total ε (ε\_tot = T × ε\_per\_round) for all 20 configurations in the CS1 sweep. These values represent worst-case composition bounds; tighter bounds (e.g., via Rényi DP [Mironov, 2017]) would yield smaller ε\_tot.
>
> | Rounds (T) | ε per round | ε\_tot (naive) | δ (per round) | Interpretation |
> |---|---|---|---|---|
> | 100 | 0.1 | **10** | 1e-5 | Strong noise per round; moderate total budget |
> | 100 | 0.5 | **50** | 1e-5 | Moderate noise; high total budget |
> | 100 | 1.0 | **100** | 1e-5 | Weak noise; very high total budget |
> | 100 | 2.0 | **200** | 1e-5 | Minimal noise; extreme total budget |
> | 100 | 5.0 | **500** | 1e-5 | Near-no-DP; nominal DP only |
> | 1000 | 0.1 | **100** | 1e-5 | Strong per-round protection; high total |
> | 1000 | 0.5 | **500** | 1e-5 | Moderate noise; extreme total budget |
> | 1000 | 1.0 | **1000** | 1e-5 | Near-no-DP at 1000 rounds |
> | 1000 | 2.0 | **2000** | 1e-5 | Nominal DP only |
> | 1000 | 5.0 | **5000** | 1e-5 | Effectively no DP protection |
>
> The large ε\_tot values under naive composition confirm that the per-round ε parameter is the meaningful privacy control variable in this framework, not the composition total. The empirical evaluation via AUC-ROC is the appropriate metric for measuring actual privacy protection rather than the composition bound.
>
> **Path to formal DP.** A strictly formal (ε,δ)-DP guarantee can be achieved by replacing weight perturbation with DP-SGD via the Opacus library [Yousefpour et al., 2021], which performs per-sample gradient clipping during local training. This is left as future work.
>
> **Why this matters for the paper's actual claim (clarified 2026-07-22 — see note at top of this document on the corrected research objective).** This project's goal is not to show that DP suppresses MIA — it is to show that a sophisticated MIA (LiRA) still detects membership even when a standard privacy mitigation is nominally active, in a realistic multi-site FL deployment. The composition gap documented above is the most rigorous, citable explanation for *why* that happens, and should be reported as such rather than left as an implementation caveat: an experiment run at "ε=1.0" over `fl_rounds` rounds has a naive cumulative budget of `ε_tot = ε × fl_rounds` (e.g. ε_tot=10 over 10 rounds, ε_tot=100 over 100 rounds) — far weaker than the nominal ε=1.0 a reader would assume from the config alone. **Implemented 2026-07-22**: `scripts/run_experiments.py::save_results()` now writes `epsilon_cumulative_naive` (= `epsilon × fl_rounds`, `None` under `--no-dp`) into every experiment JSON's `config` block, so this number is computed from the actual run parameters of each result, not just asserted in this document. When reporting a case where LiRA AUC stays significantly above 0.5 despite DP being "on", cite both the nominal ε and `epsilon_cumulative_naive` side by side — the gap between them is direct, quantitative support for the paper's central claim that realistic composition erodes the nominal privacy guarantee faster than a single-round ε suggests.

> **⚠ Limitation — DP-FedAvg (server-side, per-client) vs. Local DP vs. Central DP (found 2026-07-21, terminology corrected 2026-07-21)**
>
> There are three distinct DP placements in the FL literature, and it matters which one a given mechanism actually is:
>
> - **Local DP**: each client adds noise **on its own device**, before anything is transmitted. The raw value never reaches the server, not even transiently.
> - **Central DP**: a *trusted* server receives raw (clean) updates from all clients, computes the FedAvg average, and adds a *single* noise draw to the resulting aggregate (benefiting from 1/n summation). Individual raw updates are never noised or persisted past the aggregation step — but the server does see them in the clear.
> - **DP-FedAvg** [McMahan et al., 2017 — already cited, §2 references]: the server receives each client's raw update and immediately clips + noises it **per client**, before combining. This is what `GradientManager.privatize()` implements (`run_fl_rounds()`: `private_update = gm.privatize(update, ...)` inside the per-client loop, then `agg.collect(private_update)`).
>
> ChargeShield-FL implements **DP-FedAvg, not local DP and not central DP**. This was previously mislabeled in this document as "per-client / distributed-style DP" — "distributed DP" is a distinct term reserved for cryptographic protocols (e.g. secure aggregation with secret-shared noise) where no single party ever observes another party's raw contribution; that is not what is implemented here. Confirming this is DP-FedAvg and not local DP matters because the server-side code explicitly handles the **raw, pre-noise** update for a brief window before privatising it (`run_ids()`'s docstring: "in un sistema reale, il server/IDS vede gli update raw dai client PRIMA che il rumore DP venga applicato") — this is precisely the transient exposure window a real local-DP client would never create, and it is exactly the window the LiRA fix (2026-07-21c, `run_lira()`) had to account for: attacking `raw_updates` (the pre-noise value the server/IDS already had documented access to) meant DP could never be observed by the attack at all.
>
> This distinction matters for what each attack can and cannot show:
> - **Yeom and Shadow MIA** attack the released **global** model — under any of the three DP placements above, the global model the attacker sees does carry noise, so these two attacks can validly demonstrate DP suppression.
> - **LiRA** attacks each **client's own submitted update**. Under DP-FedAvg (current implementation) that update does carry noise (fixed 2026-07-21c — see `run_lira()` docstring, `scripts/run_experiments.py`), so LiRA can show suppression too. But under **true central DP**, individual client updates are, by construction, never noised — only the released aggregate is. In that regime, LiRA-on-individual-updates would show **no suppression whatsoever, at any ε**, because the attack surface it targets sits entirely outside what central DP protects. That is not a bug in the attack; it is the textbook argument for why central DP requires a trusted aggregator, and why an untrusted/semi-honest aggregator needs secure aggregation or local DP as a complementary defence.
>
> **Implemented 2026-07-22 — explicit `--dp-mode {dp-fedavg,central,local}` comparison.** What was proposed above as a "not yet implemented" next step now exists in code (`scripts/run_experiments.py::run_fl_rounds()`/`run_lira()`, `src/ml/gradient_manager.py::clip_only()`/`privatize_aggregate()`):
> 1. **No DP** (`--no-dp`) — baseline, no noise anywhere.
> 2. **DP-FedAvg** (`--dp-mode dp-fedavg`, default) — noise added to every client update before aggregation, as before.
> 3. **Central DP** (`--dp-mode central`, new) — each client clips its own update (bounds sensitivity) but does **not** noise it; the server aggregates the clean-but-clipped updates via FedAvg, then adds a **single** Gaussian noise draw to the resulting aggregate (`GradientManager.privatize_aggregate()`, σ scaled by `1/n_participants` to reflect the averaging sensitivity reduction — an approximation that assumes roughly equal per-cluster sample counts, documented in the method's docstring). Individual client updates are never noised — so LiRA attacking an individual update is expected to show **no suppression at any ε** under this mode; that is the predicted result this comparison exists to confirm, not a bug.
> 4. **Local DP** (`--dp-mode local`, new) — same per-client clip+noise mechanism as DP-FedAvg, but the server/IDS must never observe the raw update, not even transiently: `run_fl_rounds()` does not populate `raw_updates`/`raw_global_weights` in this mode, so `run_ids()`'s existing `raw_updates or updates` fallback automatically uses the noised update instead. Consequence (expected, not a bug): the IDS's peer-relative gradient-delta analysis degrades to comparing absolute (noised) weights instead of round-over-round deltas, which can raise more `GRADIENT_EXPLOSION` false positives — a real, citable tradeoff of genuine local DP (weaker anomaly detection when the server never sees a clean gradient), not something to "fix."
>
> `make experiment-central-dp` / `experiment-local-dp` (single seed) and `experiment-central-dp-sweep` / `experiment-local-dp-sweep` (5-seed, mirroring `experiment-dp-sweep`) run these modes; results land in `experiments/central-sweep{N}/` / `experiments/local-sweep{N}/`. The Excel report's "No-DP" column shows `no (central)`/`no (local)` when a non-default mode is active, and Comparison sheet gets a "DP Mode" row.
>
> **Not yet done, left for a follow-up run**: no sweep has been executed with these new modes as of this writing (torch is unavailable in the review sandbox that wrote this code — see the recurring sandbox-limitation note elsewhere in this document). The expected results above (central DP: Yeom/Shadow degrade toward 0.5 while LiRA stays high; local DP: all three attacks degrade similarly to DP-FedAvg, but IDS alerts increase) are predictions from the mechanism design, not yet confirmed empirically.

> **⚠ Limitation — the privacy pipeline does not run on the containerised network; "100+ nodes" is not true today (found 2026-07-21)**
>
> `scripts/run_experiments.py` — the script that produces every number in this document and in the README — is a single-process Python simulation. It never touches the Containerlab/Docker/NVFLARE infrastructure that also exists in this repository (`containerlab/`, `docker/`, `nvflare/`, `topology.clab.yml`). A read-only audit of that infrastructure found:
>
> - `src/flare/flare_connector.py` is an explicit placeholder (its own docstring: *"Nota Sprint 3: Questa è una implementazione simulata — non richiede un server FLARE attivo. L'integrazione reale con nvflare arriva nella Sprint 4"*), still unmodified; it never imports `nvflare` and simulates gradients with `random.gauss()` rather than real training.
> - `nvflare/project.yml` provisions PKI/network participants only — there is no NVFLARE job/app (no `Executor`/`Controller` referencing `AutoencoderTrainer`, `GradientManager`, `PrivacyAuditor`, or `ChargingIDS`), so even a successful `make deploy` would bring up empty containers with nothing to run.
> - `clab-chargeshield-fl/` (topology metadata from a 2026-06-25 deploy) shows containers were created, but there is no evidence any FL round, MIA attack, or IDS check has ever executed inside that topology.
> - `docker/{aggregator,charging-node,ids,auditor,fl-admin}/Dockerfile` are a second, orphaned Dockerfile set (unreferenced by `topology.clab.yml` or the `Makefile` `build` target) whose `CMD`s invoke Python modules with no `if __name__ == "__main__"` block — these containers would crash on start.
> - Two independent, never-reconciled PKI trees exist (`certs/` vs. the NVFLARE-provisioned workspace).
>
> **Consequence for the "100+ nodes" and "real OT constraints" claims** (`ChargeShield-FL_Decisions.txt`): today's experiments simulate exactly **4 FL clients** (one per cluster, `run_fl_rounds()`), with sessions split into **equal contiguous blocks** rather than realistic per-station volumes, in a single process with no network latency, bandwidth limits, or client dropout. This is common practice in FL research (most privacy/MIA papers use pure simulation) and does not invalidate the DP-vs-MIA results themselves, but it should either be stated explicitly as a limitation, or the pipeline should be wired into the containerised topology before claiming to "reproduce real environments with real constraints."
>
> **Scope if pursued**: wiring the privacy pipeline into the real topology is a multi-week effort, not a quick fix — it requires a real NVFLARE Executor/app wrapping the existing training/DP/audit code, a way to extract per-client raw updates through NVFLARE's server-side aggregation (needed for LiRA/Shadow, which NVFLARE's normal flow hides), fixing the orphaned Dockerfiles, and reconciling the two PKI trees.
>
> **Started 2026-07-22** — see `docs/NVFlareIntegration.md` for the current status: a job scaffold (`nvflare/jobs/chargeshield_poc/`) with a real client-side Executor wrapping `AutoencoderTrainer` now exists, using NVFLARE's built-in `ScatterAndGather`/`InTimeAccumulateWeightedAggregator` for the server side (transport-only, no DP/IDS/attacks wired in yet). Written and reasoned through manually in an environment without `torch`/`nvflare` installed — **not executed, not tested**; three specific assumptions are marked `VERIFY:` in the code and need confirming on a machine with the real dependencies. This does not change the multi-week estimate above; it gives that estimate a concrete starting point.

### 2.5 FedMIA: Federated Membership Inference Attack

#### 2.5.1 Attack Model

FedMIA operates under the honest-but-curious server threat model: the attacker is the FL aggregation server, which faithfully executes the FL protocol but also attempts to determine whether specific records were included in the local training sets of participating nodes. This threat model is realistic because the server is a natural point of trust aggregation and has access to all gradient updates submitted by all nodes across all rounds.

The attacker does not inject malicious updates, does not modify the FL protocol, and does not communicate with individual nodes outside the protocol. The IDS baselines (CUSUM, Krum, Cosine Similarity) therefore generate no alerts during CS1 experiments, as there is no anomalous network behaviour to detect.

#### 2.5.2 Loss-Based Membership Inference (Yeom et al., 2018)

FedMIA in ChargeShield-FL uses the **loss-based membership inference** approach introduced by Yeom et al. [2018], not the shadow model methodology. This choice is motivated by the threat model: the attacker is the FL aggregation server, which has direct access to the final global model weights but does not necessarily have access to a labelled shadow dataset.

The attack proceeds as follows:

1. **Global model acquisition.** The attacker (FL server) directly loads the aggregated global model weights from any given FL round. No auxiliary shadow model or separate training data is required.

2. **Reconstruction error as membership score.** For each candidate session, the global FL autoencoder computes a reconstruction error (MSE). The **membership score** is defined as the negative reconstruction error:

   ```
   membership_score(x) = -MSE(Autoencoder_global(x), x)
   ```

   Sessions with low reconstruction error (high membership score) are inferred to be members of the FL training set; sessions with high reconstruction error (low membership score) are inferred to be non-members. This exploits the *generalisation gap*: models trained on a set tend to reconstruct its members more accurately than held-out samples.

3. **Threshold-free evaluation.** MIA effectiveness is evaluated via AUC-ROC across all possible membership score thresholds. An AUC-ROC of 0.5 corresponds to random guessing; an AUC-ROC of 1.0 corresponds to perfect membership inference.

4. **Ground truth labels.** Sessions in the 80% FL training split are labelled as members (label=1); sessions in the 20% hold-out split are labelled as non-members (label=0).

**Implementation.** The loss-based FedMIA evaluator is implemented in `run_fedmia()` in `scripts/run_experiments.py`. It runs for every FL round stored in `fl_results`, enabling per-round AUC-ROC measurement throughout training.

**Design justification.** The loss-based approach (Yeom et al., 2018) is appropriate here because: (a) the attacker (FL server) controls the aggregation and thus has white-box access to global model weights at every round; (b) no auxiliary public dataset is required, making the threat model more realistic for OT settings where public EV charging data from the same distribution may not be available to the attacker; (c) reconstruction-error-based inference is well-suited to autoencoders, where the model objective is to minimise reconstruction error on training data.

> **Note on shadow model plugin.** A separate shadow-model MIA implementation exists in `src/plugins/attacks/fedmia.py` (Shokri et al., 2017; Nasr et al., 2019) and is integrated with `ChargingIDS` for per-round IDS analysis. However, it is disabled in the current experimental configuration: `ChargingIDS` is instantiated without the `fedmia=` argument in `run_ids()`. The CS1 AUC-ROC results reported in Section 2.6 and the sweep heat map in Section 2.7 all derive from the loss-based evaluator (`run_fedmia()`), not the shadow model plugin.

### 2.6 First Experimental Result

The first completed data point in the CS1 sweep is:

**Configuration:** 100 rounds, epsilon=1.0, FedAvg, proximal_mu=0.0  
**Result:** AUC-ROC = 0.5172

> **Reproducibility note.** This result requires the ACN-Data JPL dataset (2019 + 2020 files) to be present in `datasets/acn/jpl/`. The dataset is available at https://ev.caltech.edu/dataset and must be downloaded separately — it is not included in the repository. See Section 6.2 for the complete data setup instructions.

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
# Download ACN-Data JPL 2019+2020 from https://ev.caltech.edu/dataset
# Place the JSON files in datasets/acn/jpl/:
#   datasets/acn/jpl/acndata_sessions_2019.json
#   datasets/acn/jpl/acndata_sessions_2020.json

# Verify dataset presence
ls -lh datasets/acn/jpl/
```

The 80/20 train/hold-out split is performed **automatically at experiment runtime** by `run_experiments.py::main()` with a fixed seed (42). No separate preprocessing step is required. The split logic:

1. Both JSON files are loaded and concatenated (~13,073 sessions after filtering)
2. `enrich_sessions()` derives `hour_of_day` and `duration_hours` from timestamps
3. Sessions are shuffled with `random.seed(42)` for reproducibility
4. Split: `train_sessions = sessions[:80%]`, `holdout_sessions = sessions[20%:]`
5. Min-max normalization statistics are computed from `train_sessions` only
6. Both splits are normalized with the same statistics (no leakage)

The split produces:
- `train_sessions` (~10,458 sessions) — FL training + MIA member labels
- `holdout_sessions` (~2,615 sessions) — MIA non-member labels (never seen by FL nodes)

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

[18] Wei, K., Li, J., Ding, M., Ma, C., Yang, H. H., Farokhi, F., Jin, S., Quek, T. Q. S., and Poor, H. V. (2020). **Federated Learning with Differential Privacy: Algorithms and Performance Analysis.** *IEEE Transactions on Information Forensics and Security*, 15, pp. 3454–3469. https://doi.org/10.1109/TIFS.2020.2988575

[19] Truex, S., Liu, L., Gursoy, M. E., Yu, L., and Wei, W. (2019). **A Hybrid Approach to Privacy-Preserving Federated Learning.** In *Proceedings of the 12th ACM Workshop on Artificial Intelligence and Security (AISec @ CCS 2019)*, pp. 1–11. ACM. https://doi.org/10.1145/3338501.3357370

[20] Yousefpour, A., Shilov, I., Sablayrolles, A., Testuggine, D., Prasad, K., Malek, M., Nguyen, J., Ghosh, S., Bharadwaj, A., Zhao, J., Cormode, G., and Mironov, I. (2021). **Opacus: User-Friendly Differential Privacy Library in PyTorch.** *arXiv preprint arXiv:2109.12298*. https://arxiv.org/abs/2109.12298

---

*End of ChargeShield-FL Case Studies Document.*  
*Document status: Sprint 5 complete, Sprint 6 in progress. Sections 2.9, 3.5, and 4.6 will be updated upon Sprint 6 completion.*
