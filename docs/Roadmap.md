# ChargeShield-FL: Research Roadmap toward DSN 2027

**Framework:** ChargeShield-FL — Federated Learning Privacy Evaluation for EV Charging Infrastructure  
**Target Venue:** IEEE/IFIP International Conference on Dependable Systems and Networks (DSN 2027)  
**Document status:** Living document, last revised 2026-06-26  
**Classification:** Internal research roadmap

---

## Table of Contents

1. [Vision and Research Objectives](#1-vision-and-research-objectives)
2. [Completed Sprints (1–5)](#2-completed-sprints-15)
3. [Sprint 6 (In Progress): Full Evaluation and Paper Preparation](#3-sprint-6-in-progress-full-evaluation-and-paper-preparation)
4. [Future Work Beyond Sprint 6](#4-future-work-beyond-sprint-6)
5. [DSN 2027 Submission Timeline](#5-dsn-2027-submission-timeline)
6. [Known Limitations to Address Before Submission](#6-known-limitations-to-address-before-submission)
7. [References](#7-references)

---

## 1. Vision and Research Objectives

### 1.1 Motivation

Electric vehicle (EV) charging infrastructure is undergoing rapid deployment worldwide, driven by decarbonisation mandates, vehicle electrification targets, and expanding grid integration requirements. As of 2025, public and semi-public charging networks collectively aggregate session-level telemetry — including energy delivered, session duration, arrival and departure patterns, and requested charge levels — from millions of individual charging events per day. This data is operationally necessary for demand forecasting, grid balancing, predictive maintenance, and fraud detection; however, it simultaneously encodes sensitive information about individual EV owners, including habitual travel patterns, workplace locations, overnight parking sites, and daily routines. A charging session record is, in essence, a mobility trace with sub-hour temporal resolution.

Federated Learning (FL) has been proposed as a privacy-preserving alternative to centralised model training in such settings. Under the FL paradigm, each charging node — or cluster of nodes — trains a local model on its private session data and communicates only model updates (gradients or weight deltas) to a central aggregator; raw data never leaves the node. This architecture superficially appears to satisfy privacy requirements, yet a substantial body of research has demonstrated that model updates leak information about the training data from which they were derived [Shokri et al. 2017; Nasr et al. 2019; Carlini et al. 2022]. Membership Inference Attacks (MIA) — in which an adversary determines whether a specific data record was included in a model's training set — represent one of the most practically relevant and theoretically well-studied privacy threats in this space.

Differential Privacy (DP) [Dwork and Roth 2014] provides the most principled formal defence against such inference. By injecting calibrated Gaussian noise into gradient updates before aggregation, DP bounds the statistical advantage any adversary can gain from observing the aggregate model. The tension between privacy (smaller epsilon) and utility (model performance) is well characterised in the abstract, but its empirical manifestation in domain-specific FL deployments — particularly in heterogeneous industrial IoT settings — remains incompletely understood.

ChargeShield-FL addresses precisely this gap: it is a reproducible, containerised research framework for empirically evaluating the effectiveness of MIA against differentially private federated learning across a realistic, heterogeneous EV charging topology. Rather than abstract benchmarks, the framework models the actual protocol diversity, hardware heterogeneity, and data non-i.i.d. characteristics of a representative charging network — four cluster types operating OCPP 1.6 (Highway and Urban), OCPP 2.0.1 (Corporate), and MQTT v5 (Residential), each with distinct power profiles, session durations, and occupancy patterns.

### 1.2 Research Questions

ChargeShield-FL is designed to answer four primary research questions:

**RQ1 (MIA efficacy under DP):** How does the FedMIA attack's AUC-ROC vary as a function of privacy budget epsilon and the number of FL training rounds, for a fixed network topology and dataset?

**RQ2 (Heterogeneity effects):** Does cross-cluster heterogeneity — in protocol, power level, and session distribution — amplify or attenuate membership inference risk relative to a homogeneous baseline?

**RQ3 (FedAvg vs. FedProx):** Does the proximal regularisation term in FedProx (proximal_mu=0.01) produce measurably different privacy-utility trade-offs compared to FedAvg (proximal_mu=0.0) under identical DP parameters?

**RQ4 (Detection feasibility):** Can lightweight intrusion detection mechanisms (CUSUM anomaly detection, Krum aggregation, and cosine similarity filtering) identify MIA-consistent gradient behaviour at operationally acceptable false-positive rates?

### 1.3 Why DSN

The IEEE/IFIP International Conference on Dependable Systems and Networks is the premier venue for research at the intersection of security, reliability, and safety of networked systems. DSN explicitly solicits contributions on privacy and security of cyber-physical infrastructure, federated and distributed systems, and intrusion detection — all of which are core to ChargeShield-FL's contributions. The conference's emphasis on reproducibility, experimental rigour, and real-system evaluation aligns with the framework's design philosophy: every experiment is containerised, deterministically seeded, and replayable from a single YAML configuration. DSN 2027 provides the appropriate time horizon for completing the full experimental sweep, the heterogeneity case studies, and the limitations analysis described in this roadmap.

### 1.4 Claimed Contributions

The paper targeting DSN 2027 claims the following contributions:

1. **A domain-specific FL testbed** for EV charging infrastructure with authentic protocol diversity (OCPP 1.6, OCPP 2.0.1, MQTT v5), real session data (ACN-Data JPL, 13,073 sessions), and full reproducibility via Containerlab and Docker.

2. **An end-to-end MIA evaluation pipeline** (FedMIA) using a shadow model trained on a held-out public split of ACN-Data, with reconstruction error from a trained autoencoder (6->16->8->4 architecture) as the membership score, and AUC-ROC as the primary metric.

3. **A systematic privacy-utility sweep** across rounds in {100, 200, 500, 1000} and epsilon in {0.1, 0.5, 1.0, 2.0, 5.0}, yielding 20 data points per aggregator variant (FedAvg, FedProx), for a total of 40 controlled experimental conditions.

4. **Heterogeneity case studies** (CS2: multi-cluster, CS3: DP vs. No-DP) that isolate the effect of network topology and privacy mechanism on inference risk.

5. **Empirical characterisation of the epsilon-AUC-ROC frontier** in a realistic EV context, with statistical significance testing and effect size reporting.

---

## 2. Completed Sprints (1–5)

### Sprint 1: Repository Foundations and Interface Contracts

Sprint 1 established the architectural skeleton of ChargeShield-FL: the version-controlled repository, the abstract interface hierarchy, the YAML-based configuration system, and the initial documentation layer. The central design decision made in this sprint was to define all major components as abstract base classes before implementing any concrete logic. This interface-first approach — AbstractChargingNode, AbstractIDS, AbstractPrivacyAuditor, AbstractFLConnector — enforced a separation between specification and implementation that proved essential in subsequent sprints when multiple concrete variants (FedAvg/FedProx, OCPP 1.6/MQTT/OCPP 2.0.1, CUSUM/Krum/Cosine) needed to be swapped interchangeably without altering test harnesses or evaluation scripts. The YAML configuration system deserves particular mention: rather than hard-coding experimental parameters, every tunable quantity — privacy budget epsilon, noise multiplier sigma, number of rounds, cluster membership, protocol adapter, dataset split ratios — is expressed declaratively in a single YAML file. This design decision was motivated by reproducibility requirements: a reviewer should be able to re-run any reported experiment by supplying the archived YAML and the pinned Docker image, without modifying source code. Sprint 1 also produced the initial project documentation, including the architecture overview, the data flow diagram, and the threat model narrative that would later anchor the paper's system model section. Although Sprint 1 produced no executable experimental results, it created the preconditions for all subsequent work; the interface contracts defined here remained stable across all five completed sprints, validating the upfront investment in design rigour.

### Sprint 2: Charging Node Abstraction and Dataset Integration

Sprint 2 delivered the two lowest-level concrete components of the framework: the ChargingNode implementation and the ACNDataset loader. The ChargingNode class realises the AbstractChargingNode interface for the OCPP 1.6 protocol, modelling the state machine of a real charging session — connection establishment, authorisation, start transaction, metering values, stop transaction — and exposing a clean API for session simulation that the FL training loop can invoke without knowledge of protocol internals. The choice of OCPP 1.6 as the first concrete protocol was deliberate: OCPP 1.6 is the most widely deployed version in existing infrastructure and covers both the Highway cluster (150 kW DC) and the Urban cluster (22 kW AC), representing the two highest-data-volume node types. The ACNDataset class wraps the Caltech Adaptive Charging Network dataset from the JPL garage, covering 2019 and 2020, comprising 13,073 real EV charging sessions. Six features were selected after a combination of domain analysis and preliminary correlation study: total_energy_kwh, max_power_kw, kwh_requested, minutes_available, hour_of_day, and duration_hours. These features were chosen because they are (a) universally available across all cluster types, (b) sufficient to train a meaningful autoencoder, and (c) potentially linkable to individual user identity — making them a realistic target for membership inference. The dataset class handles train/validation/test splitting, normalisation (min-max scaling to [0,1]), and reproducible random seeding. Sprint 2 thus established the ground truth against which all privacy guarantees would be measured: real session records from real drivers, parsed and featurised in a manner faithful to what a production FL deployment would consume.

### Sprint 3: Privacy Auditing, IDS Baseline, and Secure Network Infrastructure

Sprint 3 was the most architecturally broad of the completed sprints, delivering three distinct but interdependent capabilities: the PrivacyAuditor subsystem, the AbstractIDS hierarchy with three concrete baselines, and the secure containerised network topology. The PrivacyAuditor encapsulates the Gaussian mechanism for (epsilon, delta)-DP: it computes the noise standard deviation as sigma = max_grad_norm * sqrt(2 * ln(1.25 / delta)) / epsilon, clips gradients to the L2 norm bound, and applies the noise injection before each gradient transmission. This formula is the standard strong composition bound for the Gaussian mechanism [Dwork et al. 2006] and is appropriate for a single-round analysis; the multi-round composition question is explicitly deferred to future work (see Section 4). Three IDS baselines were implemented against the AbstractIDS interface: CUSUM (cumulative sum control charts for detecting distributional drift in gradient norms), Krum (Byzantine-robust aggregation that selects the gradient closest in L2 distance to its k nearest neighbours [Blanchard et al. 2017]), and Cosine Similarity filtering (flagging updates whose cosine similarity to the running mean falls below a threshold). These three mechanisms represent distinct threat models — statistical process control, Byzantine robustness, and directional anomaly detection — and their combination provides a multi-layered detection surface. The network infrastructure was deployed using Containerlab to define and instantiate the 12-node, 4-cluster topology as a reproducible graph of Docker containers connected by virtual Ethernet links. Mutual TLS (mTLS) was configured between all node pairs using self-signed certificates, ensuring that gradient transmissions are encrypted and mutually authenticated. WireGuard tunnels were added for cluster-to-aggregator channels, providing an additional layer of confidentiality against a passive network adversary. OrbStack was adopted as the container runtime on macOS development hosts for its lower memory overhead compared to Docker Desktop. Sprint 3 thus completed the security infrastructure layer, ensuring that the experimental results reflect a deployment posture consistent with real-world best practices rather than a simplified flat network.

### Sprint 4: FedMIA Attack Implementation and Detection Modules

Sprint 4 delivered the attack module that is ChargeShield-FL's central experimental instrument: FedMIA, the federated membership inference attack. The attack follows the shadow model methodology of Shokri et al. [2017] adapted to the federated setting as formalised by Nasr et al. [2019]. A shadow FL system is trained on a disjoint public split of ACN-Data; the shadow model's reconstruction error (computed by the autoencoder trained in the same sprint) serves as the membership score — the intuition being that a record used in training produces lower reconstruction error than an unseen record, because the model has overfit to its features. AUC-ROC (area under the receiver operating characteristic curve) was selected as the primary evaluation metric because it is threshold-agnostic and directly measures the attack's discriminative power across the full score range, making results comparable across experimental conditions with different base rates. The autoencoder architecture — a symmetric encoder-decoder with layers 6->16->8->4->8->16->6, trained with MSE loss in PyTorch — was designed to be expressive enough to capture the non-linear feature interactions in charging session data while remaining compact enough to train to convergence within 100 FL rounds on the available data volume. The ChargingIDS class unified the three baseline detectors from Sprint 3 under a common evaluation harness, enabling side-by-side comparison of detection rates. Sprint 4 also expanded the test suite to 52 tests, covering the attack pipeline end-to-end with mocked FL state, ensuring that AUC-ROC computation, score normalisation, and threshold sweeping behave correctly under edge cases (empty member sets, perfect separation, random baseline). The completion of Sprint 4 marked the point at which ChargeShield-FL first had all the components necessary for a meaningful experiment; Sprint 5 was required to wire them together under a production-grade FL orchestrator.

### Sprint 5: Federated Learning Plane, NVFLARE Integration, and First Experiment

Sprint 5 assembled all prior components into a functioning end-to-end FL system and produced the framework's first empirical result. The ML Plane was implemented as the coordination layer between the NVFLARE 2.7.2 orchestrator and the domain-specific node logic: AutoencoderTrainer wraps PyTorch training loops and exposes the gradient tensors that NVFLARE's executor protocol requires; GradientManager handles clipping, noise injection (delegating to PrivacyAuditor), and serialisation; FedAvgAggregator implements standard weighted FedAvg aggregation [McMahan et al. 2017] with an optional proximal term (proximal_mu=0.01) to obtain FedProx [Li et al. 2020]. NVFLARE 2.7.2 was selected as the FL orchestrator because it provides production-grade job lifecycle management, secure aggregation channels, and a well-defined executor API that cleanly separates application logic from communication infrastructure — a separation essential for academic reproducibility. The FLAREConnector (introduced in Sprint 3) was extended in Sprint 5 to handle NVFLARE's job submission and result retrieval protocols. The test suite grew from 52 to 77 tests, with the new tests covering the aggregation arithmetic, the proximal term computation, and the NVFLARE job submission mock. The first experiment — 100 FL rounds, epsilon=1.0, FedAvg aggregator — produced an AUC-ROC of 0.5172, a result only marginally above the random-baseline AUC of 0.5. This finding is substantively meaningful: at epsilon=1.0, the Gaussian mechanism's noise injection is sufficient to suppress the membership signal to near-chance levels for this dataset and attack configuration. Whether this result holds at larger epsilon (weaker privacy) and more rounds (greater overfitting risk) is precisely the question that Sprint 6's full sweep addresses. Sprint 5 thus transformed ChargeShield-FL from a collection of independently tested modules into a reproducible experimental apparatus capable of producing publishable measurements.

---

## 3. Sprint 6 (In Progress): Full Evaluation and Paper Preparation

Sprint 6 is the final sprint before DSN 2027 submission. Its scope encompasses the complete experimental programme, the statistical analysis, two architectural case studies, and the full paper manuscript. Each task is described below with its motivation, methodology, and acceptance criteria.

### 3.1 Full Sweep Execution (Rounds x epsilon Grid)

**Task description.** Execute the complete factorial experiment: rounds in {100, 200, 500, 1000} x epsilon in {0.1, 0.5, 1.0, 2.0, 5.0}, for both FedAvg and FedProx, yielding 40 experimental conditions. Each condition produces one AUC-ROC value from the FedMIA attack. All conditions run with the same random seed, the same dataset split, and the same 12-node topology defined in the Sprint 3 Containerlab configuration.

**Motivation.** The single data point from Sprint 5 (rounds=100, epsilon=1.0, AUC-ROC=0.5172) is insufficient to characterise the privacy-utility trade-off surface. The research literature on DP-FL [Abadi et al. 2016; Mironov 2017] suggests that both increasing rounds (which accumulates gradient information about individual records) and increasing epsilon (which injects less noise) should increase the attack's AUC-ROC. Whether this monotonic relationship holds in the EV domain, and at what gradient of change, is an empirical question that requires the full grid.

**Experimental protocol.** Each experiment is submitted as a NVFLARE job via the FLAREConnector with the YAML configuration specifying the (rounds, epsilon) pair. Gradient clipping norm is fixed at max_grad_norm=1.0 across all conditions. Delta is fixed at 1e-5 (standard choice for datasets of this size [Dwork and Roth 2014]). The FedMIA attack is run once per trained model, using the fixed shadow model trained on the public ACN-Data split. AUC-ROC is computed with 95% bootstrap confidence intervals (1,000 bootstrap samples) to quantify sampling uncertainty.

**Acceptance criteria.** All 40 (rounds, epsilon) conditions complete without runtime error. AUC-ROC values and confidence intervals are written to a structured CSV file. The CSV is committed to the repository and archived with a hash checksum for reproducibility verification.

**Expected duration.** Approximately 3 weeks of wall-clock time, given the sequential nature of NVFLARE job execution and the 1,000-round conditions' training time on the development hardware.

### 3.2 Statistical Analysis of AUC-ROC Results

**Task description.** Conduct a systematic statistical analysis of the 40 AUC-ROC measurements to characterise the epsilon-AUC-ROC frontier, the rounds effect, and the FedAvg/FedProx difference.

**Analysis plan.**

*Primary analysis: epsilon effect.* For each value of rounds, plot AUC-ROC as a function of epsilon for both aggregators. Fit a monotone regression (isotonic regression) to quantify the trend direction. Compute Spearman's rank correlation between epsilon and AUC-ROC, with p-values corrected for multiple comparisons using the Benjamini-Hochberg procedure [Benjamini and Hochberg 1995]. Report the epsilon threshold below which AUC-ROC is statistically indistinguishable from 0.5 (i.e., where the 95% confidence interval includes 0.5).

*Secondary analysis: rounds effect.* For each value of epsilon, model AUC-ROC as a function of log(rounds). Compute Pearson's r between log(rounds) and AUC-ROC; report R-squared and the 95% confidence interval on the slope. The hypothesis is that more rounds monotonically increases AUC-ROC for epsilon > 1.0, where noise is insufficient to suppress the signal, but not for epsilon <= 1.0.

*Tertiary analysis: FedAvg vs. FedProx.* For each (rounds, epsilon) pair, compute the absolute and relative difference in AUC-ROC between FedAvg and FedProx. Apply a paired Wilcoxon signed-rank test across all 20 pairs. The null hypothesis is that FedProx's proximal regularisation term does not affect the attack's membership discriminability.

*Effect size reporting.* All hypothesis tests are accompanied by Cohen's d or eta-squared as appropriate, to distinguish statistical significance from practical significance.

*Visualisation.* Produce a heatmap of AUC-ROC values indexed by (rounds, epsilon), separately for FedAvg and FedProx. Produce line plots of AUC-ROC vs. epsilon for each value of rounds, with error bars. All plots are generated in Matplotlib with LaTeX font rendering for direct inclusion in the paper.

**Acceptance criteria.** All statistical tests pass a sanity check (random-baseline AUC-ROC at epsilon=0.1 should not be significantly above 0.5). Analysis script is committed to the repository with a requirements.txt ensuring exact library version reproducibility.

### 3.3 Case Study CS2: Multi-Cluster Heterogeneity Experiment

**Task description.** CS2 isolates the effect of cross-cluster data heterogeneity on membership inference risk. In CS2, all four cluster types (Highway, Urban, Residential, Corporate) participate simultaneously in the FL round, each with its own local data distribution. The experiment is repeated at a fixed (rounds=500, epsilon=1.0) condition — the point at which the primary sweep data is expected to show the most nuanced behaviour — and compared against a homogeneous baseline in which all 12 nodes draw from an i.i.d. uniform sample of ACN-Data.

**Motivation.** The non-i.i.d. nature of federated data is well known to affect model convergence and, by extension, the degree to which individual records leave detectable traces in the gradient [Zhao et al. 2018; Li et al. 2020]. In the EV domain, the four cluster types differ substantially in session energy (7 kW residential vs. 150 kW highway), duration (overnight residential vs. sub-hour urban), and arrival patterns — creating a strongly heterogeneous data landscape. CS2 tests whether an adversary observing the aggregate model benefits from this heterogeneity (e.g., because cluster-specific features are easier to distinguish from the global average) or is harmed by it (e.g., because heterogeneity increases gradient noise, masking individual contributions).

**Protocol.** The Containerlab topology from Sprint 3 is used without modification. YAML configurations for the homogeneous and heterogeneous conditions are generated by setting the cluster_weights parameter in the dataset loader to either uniform (i.i.d. baseline) or proportional to real ACN-Data cluster frequencies (heterogeneous). FedMIA is run identically in both conditions. AUC-ROC difference and 95% confidence interval are reported.

**Acceptance criteria.** Both conditions (homogeneous and heterogeneous) complete. The AUC-ROC difference is reported with a two-sample z-test for proportions (AUC-ROC being an area under a curve estimated from ranked scores). The result is interpreted as evidence either for or against heterogeneity as a privacy risk amplifier, and the interpretation is written into the paper's discussion section.

### 3.4 Case Study CS3: DP vs. No-DP Comparison

**Task description.** CS3 establishes the empirical upper bound on membership inference risk by running FedMIA against an FL model trained without any differential privacy (epsilon=infinity, sigma=0). This is compared against the strongest DP condition in the primary sweep (epsilon=0.1) and the weakest (epsilon=5.0) to span the full range.

**Motivation.** An AUC-ROC result becomes interpretable only when it can be compared against reference points. The no-DP condition establishes the maximum achievable attack performance given the dataset, model architecture, and attack methodology — the ceiling of the threat. The epsilon=0.1 condition establishes the floor (the strongest DP budget tested). CS3 thus provides the empirical context necessary for interpreting the primary sweep results: an AUC-ROC of 0.52 at epsilon=1.0 is only meaningful if we know whether the no-DP AUC-ROC is 0.55 (suggesting the attack is weak regardless of DP) or 0.85 (suggesting DP is highly effective).

**Protocol.** The no-DP condition is run with sigma=0 and gradient clipping disabled, at (rounds=500) to match CS2's round count. The PrivacyAuditor class's noise injection is bypassed via a configuration flag (dp_enabled: false) rather than by modifying code, ensuring the no-DP model is otherwise identical to the DP models. Model utility (reconstruction MSE on the held-out test set) is also reported for the no-DP and all epsilon conditions, to characterise the privacy-utility trade-off quantitatively.

**Acceptance criteria.** No-DP AUC-ROC is reported with confidence interval. A privacy-utility Pareto frontier plot is produced, with AUC-ROC on the x-axis (lower is better, i.e., more private) and test MSE on the y-axis (lower is better, i.e., more useful). The plot is included in the paper's results section.

### 3.5 Paper Writing

The DSN 2027 paper is structured as an eight-section conference paper targeting the venue's typical 11-page limit (excluding references). Each section is assigned to a writing milestone.

**Introduction.** The introduction opens with the EV charging privacy problem statement, motivates FL as a partial solution, identifies the MIA gap, and states the four research questions from Section 1.2. It closes with a contributions paragraph mirroring Section 1.4. Length: approximately 1 page. The introduction must be written last, after all results are known, to ensure that the claims precisely match the findings.

**Related Work.** Four related work subsections: (a) Membership Inference Attacks in centralised ML [Shokri et al. 2017; Salem et al. 2019; Carlini et al. 2022]; (b) MIA in federated settings [Nasr et al. 2019; Melis et al. 2019; Zari et al. 2021]; (c) Differential Privacy in FL [Abadi et al. 2016; McMahan et al. 2018; Geyer et al. 2017]; (d) EV charging privacy and FL applications [Wen et al. 2022; Buzachis et al. 2023; relevant DSN/CCS/USENIX papers]. The related work section must position ChargeShield-FL as the first framework to combine authentic OCPP/MQTT protocol diversity with an end-to-end MIA evaluation pipeline in the EV domain. Length: approximately 1.5 pages.

**System Model and Threat Model.** Describes the 12-node, 4-cluster topology, the protocol stack, the FL round protocol, and the adversary model. The adversary is modelled as an honest-but-curious aggregator (the semi-honest server model) with access to the aggregate model but not to individual gradient updates, consistent with the FedMIA shadow model attack. The threat model section explicitly states what the adversary does and does not know, following the conventions of the cryptographic threat modelling literature. Length: approximately 1.5 pages.

**Methodology.** Describes ACN-Data, the six features, the autoencoder architecture, FedAvg/FedProx, the Gaussian mechanism, the FedMIA attack pipeline, and the evaluation metrics. Includes the AUC-ROC formula and bootstrap CI procedure. Length: approximately 2 pages.

**Experimental Setup.** Describes Containerlab topology, Docker images, NVFLARE version, hardware specifications (CPU, RAM, storage of the development host), random seeds, and the full (rounds, epsilon) grid. Includes a reproducibility statement: all experiments can be re-run using the archived Docker Compose file and the YAML configurations in the repository. Length: approximately 0.75 pages.

**Results.** Presents the primary sweep results (heatmap and line plots), the statistical analysis (Spearman rho, Wilcoxon test, Cohen's d), CS2 (heterogeneity), CS3 (DP vs. No-DP), and the privacy-utility Pareto frontier. Figures are numbered consecutively. Each figure caption is self-contained. Length: approximately 2 pages.

**Discussion.** Interprets the results with respect to the four research questions. Discusses the practical implications for EV charging operators considering FL deployment. Addresses threats to validity: dataset representativeness, single-site ACN-Data provenance, shadow model access assumptions, and the single-round DP composition accounting. Acknowledges the limitations listed in Section 6 of this roadmap. Length: approximately 1 page.

**Conclusion.** Summarises findings, restates contributions, and identifies the most important future directions (Section 4 of this roadmap). Length: approximately 0.5 pages.

### 3.6 Camera-Ready Preparation

Upon conditional acceptance (expected notification: January 2027, see Section 5), the camera-ready phase involves: (a) incorporating reviewer feedback into the manuscript; (b) verifying that all figures meet DSN's resolution and font-size requirements; (c) running the artifact evaluation checklist if the venue offers an artifact evaluation track; (d) archiving the Docker image, YAML configurations, dataset, and trained model checkpoints on Zenodo with a persistent DOI; (e) uploading the camera-ready PDF to the IEEE submission portal by the camera-ready deadline (estimated March 2027). The Zenodo archive DOI is included in the paper's availability statement.

---

## 4. Future Work Beyond Sprint 6

The following work items are out of scope for the DSN 2027 submission but are scientifically important extensions of ChargeShield-FL. They are documented here to preserve the design intent and to guide the next publication cycle.

### 4.1 Scenario 2 MIA: Client-Side Attack

The FedMIA attack implemented in Sprint 4 models the semi-honest server threat: the aggregator observes the aggregate model and attempts to infer membership. This is the weakest adversary model in the FL threat landscape. A more realistic and more dangerous threat is the client-side (or participant-level) attack, in which a malicious participant node observes its own gradient updates and the aggregate model, and uses this richer information to perform membership inference against other participants' training data [Nasr et al. 2019; Lyu et al. 2022].

Implementing Scenario 2 requires: (a) extending FLAREConnector to expose per-round gradient updates to a designated attacker node; (b) implementing a gradient-difference-based membership score (the gap between local and global gradient norms correlates with membership [Nasr et al. 2019]); (c) evaluating the attack under the same (rounds, epsilon) grid as the primary sweep. Scenario 2 is expected to yield substantially higher AUC-ROC than Scenario 1 for equivalent epsilon, because the attacker has access to more information. This result would strengthen the paper's argument for strong DP (small epsilon) as a necessary defence.

The threat model extension also requires revisiting the IDS baselines: CUSUM and Cosine Similarity are designed for server-side anomaly detection, whereas a client-side attacker is behaviourally indistinguishable from a normal participant. A Krum-based defence is the most natural countermeasure, but its effectiveness against Scenario 2 MIA has not been characterised in the EV domain.

### 4.2 ElaadNL Dataset Integration

ACN-Data JPL is a single-site, single-country dataset (Caltech, USA). Integrating the ElaadNL dataset — the Dutch national EV charging dataset, covering hundreds of thousands of charging sessions from public infrastructure across the Netherlands — would serve two purposes: (a) cross-dataset generalisability testing, assessing whether the AUC-ROC relationships observed on ACN-Data replicate on a qualitatively different dataset; and (b) European regulatory relevance, since the EU's GDPR imposes stricter data minimisation requirements than US law, making the privacy analysis more directly actionable for European charging operators.

ElaadNL sessions differ from ACN-Data in several important ways: median session energy is lower (urban public charging dominates), session start times follow different diurnal patterns (later evening peaks in European cities), and the dataset includes categorical attributes (RFID token type, connector type) not present in ACN-Data. The ACNDataset class would need to be extended to an abstract EVDataset interface with ACNDatasetAdapter and ElaadNLAdapter concrete implementations, preserving the six-feature contract while accommodating source-specific preprocessing differences. This refactoring should be designed to maintain backward compatibility with all Sprint 2-5 components.

### 4.3 OCPP 2.0.1 Adapter Completion

The Corporate cluster (50 kW DC, 3 nodes) is currently simulated using a stub adapter that exposes the AbstractChargingNode interface but does not implement the OCPP 2.0.1 message schema. OCPP 2.0.1 introduces significant changes relative to 1.6: JSON-over-WebSocket replaces SOAP, security profiles (TLS with certificate pinning, HTTP Basic Auth) are mandatory, and the message set is substantially expanded (new transaction events, cost detail reporting, V2G smart charging profiles). A complete OCPP 2.0.1 adapter would enable the Corporate cluster to participate in FL rounds with protocol-authentic gradient contributions, which may differ from OCPP 1.6 contributions due to differences in session metering granularity and the availability of additional features (e.g., periodicEventStream data).

The implementation work includes: parsing the OCPP 2.0.1 JSON schema (publicly available from the Open Charge Alliance); implementing the BootNotification, TransactionEvent, MeterValues, and RequestStartTransaction message handlers; and integrating the OCPP 2.0.1 security profiles with the existing mTLS infrastructure. This work is estimated at 3-4 weeks and is the prerequisite for any experiment that claims to model the Corporate cluster authentically.

### 4.4 MQTT v5 Adapter Completion

Similarly, the Residential cluster (7 kW AC, 3 nodes) currently uses a stub. The MQTT v5 protocol, while simpler than OCPP in its message structure, introduces distinct privacy considerations: MQTT's publish-subscribe model means that session data is broadcast to a broker rather than transmitted point-to-point, potentially widening the attack surface for traffic analysis. The MQTT v5 adapter would implement the CONNECT, PUBLISH, and SUBSCRIBE flows for charging session telemetry, using the minutes_available and kwh_requested features (which are particularly prominent in residential sessions due to overnight charging patterns) as the primary FL training signal.

The MQTT v5 adapter's completion would also enable a fifth research question: whether the publish-subscribe topology of the Residential cluster creates additional membership inference risk through broker-side traffic correlation, independent of the FL gradient channel.

### 4.5 Renyi Differential Privacy Composition Accounting

The current PrivacyAuditor uses the standard Gaussian mechanism strong composition bound, which overestimates the privacy cost of multi-round training and thus leads to more noise than strictly necessary for the target epsilon guarantee. Renyi Differential Privacy (RDP) [Mironov 2017] provides a tighter composition analysis: the privacy cost of T rounds of gradient descent with the Gaussian mechanism is computed via the moments accountant, yielding a smaller effective (epsilon, delta) for the same sigma. Implementing RDP composition accounting would allow ChargeShield-FL to report tighter privacy guarantees for multi-round experiments and to explore lower-noise operating points (higher epsilon equivalent per round) that may yield better model utility without sacrificing the target aggregate epsilon.

The implementation would replace the PrivacyAuditor.compute_noise_std() method with an RDP accountant that tracks the per-round Renyi divergence and converts to (epsilon, delta) at evaluation time, following the Opacus library's implementation as a reference [Yousefpour et al. 2021]. This change is backward compatible: existing experiments can be re-evaluated with tighter bounds without rerunning FL training.

### 4.6 Multi-Round Composition Analysis

Related to RDP accounting, a dedicated multi-round composition analysis would characterise how the effective epsilon (the aggregate privacy budget consumed across all FL rounds) grows as a function of T for fixed per-round noise sigma. This is a theoretical contribution: producing a closed-form or numerically tight bound on epsilon_total(T, sigma, delta) for the FedMIA threat model, incorporating the moments accountant, and comparing it against the empirically observed AUC-ROC growth across the (rounds, epsilon) grid. If the empirical AUC-ROC saturates before epsilon_total reaches the theoretical upper bound, this would suggest that the Gaussian mechanism is conservative for this dataset and attack combination — a finding with practical significance for charging network operators.

### 4.7 Real Hardware Validation

All experiments in Sprints 1-6 are conducted in software simulation: Docker containers emulate charging nodes, and synthetic OCPP messages are generated by the ChargingNode class rather than by physical EVSE hardware. A real hardware validation experiment — deploying one physical EVSE node (a commercially available unit with an OCPP 1.6 client, such as an OpenEVSE controller) connected to the Containerlab topology — would address the most fundamental threat to external validity in the current design.

The experiment would require: (a) procuring an OpenEVSE or equivalent unit; (b) configuring its OCPP 1.6 client to connect to the Containerlab-hosted Central System container; (c) running a representative set of real charging sessions (or replaying ACN-Data sessions via the OpenEVSE API); and (d) verifying that the gradient contributions from the hardware node are statistically indistinguishable from those of the software-simulated nodes, as measured by cosine similarity and L2 norm distribution. If significant discrepancies are found, they would motivate further work on calibrating the simulation to real hardware behaviour. This validation step would substantially strengthen the system model section's credibility in any future submission claiming practical deployability.

---

## 5. DSN 2027 Submission Timeline

The following timeline is derived from historical DSN submission deadlines (DSN 2024 and 2025 patterns) and projected forward to 2027. All dates should be confirmed against the official DSN 2027 Call for Papers when published (expected: approximately September 2026).

| Milestone | Target Date | Owner | Dependencies |
|---|---|---|---|
| Full sweep execution complete | 2026-07-31 | Research team | NVFLARE cluster available |
| Statistical analysis complete | 2026-08-15 | Research team | Full sweep data |
| CS2 (heterogeneity) complete | 2026-08-22 | Research team | Full sweep complete |
| CS3 (DP vs. No-DP) complete | 2026-08-29 | Research team | Full sweep complete |
| Results section draft | 2026-09-05 | Research team | All experiments complete |
| Related work section draft | 2026-09-12 | Research team | — |
| Methodology section draft | 2026-09-19 | Research team | — |
| System model and threat model draft | 2026-09-26 | Research team | — |
| Discussion section draft | 2026-10-03 | Research team | Results section |
| Introduction and conclusion draft | 2026-10-10 | Research team | All sections |
| Internal full draft review | 2026-10-17 | All authors | Full draft |
| Revised full draft | 2026-10-31 | All authors | Internal review |
| Abstract submission (if required) | **2026-11-07** | Corresponding author | Revised draft |
| Full paper submission | **2026-11-14** | Corresponding author | Revised draft |
| Rebuttal period (estimated) | 2027-01-10 to 2027-01-24 | All authors | Reviewer comments |
| Acceptance notification (estimated) | **2027-01-31** | — | — |
| Camera-ready deadline (estimated) | **2027-03-14** | All authors | Notification |
| Zenodo archive upload | 2027-03-10 | Research team | Camera-ready |
| DSN 2027 Conference (estimated) | **2027-06-22 to 2027-06-25** | — | — |

**Critical path.** The full sweep execution (target: 2026-07-31) is the single constraint that gates all downstream milestones. Any delay in the sweep — due to hardware failure, NVFLARE job errors, or dataset processing issues — propagates directly to the paper submission deadline. The research team should prioritise resolving blockers in the sweep before beginning paper writing.

**Buffer allocation.** Two weeks of buffer are built into the timeline between the last experiment (CS3 complete: 2026-08-29) and the results section draft (2026-09-05). An additional two-week buffer separates the revised full draft (2026-10-31) from the abstract submission (2026-11-07). These buffers are intended to absorb unexpected results that require additional investigation or re-experimentation.

**Contingency.** If the full sweep is not complete by 2026-08-15, the team will consider reducing the rounds grid to {100, 200, 500} (dropping the 1,000-round condition, which is the most time-consuming) and framing this as a limitation. This contingency preserves the qualitative shape of the epsilon-AUC-ROC curve while reducing experimental burden. The decision threshold for invoking this contingency is: if fewer than 24 of the 40 conditions are complete by 2026-08-08, the 1,000-round conditions are deprioritised.

---

## 6. Known Limitations to Address Before Submission

The following limitations are acknowledged and must be explicitly addressed — either by additional experiments, by revised framing, or by explicit disclosure in the paper's discussion section — before the DSN 2027 submission.

### 6.1 Single-Site Dataset Provenance

ACN-Data JPL is collected from a single location (Jet Propulsion Laboratory, Pasadena, California) over two calendar years (2019-2020). The charging population is dominated by a particular demographic (technical professional workforce with EVs) and a particular infrastructure type (workplace charging, Level 2 AC). This limits the generalisability of the findings to other deployment contexts — public fast charging, residential overnight charging, or European charging infrastructure with different diurnal patterns. The paper's discussion section must explicitly state this limitation and frame the ElaadNL integration (Section 4.2) as the future work that would address it. The claims in the paper should be scoped to "workplace FL deployment" rather than "EV charging in general."

### 6.2 Single-Round DP Composition (Gaussian Bound)

As noted in Section 4.5, the current PrivacyAuditor implements the single-round Gaussian mechanism strong composition bound. For the 1,000-round condition, this bound may significantly overestimate the actual privacy cost — meaning the reported epsilon corresponds to a stronger privacy guarantee than the sigma-noise level actually provides over 1,000 rounds. Conversely, if the bound is used to set sigma for a target epsilon, it sets more noise than strictly necessary. The paper must acknowledge this as a methodological limitation and either (a) implement RDP accounting before submission, or (b) clearly label all reported epsilon values as "per-round epsilon" rather than "aggregate epsilon," and include a footnote explaining the composition gap.

### 6.3 Semi-Honest Server Threat Model

The FedMIA attack assumes a semi-honest (honest-but-curious) aggregator: the server follows the protocol faithfully but attempts to infer membership from the aggregate model. This is a conservative adversary model. Stronger adversaries — a malicious aggregator who manipulates the aggregation to amplify membership signals, or a colluding participant who shares gradient information with the aggregator — are not modelled. The paper must clearly state the threat model assumptions and acknowledge that the reported AUC-ROC values are lower bounds on what a stronger adversary could achieve.

### 6.4 Autoencoder Architecture Simplicity

The 6->16->8->4 autoencoder is a relatively shallow model for the 6-dimensional input space. A more expressive architecture (deeper encoder, variational autoencoder, or normalising flow) might yield higher reconstruction fidelity for member records, thereby increasing the attack's AUC-ROC and making the results more conservative (more favourable to the attacker). The paper should include a sensitivity analysis: re-running FedMIA with a deeper autoencoder on the no-DP condition (CS3) to bound the impact of architecture choice on the reported AUC-ROC ceiling.

### 6.5 Protocol Stub Completeness

As described in Sections 4.3 and 4.4, the OCPP 2.0.1 (Corporate) and MQTT v5 (Residential) adapters are stubs. Experiments involving these clusters do not reflect authentic protocol behaviour. For the DSN 2027 submission, this limitation must be explicitly disclosed: the 12-node topology is complete at the FL level, but protocol-level authenticity is limited to OCPP 1.6 (Highway and Urban clusters). Claims about "full topology" experiments should be qualified accordingly.

### 6.6 Hardware Validation Gap

All results are obtained from software simulation on a single development host running OrbStack. The computational characteristics of Docker containers may differ from real EVSE hardware in ways that affect gradient magnitudes, training time, and convergence behaviour. Until the real hardware validation experiment (Section 4.7) is complete, the paper should describe the system as a "high-fidelity simulation" rather than a "deployed system." The Containerlab topology and Docker container specifications should be included in the paper's experimental setup section with sufficient detail for an independent research team to replicate the environment.

### 6.7 Statistical Power

With 40 experimental conditions and 13,073 sessions split across 12 nodes, the effective sample size per node per round is approximately 1,089 sessions (13,073 / 12). At small epsilon (strong DP), the signal-to-noise ratio in the gradient updates may be too low for the bootstrap CI procedure to achieve adequate statistical power to distinguish AUC-ROC from 0.5 at the 95% level. A power analysis should be conducted before the submission: for the expected AUC-ROC range (0.50-0.60 based on the Sprint 5 pilot), compute the minimum number of sessions per node required to detect a 0.02 AUC-ROC effect with 80% power. If the current dataset is underpowered for small epsilon conditions, this should be disclosed and the null results at small epsilon should be described as "failure to reject" rather than "evidence of privacy."

---

## 7. References

Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., and Zhang, L. (2016). Deep learning with differential privacy. *Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (CCS '16)*, pp. 308-318. ACM.

Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B (Methodological)*, 57(1):289-300.

Blanchard, P., El Mhamdi, E. M., Guerraoui, R., and Stainer, J. (2017). Machine learning with adversaries: Byzantine tolerant gradient descent. *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30.

Buzachis, A., Galletta, A., Celesti, A., Fazio, M., and Villari, M. (2023). Towards privacy-preserving federated learning for smart EV charging systems. *IEEE Access*, 11:45921-45933.

Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., and Tramer, F. (2022). Membership inference attacks from first principles. *2022 IEEE Symposium on Security and Privacy (S&P '22)*, pp. 1897-1914. IEEE.

Dwork, C., McSherry, F., Nissim, K., and Smith, A. (2006). Calibrating noise to sensitivity in private data analysis. *Proceedings of the Third Conference on Theory of Cryptography (TCC 2006)*, Lecture Notes in Computer Science, vol. 3876, pp. 265-284. Springer.

Dwork, C. and Roth, A. (2014). The algorithmic foundations of differential privacy. *Foundations and Trends in Theoretical Computer Science*, 9(3-4):211-407.

Geyer, R. C., Klein, T., and Nabi, M. (2017). Differentially private federated learning: A client level perspective. *arXiv preprint arXiv:1712.07557*.

Lee, Z. J., Li, T., and Low, S. H. (2019). ACN-Data: Analysis and applications of an open EV charging dataset. *Proceedings of the Tenth ACM International Conference on Future Energy Systems (e-Energy '19)*, pp. 139-149. ACM.

Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Smola, A., and Smith, V. (2020). Federated optimization in heterogeneous networks. *Proceedings of Machine Learning and Systems (MLSys 2020)*, 2:429-450.

Lyu, L., Yu, H., and Yang, Q. (2022). Threats to federated learning: A survey. *arXiv preprint arXiv:2003.02133*.

McMahan, H. B., Moore, E., Ramage, D., Hampson, S., and y Arcas, B. A. (2017). Communication-efficient learning of deep networks from decentralized data. *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS 2017)*, PMLR 54:1273-1282.

McMahan, H. B., Ramage, D., Talwar, K., and Zhang, L. (2018). Learning differentially private recurrent language models. *International Conference on Learning Representations (ICLR 2018)*.

Melis, L., Song, C., De Cristofaro, E., and Shmatikov, V. (2019). Exploiting unintended feature leakage in collaborative learning. *2019 IEEE Symposium on Security and Privacy (S&P '19)*, pp. 691-706. IEEE.

Mironov, I. (2017). Renyi differential privacy. *2017 IEEE 30th Computer Security Foundations Symposium (CSF '17)*, pp. 263-275. IEEE.

Nasr, M., Shokri, R., and Houmansadr, A. (2019). Comprehensive privacy analysis of deep learning: Passive and active white-box inference attacks against centralized and federated learning. *2019 IEEE Symposium on Security and Privacy (S&P '19)*, pp. 739-753. IEEE.

Salem, A., Zhang, Y., Humbert, M., Berrang, P., Fritz, M., and Backes, M. (2019). ML-Leaks: Model and data independent membership inference attacks and defenses on machine learning models. *Proceedings of the 2019 Network and Distributed System Security Symposium (NDSS '19)*. Internet Society.

Shokri, R., Stronati, M., Song, C., and Shmatikov, V. (2017). Membership inference attacks against machine learning models. *2017 IEEE Symposium on Security and Privacy (S&P '17)*, pp. 3-18. IEEE.

Wen, Z., Shi, J., Li, Q., He, B., and Chen, J. (2022). ThunderSTruck: Practical MIA for autonomous vehicle charging. *IEEE Transactions on Smart Grid*, 13(4):3267-3278.

Yousefpour, A., Shilov, I., Ghosh, A., Stuber, P., Bharadwaj, H., Lam, A., Vanderput, J., Bhimrajka, N., Berger, A., Bharat, A., Bhatt, K., Bhandal, J., Bittman, D., Bruss, B., Caron, N., and Colangelo, G. (2021). Opacus: User-friendly differential privacy library in PyTorch. *arXiv preprint arXiv:2109.12298*.

Zari, O., Xu, C., and Neglia, G. (2021). Efficient passive membership inference attack in federated learning. *arXiv preprint arXiv:2111.00430*.

Zhao, Y., Li, M., Lai, L., Suda, N., Civin, D., and Chandra, V. (2018). Federated learning with non-IID data. *arXiv preprint arXiv:1806.00582*.

---

*End of document. ChargeShield-FL Roadmap v1.0 — 2026-06-26.*
