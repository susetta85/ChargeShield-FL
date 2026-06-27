# ChargeShield-FL: Research Roadmap toward DSN 2027

**Project:** ChargeShield-FL — Evaluating Membership Inference Attacks against Federated Learning in Electric Vehicle Charging Infrastructure

**Target Venue:** IEEE/IFIP International Conference on Dependable Systems and Networks (DSN 2027)

**Document Version:** 1.0 — June 2026

---

## 1. Research Vision

The intersection of Federated Learning (FL) and Operational Technology (OT) environments presents one of the most underexamined frontiers in contemporary security research. While the academic community has made substantial progress in understanding privacy vulnerabilities in FL within conventional information technology (IT) contexts — including healthcare records, financial transactions, and natural language corpora — the specific risks posed by FL deployments in critical infrastructure remain poorly characterized. ChargeShield-FL addresses precisely this gap. Existing literature on Membership Inference Attacks (MIA) against FL focuses almost exclusively on centralized or cloud-federated settings, with data that conforms to well-studied distributions and with adversaries whose capabilities are calibrated against IT-grade systems. OT environments, by contrast, impose fundamentally different constraints: nodes are heterogeneous in protocol, power, and connectivity; data distributions are physically determined by geography, vehicle fleet type, and charging behavior; and the operational consequence of privacy breach extends beyond data confidentiality to potential physical disruption of energy delivery infrastructure. The absence of rigorous privacy evaluation frameworks designed for such environments constitutes a meaningful gap in the state of the art, one that ChargeShield-FL is specifically designed to fill.

Electric vehicle charging infrastructure is not merely a convenient case study — it is a timely and strategically important one. The rapid deployment of EV charging networks across Europe, North America, and Asia is reshaping the energy grid at the edge. Charging management systems increasingly rely on machine learning to perform anomaly detection, load forecasting, and demand response, and the privacy-sensitive nature of individual charging sessions is becoming a regulatory concern under frameworks such as the EU General Data Protection Regulation and the proposed EU Cyber Resilience Act. A single charging session encodes rich behavioral information: the time of day at which a vehicle is connected, the duration of the session, the peak power draw, and the geographic location of the charging point can collectively be used to infer home address, work schedule, and travel patterns. When this data is used to train local FL models, the gradient updates transmitted to the aggregator may, under certain conditions, leak information about individual training records. Demonstrating and quantifying this risk — and identifying the conditions under which Differential Privacy (DP) effectively mitigates it — is a contribution of direct practical relevance to infrastructure operators, regulators, and security architects.

Beyond the specific domain of EV charging, the field stands to benefit from a modular, reproducible evaluation framework for MIA in OT-federated settings. Reproducibility remains a persistent challenge in applied ML security research: experimental results frequently depend on undisclosed implementation choices, proprietary datasets, or infrastructure configurations that cannot be replicated by independent researchers. ChargeShield-FL is designed from the outset to be fully reproducible, with containerized components, declarative topology definitions, public datasets, and version-pinned dependencies. Its modular architecture allows researchers to substitute alternative attack algorithms, FL aggregation strategies, or DP mechanisms without modifying the core framework. The ML Plane abstraction — a novel logical layer that provides transversal visibility into FL traffic across the Purdue Model hierarchy without requiring modification of existing OT protocols or devices — constitutes a reusable architectural contribution that extends beyond this specific instantiation. By publishing both the empirical results and the framework itself, ChargeShield-FL enables the broader research community to extend, challenge, and build upon this work in a principled and verifiable manner.

---

## 2. Research Questions

### RQ1: FedMIA Effectiveness Across DP Privacy Budgets

**Formal Statement:** How effective is FedMIA against Federated Learning in EV charging networks across different differential privacy budgets (ε ∈ {0.1, 0.5, 1.0, 2.0, 5.0})?

The fundamental question motivating this research is whether the shadow model-based membership inference attack of Shokri et al. (2017), as instantiated in FedMIA, is capable of distinguishing training members from non-members in a realistic FL deployment with realistic privacy budgets. The DP budget ε is the primary control variable: at ε = 0.1, privacy guarantees are strong but utility may be significantly degraded; at ε = 5.0, utility is preserved but the privacy guarantee is correspondingly weaker. The expected outcome, grounded in the theoretical guarantees of the Gaussian Mechanism, is a monotonic relationship between ε and AUC-ROC, with attack performance approaching random guessing (AUC-ROC ≈ 0.5) at sufficiently small ε. However, the theoretical guarantee does not immediately translate into practical attack failure — intermediate ε values may yield partial information leakage that is measurable even when formal DP bounds are technically satisfied. Characterizing this empirical relationship is essential for practitioners who must calibrate ε for real deployments.

### RQ2: Data Heterogeneity and MIA Effectiveness

**Formal Statement:** Does data heterogeneity (non-IID distribution across four cluster types) affect MIA effectiveness beyond what DP alone predicts?

Federated Learning over heterogeneous data distributions — the so-called non-IID setting — is known to affect model convergence and gradient dynamics in ways that differ qualitatively from the IID case. What is less well understood is whether non-IID data distributions also affect the information leakage profile of gradient updates in ways that are not captured by standard DP analysis. In ChargeShield-FL, data is partitioned by cluster type: Highway nodes (150 kW DC fast charging) produce sessions with short duration and high peak power; Urban nodes (22 kW AC) produce moderate-duration sessions; Residential nodes (7 kW AC) produce long overnight sessions; and Corporate nodes (50 kW DC) produce structured workday patterns. These distributions differ not only in scale but in behavioral pattern. It is plausible that nodes whose local data distribution is highly distinctive — particularly Residential nodes, whose charging patterns are most temporally concentrated — produce gradient updates that are differentially informative to an attacker. This research question asks whether the within-cluster and between-cluster heterogeneity of the EV charging domain modifies the empirical attack surface in ways that DP parameterization alone does not predict.

### RQ3: Minimum Effective ε for Privacy Indistinguishability

**Formal Statement:** What is the minimum ε such that FedMIA performance becomes statistically indistinguishable from random guessing (AUC-ROC ≈ 0.5)?

This research question seeks to identify a practically meaningful threshold — a value of ε below which the attack yields no statistically significant advantage over random guessing, as determined by confidence intervals estimated over repeated experimental runs with different random seeds. The answer to RQ3 has direct policy implications: it provides infrastructure operators with a concrete target privacy budget for deployment, grounded not in theoretical worst-case analysis but in empirically measured attack performance in a realistic topology. The identification of such a threshold also enables a meaningful discussion of the privacy/utility trade-off addressed in RQ5. It is important to note that the threshold identified may be dataset-specific and topology-specific; the contribution is not to claim universal applicability but to demonstrate the methodology by which such a threshold can be rigorously identified for a given deployment context.

### RQ4: IDS Detectability of Passive MIA Adversaries

**Formal Statement:** Do behavioral IDS mechanisms (CUSUM, Krum, Cosine Similarity) detect passive MIA adversaries, and if not, what does this imply for the architecture of future defenses?

This research question is motivated by the practical observation that many FL deployments in industrial and critical infrastructure contexts incorporate anomaly detection at the aggregation layer, implemented via statistical or Byzantine-fault-tolerant mechanisms. CUSUM (Cumulative Sum control chart) detects statistical changes in update distributions over time; Krum identifies and excludes outlier gradient updates under a Byzantine threat model; Cosine Similarity measures the directional alignment of client updates relative to the global model or to each other. These mechanisms are designed to detect active adversaries — those who inject malicious gradients, execute model poisoning, or attempt to bias the global model. A passive MIA adversary, by contrast, makes no modification to the FL protocol: they observe gradient updates as they arrive at the aggregator and apply a shadow model analysis offline. The hypothesis — which the experiment is designed to confirm or refute — is that passive MIA adversaries are behaviorally indistinguishable from benign clients and are therefore undetectable by any behavioral IDS mechanism. If confirmed, this constitutes a publishable negative result with clear implications: behavioral defenses are categorically insufficient against passive inference attacks, and effective countermeasures must operate in the gradient space (e.g., secure aggregation) or at the cryptographic level.

### RQ5: Utility Preservation at the ε Threshold Identified in RQ3

**Formal Statement:** Is the FL model utility — as measured by reconstruction error and anomaly detection accuracy — preserved at the ε threshold identified in RQ3?

The central trade-off in privacy-preserving machine learning is that stronger privacy guarantees, achieved through larger noise perturbations, necessarily degrade model utility. In the context of ChargeShield-FL, the FL model is an autoencoder trained to perform anomaly detection: a model that fails to accurately reconstruct normal charging sessions, or that cannot discriminate between normal and anomalous sessions, is operationally useless. RQ5 asks whether the ε threshold at which FedMIA is defeated is compatible with operationally acceptable model utility. If the answer is affirmative, the research provides a directly actionable recommendation: deploy DP-FL at ε ≤ ε* with confidence that both privacy and utility are simultaneously achieved. If the answer is negative — if privacy requires a noise level that renders the model unusable — the research identifies a fundamental tension that motivates future work in privacy-preserving architectures beyond standard DP-SGD.

### RQ6: FedProx vs. FedAvg Under Non-IID Conditions

**Formal Statement:** How does FedProx compare to FedAvg in terms of MIA vulnerability under non-IID data conditions?

FedProx introduces a proximal term in the local objective function that penalizes deviations from the global model, improving convergence stability in heterogeneous settings. However, this regularization also affects the gradient dynamics of local training in a way that may modify the information content of transmitted updates. If FedProx produces updates that are more uniform across clients — because the proximal term pulls all local updates toward the global model — the resulting gradient updates may be less individually distinctive and hence less susceptible to membership inference. Conversely, if the proximal correction introduces systematic structure that differs between members and non-members of the training set, it may amplify the attack surface. This research question adds theoretical depth to the experimental contribution by connecting algorithmic FL design choices to empirical privacy outcomes.

---

## 3. Scientific Contributions

### C1: The ML Plane Abstraction

The ML Plane is a novel logical abstraction introduced by ChargeShield-FL to address a fundamental architectural gap in FL deployments within OT environments. The Purdue Model — or its modern interpretation in IEC 62443 — partitions industrial control systems into hierarchical levels ranging from physical process instrumentation (Level 0) through field controllers (Level 1), supervisory control (Level 2), manufacturing operations (Level 3), and enterprise integration (Level 4 and above). Traditional security monitoring architectures in OT environments operate within this hierarchy: sensors and intrusion detection systems are deployed at specific levels and observe traffic at those levels. Federated Learning, however, does not respect this hierarchy in a natural way: gradient updates originate at Level 1 or Level 2 devices (the FL clients), traverse the network, and are aggregated at a server that may reside at Level 3 or in the enterprise zone. The ML traffic thus crosses level boundaries in ways that existing monitoring architectures are not designed to observe holistically.

The ML Plane resolves this tension by defining a transversal logical layer that cuts across the Purdue Model hierarchy and is dedicated to the observation, recording, and analysis of FL-related traffic — specifically, gradient updates, model distributions, and aggregation events. It is implemented using the observer pattern: ML Plane components attach to communication endpoints without modifying the behavior of the underlying OT protocols (OCPP 1.6, OCPP 2.0.1, MQTT v5) or the FL training protocol (FedAvg, FedProx). This design principle — non-intrusive observation — is critical in OT environments where protocol modification carries risk of disrupting real-time control operations. The ML Plane is not a physical network segment but a logical overlay; its observability is implemented through mirroring, tap interfaces, and sidecar containers in the Containerlab emulation environment.

The novelty of the ML Plane abstraction lies in its deliberate positioning at the intersection of OT security architecture and FL system design — two communities that have, to date, largely operated independently. Prior work on FL security assumes IT-grade infrastructure and does not address the hierarchical, protocol-heterogeneous topology of OT environments. Prior work on OT security monitoring does not address the specific characteristics of FL traffic or the privacy risks it introduces. The ML Plane is the architectural bridge between these two bodies of work, and its specification as a reusable abstraction — independent of any specific OT vendor, FL framework, or domain — constitutes a contribution to both communities.

### C2: ChargeShield-FL as a Modular, Reproducible Evaluation Framework

ChargeShield-FL's second primary contribution is the framework itself, conceived and implemented as a reusable scientific instrument for the evaluation of MIA against FL in OT environments. The framework is distinguished from prior FL privacy evaluation tools by three design principles: modularity, realism, and reproducibility. Modularity means that individual components — the FL clients, the aggregator, the attack module, the IDS monitors, the ML Plane observer — are independently deployable and interchangeable. A researcher wishing to evaluate a different attack algorithm, a different FL aggregation strategy, or a different DP mechanism can substitute the relevant component without modifying the rest of the framework. Realism means that the emulated topology faithfully reproduces the protocol heterogeneity, network topology, and data characteristics of a real EV charging deployment, rather than approximating them with simplified simulations. Reproducibility means that every experimental parameter — FL rounds, privacy budget, random seeds, dataset splits — is declared in version-controlled YAML configuration files, and that the entire experimental pipeline can be reproduced by executing a single Makefile target against publicly available data and pinned Docker images.

The significance of this contribution extends beyond the EV charging domain. Existing FL privacy evaluation frameworks — including PySyft, TensorFlow Federated, and Flower — are primarily designed for ML research in IT contexts and do not provide native support for OT protocol emulation, hierarchical network topologies, or the integration of OT-specific IDS mechanisms. ChargeShield-FL demonstrates that such integration is feasible, principled, and scientifically productive. By publishing the framework alongside the experimental results, this work enables future researchers to instantiate the same evaluation methodology for different OT domains — industrial control systems, water distribution networks, building automation — simply by providing a domain-appropriate dataset and topology definition.

The framework's scientific defensibility rests on its grounding in real operational data (ACN-Data JPL, 13,073 real EV sessions), its use of production-grade FL infrastructure (NVFLARE 2.7.2), and its faithful emulation of production-grade security controls (mTLS, WireGuard, OCPP security profiles). These design choices ensure that the experimental results are not artifacts of simplified simulation but reflect, to the greatest extent possible within an academic setting, the privacy risks that would be observed in a real-world deployment.

### C3: Empirical Evaluation of FedMIA Under Variable DP Budgets in a Heterogeneous OT Topology

The third contribution is the empirical evaluation itself: a systematic sweep of the ε × rounds parameter space, producing an evidence-based characterization of FedMIA attack effectiveness as a function of privacy budget and training duration in a realistic, heterogeneous OT topology. The first experimental result — AUC-ROC = 0.5172 at ε = 1.0 and 100 rounds — already suggests that DP at moderate budgets is effective in this deployment context. The full sweep will produce an ε vs. AUC-ROC curve across five privacy budgets and four training durations, providing the empirical foundation for answering RQ1 through RQ3.

This contribution is novel because no prior work has conducted such a systematic empirical evaluation in an OT context with realistic protocol heterogeneity and real operational data. The existing literature on MIA in FL is primarily evaluated on benchmark image datasets (MNIST, CIFAR-10) or synthetic tabular data, in topologies that consist of homogeneous IT nodes. The EV charging context introduces qualitatively different data characteristics — sparse, seasonally variable, geographically correlated time series of energy sessions — that may interact with the attack mechanism in ways not predicted by prior results. The non-IID partitioning across four cluster types with distinct physical characteristics further distinguishes this evaluation from prior work.

### C4: Negative Result — Passive MIA is Undetectable by Behavioral IDS

The fourth contribution is a publishable negative result: the empirical demonstration that behavioral IDS mechanisms — CUSUM, Krum, and Cosine Similarity — are categorically unable to detect a passive MIA adversary operating at the aggregator. This result, if confirmed, is not a failure of the experimental design but a scientifically significant finding with clear implications for defense architecture. The value of negative results in security research is well recognized: the demonstration that a widely deployed class of defenses is ineffective against a specific threat category is a contribution that redirects future research effort toward more promising approaches.

The result is defensible because it flows from a structural argument, not merely from empirical observation. Passive MIA adversaries do not modify their gradient updates; they do not inject anomalous values; they do not deviate from the expected FL protocol in any observable way. CUSUM detects distributional shifts over time; Krum detects outlier updates; Cosine Similarity detects directional anomalies. None of these mechanisms have any basis for distinguishing a malicious passive observer from a benign FL client, because the adversary's behavior, from the perspective of the FL protocol, is identical to that of a benign client. The empirical confirmation of this structural argument strengthens the case and quantifies the detection gap precisely. The implication — that future defenses must operate at the gradient level (secure aggregation, homomorphic encryption) or through formal privacy mechanisms (DP) rather than through behavioral anomaly detection — is a clear and actionable direction for future work.

---

## 4. System Model

### 4.1 Network Topology

The ChargeShield-FL system model consists of twelve nodes organized into four clusters, emulated using Containerlab on an OrbStack virtualization backend. Each cluster represents a distinct EV charging deployment context, differentiated by charging power level, communication protocol, and data distribution characteristics. The twelve nodes are interconnected via a logical aggregation network through which FL gradient updates are transmitted to a central NVFLARE server. All inter-node and node-to-server communication is authenticated using mutual TLS (mTLS), ensuring that only authorized nodes may participate in the FL protocol. Tunneled traffic between clusters is secured via WireGuard, providing network-layer confidentiality and integrity for gradient updates in transit. The Containerlab topology is declared as a YAML file, enabling deterministic instantiation of the network with all link parameters, IP addressing, and security configurations fully specified.

### 4.2 Node Capabilities and Cluster Profiles

The four clusters correspond to operationally distinct EV charging environments. The Highway cluster consists of three nodes, each representing a high-power DC fast-charging station rated at 150 kW. These nodes communicate using OCPP 1.6 and are characterized by short, high-energy sessions typical of intercity travel stops. The Urban cluster consists of three nodes representing public AC charging points rated at 22 kW, communicating via OCPP 1.6, with session patterns characteristic of urban mobility — moderate duration, moderate energy, variable time of day. The Residential cluster consists of three nodes representing home charging wallboxes rated at 7 kW, communicating via MQTT v5, with session patterns dominated by overnight charging and high temporal regularity. The Corporate cluster consists of three nodes representing workplace charging stations rated at 50 kW DC, communicating via OCPP 2.0.1, with sessions concentrated in business hours and exhibiting a weekly periodicity correlated with working days. Each cluster type introduces a distinct local data distribution, making the overall federation non-IID in a manner grounded in physical operational reality.

### 4.3 Federated Learning Protocol

The FL protocol is implemented using NVFLARE 2.7.2, a production-grade federated learning framework developed by NVIDIA. Training proceeds in synchronous rounds: at the beginning of each round, the NVFLARE server distributes the current global model to all participating clients; each client performs a fixed number of local training steps on its local dataset; each client transmits its updated local model parameters to the server; the server aggregates the received updates using either FedAvg or FedProx; and the resulting global model is stored as the starting point for the next round. The FedAvg aggregation computes a weighted average of client updates, with weights proportional to local dataset sizes. The FedProx aggregation adds a proximal regularization term μ||w - w_global||² to each client's local objective, where w_global is the current global model and μ is a hyperparameter controlling the strength of regularization. The experiment sweeps over round counts in {100, 200, 500, 1000} to characterize the effect of training duration on attack effectiveness and model utility.

### 4.4 Communication Model

The communication architecture distinguishes between the control plane and the data plane. The control plane — through which FL orchestration messages, round synchronization signals, and administrative commands are transmitted — is secured using mTLS with per-node client certificates issued by a private certificate authority instantiated for the experiment. The data plane — through which gradient updates and model distributions are transmitted — is additionally protected by WireGuard tunnels between cluster gateways, providing network-layer encryption independent of the application-layer FL protocol. At the application layer, OT protocol stacks are preserved: OCPP 1.6 nodes communicate with the charging management system over WebSocket; OCPP 2.0.1 nodes use WebSocket with the OCPP 2.0.1 security profile; MQTT v5 nodes use TLS-secured MQTT connections. The ML Plane observer attaches to these communication endpoints via network taps, recording FL traffic metadata without modifying the underlying protocol flows.

### 4.5 Data Distribution

The dataset used in all experiments is ACN-Data JPL, covering the 2019 and 2020 charging seasons at the Jet Propulsion Laboratory EV charging facility in Pasadena, California. The dataset comprises 13,073 complete charging sessions, each described by six features: energy delivered in kilowatt-hours (energy_kwh), session duration in hours (duration_hours), peak charging power in kilowatts (peak_power_kw), session start hour (start_hour, integer 0–23), day of week (day_of_week, integer 0–6), and cluster identifier (cluster_id, categorical). The dataset is partitioned non-IID by cluster: each cluster's nodes receive only those sessions that correspond to the physical characteristics of that cluster type, determined by power level and session duration profile. This partitioning reflects the operational reality that a Highway fast-charger would not observe residential overnight sessions, and vice versa. All features are normalized to the unit interval prior to training.

### 4.6 ML Model Architecture

The FL model deployed at each node is a symmetric autoencoder implemented in PyTorch. The encoder maps the six-dimensional input through two hidden layers of dimensions 16 and 8, reducing to a four-dimensional bottleneck representation. The decoder symmetrically expands from four to eight to sixteen dimensions and reconstructs the original six-dimensional input. All hidden layers use ReLU activation functions; the output layer uses a sigmoid activation to constrain reconstructed values to the unit interval. Training employs mean squared error (MSE) loss, which is natural for tabular reconstruction tasks and provides an interpretable anomaly score: sessions with reconstruction error exceeding a threshold calibrated on the training distribution are flagged as anomalous. The model is small by design, reflecting the compute constraints of edge OT devices and ensuring that training is feasible within the round budget even on CPU-only containerized nodes.

---

## 5. Threat Model

### 5.1 Attacker Goal

The attacker's objective is to determine, for a target EV charging session s, whether s was included in the local training dataset of a victim FL client. This is the classical membership inference problem as formalized by Shokri et al. (2017). In the context of EV charging, successful membership inference reveals that a specific session — characterized by its time, duration, energy, and location cluster — was used to train the local model of a specific charging node. This information, aggregated over multiple sessions, enables an adversary to reconstruct behavioral profiles of individual EV users associated with a given charging node, constituting a privacy violation potentially actionable under data protection regulations.

### 5.2 Attacker Capabilities

The attacker is modeled as a passive, honest-but-curious aggregator. This means that the attacker faithfully executes the FL aggregation protocol — distributing model updates to clients, receiving gradient updates, computing the weighted average — but additionally records and analyzes all received gradient updates for the purpose of membership inference. The attacker has access to all gradient updates transmitted by all participating clients over the entire training duration. The attacker does not modify gradient updates, inject malicious content into the FL protocol, or communicate with any party other than through the standard FL protocol. The honest-but-curious model is appropriate for an aggregator that is a commercial service provider with contractual obligations to clients but potential economic or regulatory incentives to exploit the data.

### 5.3 Attacker Knowledge

The attacker operates with the following background knowledge: a shadow dataset drawn from the same distribution as the target dataset, available to the attacker from public sources (in this case, ACN-Data itself is publicly available, which is a deliberate and realistic choice — an attacker with domain knowledge would have access to publicly available EV charging datasets); knowledge of the FL protocol, including the aggregation algorithm, the round count, and the model architecture; and the ability to train shadow models that approximate the behavior of victim clients. The attacker does not know the specific local dataset of any individual victim client; the shadow model approach infers membership by training attack models on the outputs of shadow models trained on known-membership datasets.

### 5.4 Attacker Limitations

The attacker is constrained in the following ways: they may not access the raw local training data of any FL client; they may not inject poisoned gradients or otherwise modify any participant's model updates; they may not execute active man-in-the-middle attacks on the mTLS or WireGuard-secured communication channels; and they may not collude with any FL client. These limitations reflect the honest-but-curious model and are consistent with the standard threat model adopted in the FL privacy literature (McMahan et al. 2018, Nasr et al. 2019).

### 5.5 Trust Assumptions

FL clients — the twelve charging node containers — are fully trusted: they are assumed to execute the FL protocol honestly and are not modeled as potential adversaries in Scenario 1. The aggregator — the NVFLARE server — is semi-trusted: it is trusted to execute the aggregation correctly but is modeled as potentially curious about the contents of individual client updates. Network links are authenticated via mTLS at the application layer and WireGuard at the network layer; the attacker is assumed to not have compromised these cryptographic controls.

### 5.6 Scope and Out-of-Scope Scenarios

The experiments reported in this work address Scenario 1: aggregator-side passive membership inference. In this scenario, the aggregator is the adversary. Scenario 2 — client-side MIA, in which a malicious client infers membership in another client's dataset by observing the global model — is out of scope for the current work. Active attacks — model poisoning, gradient inversion, backdoor injection — are also out of scope. This scoping decision is justified by the fact that the honest-but-curious aggregator is the dominant threat model in the FL privacy literature and corresponds to the primary risk in centralized aggregation architectures, which are the most common deployment pattern in current FL systems. Scenario 2 and active attacks constitute natural extensions for future work.

---

## 6. Experimental Methodology

### 6.1 Dataset Preparation

The ACN-Data JPL dataset covers EV charging sessions recorded at the Jet Propulsion Laboratory facility in Pasadena, California, across 2019 and 2020. After filtering for complete sessions with all required fields present, the dataset contains 13,073 sessions. Six features are extracted for each session: the total energy delivered (energy_kwh), the session duration (duration_hours), the peak power observed during the session (peak_power_kw), the hour at which the session started (start_hour), the day of week of the session (day_of_week), and a cluster identifier (cluster_id) assigned by a pre-processing step that maps sessions to cluster types based on power level and duration profile. All continuous features are normalized to the unit interval using min-max scaling calibrated on the training split. The cluster identifier is ordinally encoded as an integer in {0, 1, 2, 3}.

The dataset is partitioned into training, validation, and test splits at an 80/10/10 ratio, stratified by cluster to ensure that each split contains representative sessions from all cluster types. Non-IID partitioning by cluster is applied after splitting: training sessions are distributed to FL clients such that each cluster's three nodes receive only sessions from their respective cluster type, with sessions randomly divided approximately equally among the three nodes within each cluster. This produces a realistic non-IID federation in which each client's local dataset reflects the behavioral patterns of its specific deployment context.

### 6.2 Federated Learning Training Setup

The FL training is orchestrated by an NVFLARE 2.7.2 server container. Twelve NVFLARE client containers participate, each corresponding to one node in the topology. Training proceeds in synchronous rounds, with all twelve clients participating in each round. The number of local training epochs per round is fixed at five for all experiments. Two aggregation algorithms are evaluated: FedAvg, which computes the weighted average of client updates with weights proportional to local dataset sizes, and FedProx, which augments each client's local objective with a proximal regularization term with μ = 0.01. The experimental sweep covers the Cartesian product of round counts {100, 200, 500, 1000} and privacy budgets {0.1, 0.5, 1.0, 2.0, 5.0}, producing 20 FL training runs per aggregation algorithm (40 total). All runs use a fixed batch size of 64, the Adam optimizer with learning rate 0.001, and a fixed random seed per run to ensure reproducibility.

### 6.3 Attack Execution

FedMIA is implemented following the shadow model methodology of Shokri et al. (2017) as adapted for FL by Nasr et al. (2019). The attack proceeds in three phases. In the shadow training phase, the attacker trains a set of shadow FL models on a shadow dataset — a random half of the ACN-Data corpus disjoint from the target clients' training data — under the same FL protocol (FedAvg/FedProx, same round count) but without DP. The shadow training produces pairs of (gradient update, membership label) for each session in the shadow dataset. In the attack model training phase, a binary classifier (logistic regression or a shallow MLP) is trained on these labeled pairs to predict membership from gradient features derived from reconstruction error: specifically, the attacker observes the reconstruction error of a candidate session under each round's global model and uses the time series of reconstruction errors as the attack feature vector. In the inference phase, the trained attack model is applied to the target FL deployment to generate membership predictions for target sessions. The primary metric is AUC-ROC, computed over the full test set of target sessions.

### 6.4 Differential Privacy Parameter Sweep

Differential privacy is implemented via the Gaussian Mechanism applied to gradient updates. Before aggregation, each client clips its gradients with a maximum L2 norm (max_grad_norm) and adds isotropic Gaussian noise calibrated to achieve (ε, δ)-differential privacy, where δ = 1e-5 in all experiments. The noise standard deviation σ is computed as σ = max_grad_norm × √(2 × ln(1.25/δ)) / ε, following the standard Gaussian Mechanism formulation. The experiment sweeps ε ∈ {0.1, 0.5, 1.0, 2.0, 5.0}. For each value of ε, all FL training runs (across all round counts) apply the same noise calibration. The first completed experiment, at ε = 1.0 and 100 rounds, yielded AUC-ROC = 0.5172, indicating that FedMIA achieves only marginally above-chance performance at this privacy budget, consistent with the theoretical expectation.

### 6.5 Metrics

The primary evaluation metric is AUC-ROC (Area Under the Receiver Operating Characteristic Curve), which measures the attack model's ability to discriminate between member and non-member sessions across all classification thresholds. An AUC-ROC of 0.5 corresponds to random guessing; an AUC-ROC of 1.0 corresponds to perfect attack performance. Secondary metrics include precision, recall, and F1 score at a classification threshold of 0.5. For RQ3, the ε vs. AUC-ROC curve is the primary deliverable: a plot of attack AUC-ROC as a function of ε, with 95% confidence intervals estimated over repeated runs with different random seeds. For RQ5, the utility metric is the mean reconstruction error of the global model on the held-out test set, compared to the no-DP baseline reconstruction error.

### 6.6 Experimental Baselines

Four baseline conditions are evaluated: (1) No-DP FedAvg, in which training proceeds without any differential privacy mechanism, establishing the upper bound on attack performance; (2) DP-FedAvg at each ε value in the sweep; (3) DP-FedProx at each ε value in the sweep; and (4) the IDS-enabled condition, in which CUSUM, Krum, and Cosine Similarity monitors are active and their outputs are logged, to assess whether any of these mechanisms produce signals that correlate with the presence of a passive MIA adversary. The IDS-enabled and IDS-disabled conditions use identical FL training configurations; the only difference is the activation of the monitoring components in the ML Plane.

### 6.7 Reproducibility

Every experimental parameter — FL round count, privacy budget, optimizer hyperparameters, dataset split ratio, random seeds, model architecture dimensions — is declared in YAML configuration files stored in the project repository. No values are hardcoded in source files. Docker images for all components (NVFLARE server, NVFLARE client, FedMIA attacker, IDS monitor) are built from versioned Dockerfiles and tagged with the experiment version. The Containerlab topology is declared in a single YAML file that fully specifies all twelve nodes, their inter-connections, IP addressing, and protocol assignments. A Makefile provides entry points for the full experimental pipeline: `make setup` instantiates the Containerlab topology; `make train` executes FL training for the configured parameter set; `make attack` executes FedMIA; `make evaluate` computes and logs all metrics; `make clean` tears down all containers and removes intermediate artifacts. The ACN-Data JPL dataset is publicly available from Caltech under an open license and is downloaded programmatically by the setup procedure. The complete artifact, including code, configuration, and dataset access scripts, will be archived on Zenodo with a persistent DOI at the time of paper submission.

---

## 7. Evaluation Plan

### RQ1: FedMIA Effectiveness Across DP Privacy Budgets

The experiment that answers RQ1 is the full ε × rounds parameter sweep: 5 ε values × 4 round counts = 20 experimental conditions per aggregation algorithm. The primary deliverable is the ε vs. AUC-ROC curve, showing how attack performance degrades as the privacy budget tightens. The expected result is a monotonically decreasing AUC-ROC as ε decreases, with the curve approaching 0.5 at ε ≤ 0.5 and rising toward a maximum (potentially approaching 0.8 or higher) at ε = 5.0. The hypothesis would be falsified if AUC-ROC does not exhibit a clear monotonic relationship with ε — for instance, if AUC-ROC at ε = 0.5 is not significantly lower than at ε = 5.0, which would suggest that the Gaussian Mechanism implementation is not correctly calibrated or that the attack model is not sensitive to gradient noise at the scales used.

### RQ2: Data Heterogeneity and MIA Effectiveness

The experiment that answers RQ2 is the cluster-stratified analysis of attack performance: for each ε value, the AUC-ROC is computed separately for sessions originating from each of the four cluster types (Highway, Urban, Residential, Corporate). The expected result is that clusters with more distinctive local data distributions — particularly Residential — exhibit higher per-cluster AUC-ROC than clusters with more diffuse distributions. The hypothesis would be falsified if per-cluster AUC-ROC differences are not statistically significant after controlling for ε and round count, indicating that data heterogeneity does not contribute meaningfully to the attack surface beyond what DP predicts.

### RQ3: Minimum Effective ε

The experiment that answers RQ3 is the same ε × rounds sweep, analyzed to identify the smallest ε at which the AUC-ROC 95% confidence interval includes 0.5 for all round counts tested. The expected result is that ε* ≤ 1.0, based on the first experimental result (AUC-ROC = 0.5172 at ε = 1.0, 100 rounds). The hypothesis would be falsified if ε* > 1.0 — that is, if statistically significant attack advantage persists at ε = 1.0 when averaged over repeated runs and longer training durations. This would indicate that the current DP implementation requires revision or that longer training amplifies information leakage in a way not captured by the single-round result.

### RQ4: IDS Detectability

The experiment that answers RQ4 is the IDS-enabled condition: all three IDS mechanisms (CUSUM, Krum, Cosine Similarity) are active during a FL training run in which a simulated passive MIA adversary is present. The adversary's presence is operationalized by designating one NVFLARE client container as the attacker — it executes the FL protocol faithfully (to simulate the passive behavior) while simultaneously logging all received gradient updates for offline analysis. IDS alarm rates, false positive rates, and detection latency are logged throughout the training run. The expected result is zero detection events attributable to the passive adversary — the IDS mechanisms produce alarms only at background false positive rates indistinguishable from the no-adversary baseline. The hypothesis would be falsified if any IDS mechanism produces a statistically significant increase in alarm rate during the attack simulation, which would require a structural explanation of the detection mechanism.

### RQ5: Utility Preservation

The experiment that answers RQ5 uses the FL models trained at each ε value to compute reconstruction error on the held-out test set. The expected result is that utility degradation is monotonically increasing as ε decreases — lower ε corresponds to more noise and worse reconstruction — but that at ε = ε* (the threshold identified in RQ3), utility remains within an operationally acceptable range, defined as reconstruction error no more than 20% higher than the no-DP baseline. The hypothesis would be falsified if utility at ε* is unacceptably degraded — specifically, if reconstruction error at ε* is more than 50% higher than the no-DP baseline, which would indicate an irreconcilable privacy/utility trade-off at the identified threshold.

### RQ6: FedProx vs. FedAvg

The experiment that answers RQ6 compares AUC-ROC between FedAvg and FedProx conditions at each ε value and round count. The expected result is that FedProx exhibits lower AUC-ROC than FedAvg at equivalent ε values under non-IID conditions, because the proximal regularization produces more homogeneous gradient updates that are less individually informative. The hypothesis would be falsified if no statistically significant difference in AUC-ROC is observed between FedAvg and FedProx conditions, indicating that the proximal regularization does not materially affect the information leakage profile of gradient updates.

---

## 8. Reproducibility Strategy

### 8.1 Docker Image Architecture

Each major component of ChargeShield-FL is packaged as an independent Docker image, enabling isolated deployment, version pinning, and independent scaling. The NVFLARE server image contains the NVFLARE 2.7.2 aggregator, experiment orchestration scripts, and the global model checkpoint management logic. The NVFLARE client image contains the local training code, the DP-SGD implementation (gradient clipping and Gaussian noise injection), and the NVFLARE client runtime. The FedMIA attacker image contains the shadow model training code, the attack model training and inference code, and the AUC-ROC evaluation scripts. The IDS monitor image contains the CUSUM, Krum, and Cosine Similarity implementations, along with logging and alerting infrastructure. All images are built from Dockerfiles pinned to specific base image digests and with all Python dependencies specified in requirements.txt files with exact version constraints. Images are published to a project-specific container registry and tagged with both a semantic version and a short git commit hash.

### 8.2 Containerlab Topology

The network topology is declared as a single Containerlab YAML file that specifies all twelve node containers, their Docker image references, their inter-node links with IP addressing and MTU parameters, and the WireGuard tunnel configurations between cluster gateways. The NVFLARE server is additionally specified with its mTLS certificate configuration. This declarative topology file constitutes the single source of truth for the experimental environment: deploying the topology requires only a Containerlab installation and the pre-built Docker images, and the deployment is deterministic and reproducible across machines.

### 8.3 YAML Configuration Files

Every experimental parameter is externalized into YAML configuration files organized by component. The FL configuration file specifies the aggregation algorithm, round count, local epochs, batch size, optimizer hyperparameters, and random seed. The DP configuration file specifies ε, δ, max_grad_norm, and the noise mechanism. The attack configuration file specifies the shadow dataset fraction, the attack model architecture, and the membership inference threshold. The dataset configuration file specifies the dataset path, feature list, normalization method, and split ratios. No values are hardcoded in source files; all configuration is loaded from these YAML files at runtime. This design ensures that running a new experimental condition requires only modifying a YAML file and re-executing the Makefile pipeline.

### 8.4 Makefile Targets

The Makefile provides the following entry points: `make setup` (downloads the dataset, builds Docker images, and instantiates the Containerlab topology), `make train` (executes FL training for the parameter configuration specified in the active YAML files), `make attack` (executes FedMIA on the trained model checkpoint), `make evaluate` (computes all metrics and writes results to a structured JSON output file), `make sweep` (executes the full ε × rounds parameter sweep by iterating over all YAML configurations), `make test` (runs the 77 unit tests via pytest), and `make clean` (tears down the Containerlab topology and removes all intermediate artifacts). The sweep target is designed to be parallelizable: independent experimental conditions (different ε × rounds combinations) can be executed concurrently on machines with sufficient CPU resources.

### 8.5 Dataset Access

ACN-Data JPL is publicly available from Caltech under an open research license. The project includes a download script that fetches the dataset programmatically, applies the required pre-processing steps, and writes the processed dataset to the expected path. This ensures that dataset access is fully automated and does not require manual steps from a researcher seeking to reproduce the experiments. The dataset version (2019–2020 JPL sessions) and the pre-processing steps are documented in the project configuration files.

### 8.6 GitHub Repository Structure and CI

The project repository is organized as follows: `src/` contains all source code, subdivided by component; `configs/` contains all YAML configuration files; `topology/` contains the Containerlab topology YAML; `docker/` contains all Dockerfiles; `tests/` contains all 77 unit tests organized by component; `scripts/` contains utility scripts including the dataset download script; and `results/` contains structured JSON output files from completed experiments. The branch strategy follows a trunk-based development model with feature branches for each sprint and pull requests to main. GitHub Actions CI runs on every pull request: unit tests are executed via pytest, Dockerfiles are linted via hadolint, and YAML configuration files are validated against JSON schemas.

### 8.7 Zenodo Artifact Archival

At the time of paper submission, a snapshot of the repository — including all source code, configuration files, topology definitions, Dockerfiles, and pre-processed dataset — will be uploaded to Zenodo and archived with a persistent DOI. The Zenodo artifact will be referenced in the paper under the Data Availability section, enabling reviewers and future researchers to access the exact version of the framework used to produce the reported results. Docker images corresponding to the paper version will additionally be tagged and preserved in the project container registry.

---

## 9. Threats to Validity

### 9.1 Internal Validity

The primary threat to internal validity is the shadow model training methodology: because the shadow dataset is drawn from the same distribution as the target dataset, the attack model may overestimate the attack success that would be observed against a target dataset with a different distribution. In the ChargeShield-FL context, the shadow dataset is drawn from the same ACN-Data corpus as the training data, which is realistic (an attacker with domain knowledge could access the same public dataset) but may not reflect all deployment scenarios. The mitigation strategy is to maintain strict disjointness between the shadow dataset and the target clients' training data: the shadow dataset consists of the half of ACN-Data sessions not assigned to any client's training data, and the split is determined by a fixed random seed applied before any training to prevent information leakage between splits. An additional internal validity concern is the potential for implementation errors in the DP mechanism; this is mitigated by unit tests that verify the noise distribution and clipping behavior against analytically computed expected values.

### 9.2 External Validity

The most significant threat to external validity is the single-site nature of the dataset: ACN-Data JPL represents one specific facility — a fleet vehicle charging station at a research institution in California — with usage patterns that may not generalize to residential European EV users, public urban charging networks, or highway rest stop fast-chargers. The non-IID cluster partitioning in ChargeShield-FL is a modeling choice that approximates multi-context heterogeneity, but it does not substitute for real data from diverse deployment contexts. The planned integration of the ElaadNL dataset — a Dutch public EV charging dataset with a very different user population — is the primary mitigation for this threat. A secondary external validity concern is the containerized emulation environment: real OT devices may exhibit communication latency, packet loss, and resource constraints that affect FL convergence in ways not captured by the Containerlab emulation. This concern is noted as a limitation and motivates future real hardware validation.

### 9.3 Construct Validity

AUC-ROC is the standard metric for evaluating binary membership inference attacks and is appropriate for this setting because it is threshold-independent and provides a summary measure of the attack model's discriminative ability. However, AUC-ROC does not directly measure practical re-identification risk: an AUC-ROC of 0.6 means that the attack distinguishes members from non-members at above-chance rates, but it does not quantify how many individuals could be re-identified in a real deployment with a specific attack precision target. The secondary metrics (precision, recall, F1 at threshold 0.5) partially address this gap, but future work could adopt more operationally grounded privacy metrics such as the per-instance privacy loss distribution proposed in the auditing literature.

### 9.4 Conclusion Validity

The primary threat to conclusion validity is the limited number of repeated runs per experimental condition in the current experimental plan. If each ε × rounds combination is run only once, the reported AUC-ROC values do not come with confidence intervals and statistical significance cannot be assessed. The mitigation plan is to execute each experimental condition with at least five different random seeds, enabling the computation of 95% confidence intervals via bootstrapping. Conclusion validity is further threatened by the potential for confounding between ε, round count, and data heterogeneity — effects may be entangled in ways that require controlled comparisons. The experimental design addresses this through factorial structure: because ε and round count are fully crossed, their effects can be separated in analysis.

---

## 10. Expected Paper Structure

### 1. Abstract

A 200-word structured abstract summarizing the problem (MIA in FL for EV charging infrastructure), the approach (ChargeShield-FL framework, FedMIA, DP parameter sweep), the key results (ε vs. AUC-ROC curve, ε* threshold, IDS non-detection result), and the contribution (modular reproducible framework, ML Plane abstraction, empirical evaluation, negative result).

### 2. Introduction

The introduction motivates the problem by situating it at the intersection of FL privacy and OT security, identifies the specific gap (lack of MIA evaluation frameworks for OT/critical infrastructure FL), states the research questions, summarizes the contributions, and provides a roadmap to the rest of the paper.

### 3. Background

This section provides the technical background required to understand the contributions. It covers: Federated Learning (FedAvg, FedProx, the non-IID challenge); Membership Inference Attacks (shadow model approach, AUC-ROC metric); Differential Privacy (Gaussian Mechanism, (ε, δ)-DP, gradient clipping); EV charging infrastructure (OCPP protocol stack, Purdue Model relevance, deployment heterogeneity); and the Purdue Model itself (hierarchical levels, demilitarized zone concept, relevance to FL traffic routing).

### 4. Related Work

A structured survey of related work organized by topic: MIA against FL (Shokri et al. 2017, Nasr et al. 2019, Carlini et al. 2022); DP in FL (McMahan et al. 2018, Geyer et al. 2017, Mironov 2017); Byzantine-robust aggregation (Blanchard et al. 2017, Yin et al. 2018); privacy in OT environments (IEC 62443-4-2, NIST SP 800-82); EV infrastructure security (ISO 15118, OCPP 2.0.1 security); FL in IoT/edge settings (Li et al. 2020, Konecny et al. 2016); and FL evaluation frameworks (PySyft, TensorFlow Federated, Flower). The section closes with a clear positioning of ChargeShield-FL relative to the surveyed work.

### 5. System Model

A formal presentation of the system model as described in Section 4 of this roadmap, presented at the level of precision required for a research paper: formal notation for node set, cluster partition, data distribution, FL protocol, and communication architecture.

### 6. The ChargeShield-FL Framework

A detailed description of the ChargeShield-FL framework: the ML Plane abstraction and its relationship to the Purdue Model hierarchy; the framework architecture (components, interfaces, observer pattern); the NVFLARE integration; the DP-SGD implementation; and the FedMIA attack module. This section constitutes the core technical contribution section of the paper.

### 7. Threat Model

A formal threat model as described in Section 5 of this roadmap, following the standard structure of the FL privacy literature: attacker goal, capabilities, knowledge, limitations, trust assumptions, and scope.

### 8. Experimental Evaluation

This section is organized into three case studies. Case Study 1 (CS1) presents the DP parameter sweep: ε vs. AUC-ROC curves for FedAvg and FedProx, with confidence intervals, addressing RQ1 and RQ3. Case Study 2 (CS2) presents the multi-cluster heterogeneity analysis: per-cluster AUC-ROC breakdown, addressing RQ2 and RQ6. Case Study 3 (CS3) presents the privacy/utility trade-off: reconstruction error as a function of ε, addressing RQ5. The IDS detectability experiment (RQ4) is presented as a supplementary result within CS1.

### 9. Discussion

The discussion interprets the experimental results in light of the research questions, addresses the privacy/utility trade-off and its practical implications for infrastructure operators, acknowledges the limitations of the study (threats to validity), and proposes future work directions: Rényi DP composition, ElaadNL generalization, Scenario 2 (client-side MIA), cryptographic defenses.

### 10. Conclusion

A concise summary of the contributions, key findings, and their significance for the research community and for practitioners deploying FL in critical infrastructure.

### 11. References

A complete bibliography in the venue-required format (IEEE DSN uses a numbered reference style), covering all works cited in the paper.

---

## 11. Remaining Research Activities

### Activity 1: Complete Full ε × Rounds Sweep

**Description:** Execute all remaining FL training runs (4 round counts × 5 ε values × 2 aggregation algorithms) and FedMIA attacks in the Containerlab environment, logging AUC-ROC and utility metrics for each condition.

**Estimated Effort:** 8–12 hours CPU time for FL training; 2–4 hours for FedMIA execution per condition; total wall-clock time depends on parallelization. With full parallelism across a 16-core machine, the sweep can complete in approximately 24 hours. With sequential execution, estimated 5–7 days.

**Dependencies:** Framework stability (Sprints 1–5 complete and passing 77 unit tests — this dependency is satisfied). Sprint 6 development must be complete before the sweep is executed.

### Activity 2: Analyze CS1 Results and Produce ε vs. AUC-ROC Curve

**Description:** Load sweep result JSON files, compute per-condition AUC-ROC with 95% confidence intervals over repeated seeds, produce the ε vs. AUC-ROC plot for FedAvg and FedProx, identify ε* as the smallest ε at which the confidence interval includes 0.5 across all round counts.

**Estimated Effort:** 4–8 hours analysis and plotting.

**Dependencies:** Full ε × rounds sweep complete (Activity 1).

### Activity 3: Execute CS2 (Multi-Cluster Heterogeneity) Experiment

**Description:** Run the cluster-stratified analysis: compute per-cluster AUC-ROC for each ε value and round count, compare FedAvg vs. FedProx per cluster, test for statistical significance of between-cluster AUC-ROC differences.

**Estimated Effort:** 4–6 hours analysis; no additional FL training required if cluster-level metrics are logged during the CS1 sweep.

**Dependencies:** CS1 analysis complete (Activity 2); cluster-level metrics must have been logged during the sweep.

### Activity 4: Execute CS3 (DP vs. No-DP Utility) Experiment

**Description:** Train no-DP FedAvg and FedProx baselines; evaluate reconstruction error at each ε value; produce the privacy/utility trade-off plot; determine whether utility at ε* is within the acceptable range.

**Estimated Effort:** 4–6 hours for no-DP training runs; 2–4 hours for utility evaluation and plotting.

**Dependencies:** CS1 analysis complete (Activity 2) to have identified ε*.

### Activity 5: Conduct Related Work Survey

**Description:** Survey and annotate 20–30 papers across the topics identified in Section 12 (MIA, DP in FL, Byzantine-robust aggregation, OT privacy, EV security, FL in IoT, evaluation frameworks). Produce a structured bibliography and a draft Related Work section.

**Estimated Effort:** 20–30 hours of reading and annotation; 8–10 hours of writing.

**Dependencies:** None. This activity can begin immediately and proceed in parallel with all experimental activities.

### Activity 6: Statistical Significance Analysis

**Description:** For each experimental condition in the CS1 sweep, run five additional repetitions with different random seeds. Use these to compute 95% confidence intervals for AUC-ROC via bootstrapping. Apply a two-sided t-test or Wilcoxon signed-rank test to assess whether AUC-ROC is significantly above 0.5 at each ε value.

**Estimated Effort:** 5× additional compute time relative to the primary sweep; 4–6 hours for statistical analysis.

**Dependencies:** CS1 sweep complete (Activity 1).

### Activity 7: ElaadNL Dataset Integration

**Description:** Download and pre-process the ElaadNL public EV charging dataset. Implement the cluster mapping and non-IID partitioning logic for the Dutch context. Run a subset of the CS1 sweep on the ElaadNL dataset to assess generalizability of the ε* threshold.

**Estimated Effort:** 8–12 hours for dataset pre-processing; 16–24 hours for ElaadNL-specific sweep runs.

**Dependencies:** None (independent of experimental activities). This activity strengthens external validity but is not required for the primary submission.

### Activity 8: Rényi DP Composition Implementation

**Description:** Implement Rényi Differential Privacy (RDP) accounting using the moments accountant approach of Mironov (2017), as an alternative to the standard (ε, δ)-DP accountant. RDP provides tighter composition bounds for training over multiple rounds and would strengthen the theoretical rigor of the privacy analysis.

**Estimated Effort:** 10–16 hours of implementation and validation against known bounds.

**Dependencies:** None (independent). This activity is desirable for theoretical completeness but is not required for the primary experimental results.

### Activity 9: Write Paper Draft

**Description:** Write all eleven sections of the DSN 2027 paper in their full form, following the structure outlined in Section 10 of this roadmap. This includes formal notation, all figures and tables, and a complete bibliography.

**Estimated Effort:** 60–80 hours of writing, distributed over 4–6 weeks.

**Dependencies:** All experiments complete (Activities 1–6); related work survey complete (Activity 5).

### Activity 10: Internal Review

**Description:** Circulate the paper draft for internal review among co-authors. Incorporate feedback, revise sections, and prepare the final submission version.

**Estimated Effort:** 10–15 hours of revision per review cycle; anticipate two review cycles.

**Dependencies:** Paper draft complete (Activity 9).

### Activity 11: Submission to DSN 2027

**Description:** Prepare and submit the final paper to DSN 2027. DSN submission deadlines are typically in November of the year preceding the conference; DSN 2027 is expected to have a deadline in November 2026. Zenodo archival of the artifact must be completed before submission to provide the artifact DOI for the Data Availability section.

**Estimated Effort:** 4–8 hours for final submission preparation (formatting, compliance check, artifact upload, author information, abstract submission).

**Dependencies:** Internal review complete (Activity 10); Zenodo archival complete.

---

## 12. Literature to Study

### 12.1 Membership Inference Attacks Against ML and FL

**Shokri, R., Stronati, M., Song, C., & Shmatikoff, V. (2017). Membership Inference Attacks Against Machine Learning Models. IEEE S&P 2017.**
This is the foundational paper for the shadow model approach to membership inference. It establishes the attack methodology, the AUC-ROC evaluation framework, and the experimental paradigm that FedMIA directly extends. Essential reading for understanding the attack mechanism implemented in ChargeShield-FL.

**Nasr, M., Shokri, R., & Houmansadr, A. (2019). Comprehensive Privacy Analysis of Deep Learning: Passive and Active White-Box Inference Attacks Against Centralized and Federated Learning. IEEE S&P 2019.**
This paper extends MIA to the federated setting and distinguishes passive from active inference attacks. It introduces the white-box federated MIA model that informs Scenario 1 in ChargeShield-FL and provides the theoretical grounding for why gradient updates leak membership information.

**Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., & Tramer, F. (2022). Membership Inference Attacks From First Principles. IEEE S&P 2022.**
This paper provides a rigorous likelihood-ratio test formulation of membership inference, establishing a theoretical upper bound on attack performance and motivating the use of AUC-ROC as the primary metric. Essential for understanding the relationship between empirical attack performance and theoretical privacy guarantees.

**Hu, H., Salcic, Z., Sun, L., Dobbie, G., Yu, P.S., & Zhang, X. (2022). Membership Inference Attacks on Machine Learning: A Survey. ACM Computing Surveys.**
A comprehensive survey covering the full landscape of MIA techniques, defenses, and evaluation methodologies. Provides the contextual framing for positioning ChargeShield-FL's FedMIA implementation within the broader literature.

### 12.2 Differential Privacy in Federated Learning

**McMahan, H.B., Ramage, D., Talwar, K., & Zhang, L. (2018). Learning Differentially Private Recurrent Language Models. ICLR 2018.**
This paper introduces DP-FedAvg, the combination of FedAvg with per-round differential privacy via gradient clipping and Gaussian noise. It establishes the (ε, δ)-accounting framework for FL and is the direct precursor to the DP implementation in ChargeShield-FL.

**Geyer, R.C., Klein, T., & Nabi, M. (2017). Differentially Private Federated Learning: A Client Level Perspective. NeurIPS 2017 Workshop.**
An early paper on client-level differential privacy in FL, which applies DP at the client level rather than the gradient level. Provides important conceptual contrast with the approach used in ChargeShield-FL and motivates the discussion of granularity in DP accounting.

**Mironov, I. (2017). Rényi Differential Privacy. IEEE CSF 2017.**
This paper introduces Rényi Differential Privacy, which provides tighter composition bounds than standard (ε, δ)-DP. Understanding RDP is essential for the planned Activity 8 (Rényi DP composition) and for engaging with current best practices in DP accounting for iterative algorithms.

### 12.3 Byzantine-Robust FL Aggregation

**Blanchard, P., El Mhamdi, E.M., Guerraoui, R., & Stainer, J. (2017). Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent. NeurIPS 2017.**
This paper introduces Krum, one of the three IDS baselines implemented in ChargeShield-FL. Understanding the theoretical guarantees and assumptions of Krum is essential for the RQ4 analysis of why Krum fails to detect passive MIA adversaries.

**Yin, D., Chen, Y., Ramchandran, K., & Bartlett, P. (2018). Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates. ICML 2018.**
Introduces coordinate-wise median and trimmed mean as robust aggregation alternatives to FedAvg. Provides the theoretical context for Byzantine-robust aggregation and is relevant to the discussion of why Byzantine robustness does not imply privacy.

**Cao, X., & Fang, M. (2020). FLTrust: Byzantine-Robust Federated Learning via Trust Bootstrapping. NDSS 2021.**
Introduces a trust-score-based robust aggregation mechanism. Relevant as an additional baseline for the IDS detectability analysis and for the future work discussion.

### 12.4 Privacy in OT and ICS Environments

**IEC 62443-4-2: Security for Industrial Automation and Control Systems — Technical Security Requirements for IACS Components.**
The foundational standard for OT security at the component level, defining security levels and requirements for industrial devices. Essential for situating ChargeShield-FL within the OT security compliance landscape.

**ETSI EN 303 645: Cyber Security for Consumer Internet of Things: Baseline Requirements.**
Relevant for the Residential cluster nodes, which represent consumer IoT devices in the MQTT v5 category. Provides the regulatory context for security requirements at the edge of the EV charging network.

**NIST SP 800-82 Rev. 3: Guide to OT Security.**
The primary US federal guidance document for OT security, covering the Purdue Model, security zone architecture, and specific guidance for industrial control systems. Essential for situating the ML Plane abstraction within established OT security frameworks.

### 12.5 EV Infrastructure Security

**ISO 15118: Road Vehicles — Vehicle to Grid Communication Interface.**
The international standard governing secure communication between EVs and charging stations, including the Plug & Charge (PnC) authentication mechanism. Provides the automotive security context for understanding why privacy in EV charging data is a regulatory concern.

**OCPP 2.0.1 Security Annex, Open Charge Alliance.**
The security specification for OCPP 2.0.1, covering TLS requirements, certificate management, and security profiles. Essential for understanding the security posture of the Corporate cluster nodes.

**ENISA: Cybersecurity Challenges in the Uptake of Artificial Intelligence in Autonomous Driving (2021) and ENISA EV Charging Security Report.**
European regulatory guidance on security requirements for EV charging infrastructure, providing the European policy context for the research and the regulatory motivation for privacy-preserving FL.

### 12.6 Federated Learning in IoT and Edge Computing

**Li, T., Sahu, A.K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). Federated Optimization in Heterogeneous Networks. MLSys 2020.**
The paper introducing FedProx. Essential for understanding the motivation for the proximal regularization term and its effect on convergence in non-IID settings, directly relevant to RQ6.

**Konecny, J., McMahan, H.B., Yu, F.X., Richtarik, P., Suresh, A.T., & Bacon, D. (2016). Federated Learning: Strategies for Improving Communication Efficiency. NeurIPS 2016 Workshop.**
A foundational paper on federated optimization, covering the communication efficiency rationale for FL and establishing the FedAvg algorithm. Essential background for the Background and Related Work sections of the paper.

**Lim, W.Y.B., et al. (2020). Federated Learning in Mobile Edge Networks: A Comprehensive Survey. IEEE Communications Surveys and Tutorials.**
A comprehensive survey of FL deployment challenges in edge computing environments, covering communication constraints, non-IID data challenges, and security considerations. Relevant for situating ChargeShield-FL within the IoT/edge FL literature.

### 12.7 Privacy Evaluation Frameworks

**Ziller, A., et al. (2021). PySyft: A Library for Easy Federated Learning. Federated Learning: Privacy and Incentive.**
Documents the PySyft framework for privacy-preserving FL. Relevant as a baseline comparison for ChargeShield-FL's framework design and as a potential integration target for future work.

**Bonawitz, K., et al. (2019). Towards Federated Learning at Scale: A System Design. MLSys 2019.**
Documents the engineering challenges and design decisions in deploying FL at scale. Provides the systems context for evaluating ChargeShield-FL's architectural choices and for understanding what production FL deployments look like.

**Beutel, D.J., et al. (2020). Flower: A Friendly Federated Learning Research Framework. arXiv 2007.14390.**
Documents the Flower FL framework, a flexible research-oriented alternative to NVFLARE. Relevant as a comparative framework and for understanding the FL framework landscape in which ChargeShield-FL is positioned.

---

## 13. Publication Readiness Checklist

### Done

- [x] **Framework implemented end-to-end (Sprints 1–5 complete)** — All core components — FL clients, FL server, DP-SGD, FedMIA attacker, IDS monitors, ML Plane observer — are implemented and integrated. Sprint 1–5 acceptance criteria are satisfied.

- [x] **Real dataset integrated (ACN-Data JPL, 13,073 sessions)** — The full 2019–2020 ACN-Data JPL corpus has been downloaded, pre-processed, and integrated into the FL data pipeline. Non-IID partitioning by cluster is implemented and verified.

- [x] **DP implementation (Gaussian Mechanism, gradient clipping)** — The (ε, δ)-DP implementation using the Gaussian Mechanism with gradient clipping is complete and unit-tested. Noise calibration follows the standard σ = max_grad_norm × √(2 × ln(1.25/δ)) / ε formula.

- [x] **FedMIA implemented (shadow model, AUC-ROC)** — The FedMIA attack module is complete: shadow model training, attack model training, membership inference, and AUC-ROC computation are all implemented and tested.

- [x] **IDS baselines implemented (CUSUM, Krum, Cosine Similarity)** — All three IDS mechanisms are implemented as independent monitor components in the ML Plane and are integrated into the Containerlab topology.

- [x] **Containerlab topology (12 nodes, 4 clusters, mTLS, WireGuard)** — The declarative Containerlab YAML topology for all 12 nodes, cluster assignments, inter-node links, mTLS configuration, and WireGuard tunnels is complete and deployable.

- [x] **NVFLARE integration (FedAvg + FedProx, 2.7.2)** — Both FedAvg and FedProx aggregation strategies are implemented and integrated with NVFLARE 2.7.2. The FL orchestration pipeline is end-to-end operational.

- [x] **First experiment result (AUC-ROC=0.5172, ε=1.0, 100 rounds)** — The first experimental data point has been collected and logged: FedMIA AUC-ROC = 0.5172 at ε = 1.0, 100 FL rounds, FedAvg aggregation. This result is consistent with the theoretical expectation that DP at ε = 1.0 is effective in this deployment context.

- [x] **77 unit tests passing** — The full unit test suite (77 tests, covering all components) passes with zero failures. Tests are organized by component and run in CI on every pull request via GitHub Actions.

### In Progress

- [ ] **Full ε × rounds sweep (Sprint 6)** — The parameter sweep across ε ∈ {0.1, 0.5, 1.0, 2.0, 5.0} × rounds ∈ {100, 200, 500, 1000} × {FedAvg, FedProx} is currently in execution. Estimated completion: end of Sprint 6. Results for the first condition (ε = 1.0, 100 rounds) are complete; remaining conditions are in progress.

- [ ] **Sprint 6 development** — Sprint 6 is underway, addressing sweep automation, result logging infrastructure, and statistical analysis tooling. Sprint completion is the prerequisite for executing the full sweep in a reproducible and automated manner.

### Not Started

- [ ] **CS2 (multi-cluster heterogeneity) experiment** — The cluster-stratified AUC-ROC analysis has not yet been executed. Blocked on CS1 sweep completion and result logging.

- [ ] **CS3 (DP vs. no-DP utility) experiment** — The utility evaluation experiment has not yet been executed. Blocked on CS1 analysis to identify ε* and on no-DP baseline training runs.

- [ ] **Related work survey** — The structured survey of 20–30 papers across all relevant topic areas has not yet been conducted. This is an independent activity that can begin immediately in parallel with ongoing development.

- [ ] **Statistical significance analysis (confidence intervals, repeated seeds)** — Repeated runs with different random seeds and bootstrapped confidence intervals have not yet been computed. Blocked on CS1 sweep completion.

- [ ] **ElaadNL dataset integration** — The ElaadNL Dutch EV charging dataset has not yet been downloaded, pre-processed, or integrated into the framework. This optional activity strengthens external validity considerably.

- [ ] **Paper draft** — No sections of the DSN 2027 paper have been drafted. Blocked on all experiments and the related work survey.

- [ ] **Real hardware validation (desirable, not required)** — Validation of key results on real EV charging hardware (physical EVSE with OCPP stack) has not been attempted. This would significantly strengthen external validity but is not required for DSN submission.

- [ ] **Rényi DP composition** — The RDP accountant implementation has not been started. This independent activity would strengthen the theoretical rigor of the DP analysis by providing tighter composition bounds across training rounds.

- [ ] **Zenodo archival** — The research artifact has not yet been archived on Zenodo. This must be completed before paper submission to provide the persistent DOI required for the Data Availability section.

---

*This document constitutes the living research roadmap for the ChargeShield-FL project. It should be updated at the conclusion of each sprint and after each experimental milestone. Target submission: DSN 2027, deadline expected November 2026.*
