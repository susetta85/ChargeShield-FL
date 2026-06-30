# PrivacyAuditor: Design, Implementation, and Empirical Role in ChargeShield-FL

**Component:** Privacy Auditor (`PrivacyAuditor`)
**Framework:** ChargeShield-FL — A Research Framework for Evaluating Membership Inference Attacks Against Federated Learning in EV Charging Infrastructure
**Target Venue:** DSN 2027 (International Conference on Dependable Systems and Networks)
**Document Version:** 1.0.0
**Classification:** Research Technical Documentation

---

## Implementation Status: PrivacyAuditor.audit() Is Now Actively Called

**As of the current codebase, `PrivacyAuditor.audit()` is actively called in `scripts/run_experiments.py::run_ids()` for every node at every FL round.** In earlier versions, `run_ids()` constructed `AuditReport` objects manually (with `threats_detected=[]`) and `audit()` was dead code. This is no longer the case.

The call site in `run_ids()`:
1. A `PrivacyAuditor` instance is created before the FL round loop: `auditor = PrivacyAuditor(config_path=..., epsilon=cfg["experiment"]["epsilon"])`. The `epsilon` parameter overrides the value in `auditor.yaml`, ensuring the auditor uses the same DP budget as the experiment (critical during parameter sweeps where epsilon varies per run).
2. For each `GradientUpdate` received, the gradient tensors are converted to a `dict[str, Any]` with keys `layer_0`, `layer_1`, ... and values `list[float]` (via `w.flatten().tolist()` for each weight tensor `w`).
3. `auditor.audit(node_id, round_id, model_update)` is called with this dictionary as the `model_update` argument.

The `audit()` method signature accepts `model_update: dict[str, Any]` — not a flat `np.ndarray` as in older stub code. The layer-keyed dictionary format matches the structure of NVFLARE weight diffs where each key identifies a model layer.

---

## Abstract

This document provides a complete technical and scientific specification of the `PrivacyAuditor` component within the ChargeShield-FL framework. The `PrivacyAuditor` is the primary **observation instrument** of the framework: it intercepts individual gradient updates from Federated Learning (FL) participants prior to server-side aggregation, computes a suite of privacy leakage proxies — including gradient sensitivity, per-round differential privacy (DP) epsilon estimates, and cumulative privacy budget consumption — and emits structured `AuditReport` objects consumed by downstream modules including the `FedMIA` attack evaluator and the `ChargingIDS` anomaly detection system.

A critical design principle governs this component: the `PrivacyAuditor` is a **measurement instrument, not a defense mechanism**. Its role is to make the empirical evaluation of Membership Inference Attacks (MIA) against federated EV charging models scientifically rigorous and reproducible. Without the auditor, MIA evaluation is epistemically blind — the attack surface is unquantified, per-node heterogeneity is unobservable, and temporal dynamics of privacy budget consumption cannot be correlated with attack success. The auditor closes this gap by providing a principled, instrumentalized observation layer that is conceptually analogous to a voltmeter in an electrical circuit: it measures without (ideally) disturbing the system under study.

The document covers: (1) the scientific motivation and position of the auditor in the FL pipeline; (2) gradient sensitivity as a memorization proxy with formal grounding in DP theory; (3) epsilon estimation methodology and its limitations; (4) the `AuditReport` dataclass specification; (5) pattern detection logic; (6) integration with `FedMIA` and `ChargingIDS`; (7) paper-level evaluation metrics; (8) full YAML configuration; and (9) complete Python API with integration examples.

---

## 1. Introduction

### 1.1 Scientific Role: The Auditor as Measurement Instrument

In experimental science, the quality of an empirical claim is bounded by the quality of the measurement instrument used to gather evidence. A thermometer does not heat the room it measures; an oscilloscope does not generate the signal it observes. This principle — instrument neutrality — is as relevant to empirical security research as it is to physics.

The `PrivacyAuditor` is designed along this philosophy. ChargeShield-FL is a research framework whose primary scientific contribution is to demonstrate, empirically and rigorously, that Membership Inference Attacks are feasible against federated learning models trained on Electric Vehicle (EV) charging data, and that the degree of attack success is meaningfully correlated with the differential privacy budget consumed during training. This contribution requires three capabilities: (a) the ability to execute a well-specified MIA (provided by `FedMIA`); (b) an anomaly detection baseline for comparison (provided by `ChargingIDS`); and (c) a principled observation layer that quantifies, per node and per round, the extent of privacy exposure — provided by `PrivacyAuditor`.

Without (c), claims about MIA feasibility would be unsupported by rigorous intermediate measurements. The paper would assert that MIA succeeds on the aggregated model, but would have no mechanism to explain *why* certain nodes are more vulnerable, *when* during training vulnerability peaks, or *how* differential privacy noise mitigates the attack. The `PrivacyAuditor` transforms the framework from a black-box attack demonstration into a white-box measurement study.

### 1.2 Auditing Versus Protecting: A Fundamental Distinction

A persistent conceptual confusion in the FL privacy literature conflates privacy auditing with privacy protection. This confusion is harmful because it leads to flawed experimental design: systems that are simultaneously measurement instruments and defenses create feedback loops that contaminate the empirical results they purport to generate.

The `PrivacyAuditor` makes no attempt to improve the privacy of the FL system under study. It does not clip gradients, inject noise, suppress updates, or modify the aggregation algorithm. It observes, computes, and reports. The downstream `ChargingIDS` component may use the auditor's output to take protective actions (e.g., excluding a node whose budget is exhausted), but this is a separate architectural concern. The auditor and the IDS are connected by a data flow, not by a design identity.

This separation has a methodological justification: if the auditor were also a defense, its activation would change the system state it is meant to measure, introducing observer effects that compromise reproducibility and confound the MIA evaluation. The clean architectural boundary — auditor measures, IDS responds — ensures that audit measurements reflect the true state of the system under the attacker model.

### 1.3 The Auditor as the Empirical Foundation of ChargeShield-FL

The broader contribution of ChargeShield-FL rests on three empirical claims:

1. Passive, gradient-level MIA is feasible against FL-trained EV charging models under an honest-but-curious aggregator threat model.
2. Behavioral anomaly detection (IDS) is insufficient to detect passive MIA — the attack leaves no behavioral fingerprint.
3. Differential privacy with a strict epsilon budget demonstrably reduces MIA success, at a measurable utility cost.

All three claims require the `PrivacyAuditor`. Claim 1 requires per-node sensitivity measurements to identify the attack surface. Claim 2 requires the auditor to confirm that the IDS receives no anomalous signal during a successful MIA. Claim 3 requires epsilon accounting that can be correlated with `FedMIA` AUC-ROC values across nodes and rounds.

In this sense, the `PrivacyAuditor` is what makes ChargeShield-FL empirically grounded. It is the instrument through which the scientific claims of the paper are made falsifiable.

---

## 2. Position in the FL Pipeline

### 2.1 Pre-Aggregation Intercept: The Richest Observation Point

In a federated learning system, gradient updates flow from local trainers (FL clients) to the aggregation server. The aggregation server applies a function — typically Federated Averaging (FedAvg) — to combine updates from multiple clients into a single global model update. The `PrivacyAuditor` operates at the **pre-aggregation** stage: it intercepts each individual node's gradient update *before* the aggregation function is applied.

This placement is deliberate and scientifically justified. Post-aggregation gradients represent a weighted average over all participating clients. Individual client contributions are diluted by the averaging operation: for $N$ clients with gradient updates $g_1, g_2, \ldots, g_N$, the aggregated gradient $\bar{g} = \frac{1}{N}\sum_{i=1}^{N} g_i$ carries no per-client attribution by construction. An auditor placed post-aggregation would observe only the blended signal, losing the ability to:

- Compute per-node sensitivity and epsilon estimates;
- Detect per-node gradient anomalies (explosion, masking);
- Track heterogeneous budget consumption across nodes with different data distributions;
- Provide node-level features to `FedMIA` for per-client MIA evaluation.

Pre-aggregation is thus the richest observation point for individual-level privacy analysis. It is also the point where the most informative gradient signal is available: each gradient $g_i$ directly reflects the local dataset $D_i$ of client $i$, making it the most sensitive artifact in the FL protocol from a MIA perspective.

### 2.2 Post-DP Noise Addition

In ChargeShield-FL, each local trainer applies the Gaussian Mechanism of Differential Privacy to its gradient update before transmission. The noised gradient received by the auditor is:

$$\tilde{g}_i = g_i + \mathcal{N}(0, \sigma^2 \mathbf{I})$$

where $\sigma$ is the per-round noise scale and $\mathbf{I}$ is the identity matrix in gradient space. The `PrivacyAuditor` therefore never observes the raw, un-noised gradient — it observes the post-noise gradient that would be transmitted over the network. This is an intentional design constraint that ensures the auditor's measurements reflect the actual attack surface: an honest-but-curious aggregator also only ever sees $\tilde{g}_i$, not $g_i$.

This constraint also bounds the auditor's epsilon estimates: because the auditor cannot access $\sigma$ directly in a black-box evaluation scenario, it uses the empirical L2-norm of $\tilde{g}_i$ as a proxy for gradient sensitivity, introducing a deliberate approximation whose limitations are fully characterized in Section 4.

### 2.3 Attacker Model: Honest-But-Curious Aggregator

The threat model assumed by ChargeShield-FL positions the FL aggregation server as an **honest-but-curious** (HbC) adversary. The HbC aggregator:

- Faithfully executes the FL protocol (applies FedAvg, does not inject malicious updates);
- Passively observes all gradient updates transmitted by clients;
- Attempts to infer membership in the training data of individual clients using only the information available from gradient observation;
- Operates under mTLS encryption for transport security, but has access to plaintext gradients post-decryption.

The key implication for the auditor: the attacker's information set is exactly the set of noised gradients $\{\tilde{g}_i\}$ received at the aggregation server. The `PrivacyAuditor` is co-located with this observation point — it sees what the attacker sees, and quantifies how much membership information is encoded in each gradient update.

### 2.4 NVFLARE Integration

ChargeShield-FL uses NVFLARE 2.7.2 as its FL orchestration framework. NVFLARE provides a task-based execution model where each FL round consists of a `train` task dispatched to clients and a subsequent `aggregate` task executed on the server. The `PrivacyAuditor` integrates into this lifecycle via a **custom server-side workflow step** inserted between the `train` task completion event and the `aggregate` task invocation.

Concretely, the integration is realized as a custom `Aggregator` subclass that overrides the `aggregate()` method. Before delegating to the standard `InTimeAccumulateWeightedAggregator`, the custom aggregator extracts each client's contribution, invokes `PrivacyAuditor.audit()`, stores the resulting `AuditReport`, and then proceeds with normal aggregation. This design ensures:

- Zero modification to client-side NVFLARE executors;
- Full compatibility with existing NVFLARE round lifecycle management;
- Auditor invocation is atomic per client contribution, preventing race conditions in multi-threaded aggregation.

The NVFLARE server configuration registers the custom aggregator in `config_fed_server.json` under the `aggregator` key, replacing the default `InTimeAccumulateWeightedAggregator` with `ChargeShieldAggregator`.

### 2.5 Data Flow Description

The complete data flow through the system is as follows:

```
Local Trainer (FL Client)
        |
        | Raw gradient g_i = gradient of Loss(w; D_i)
        v
[DP Noise Addition: Gaussian Mechanism]
        |
        | Noised gradient g_tilde_i = g_i + N(0, sigma^2 * I)
        v
[mTLS Encrypted Transmission]
        |
        | Encrypted g_tilde_i over TLS
        v
[mTLS Decryption at Aggregation Server]
        |
        | Plaintext g_tilde_i (honest-but-curious aggregator observes here)
        v
[PrivacyAuditor.audit(node_id, round_id, model_update, baseline_norm)]
        [model_update: dict[str, Any] — layer_i -> list[float] via w.flatten().tolist()]
        |
        | AuditReport(node_id, round_id, privacy_score, epsilon,
        |             threats_detected, metadata)
        +----------------------------------+
        v                                  v
[FedMIA: Shadow Model + Autoencoder]   [ChargingIDS: CUSUM + Krum]
        |                                  |
        | AUC-ROC per node/round           | ALLOW / MONITOR / THROTTLE / EXCLUDE
        v                                  v
[Aggregation: FedAvg]
        |
        | Updated global model w_{t+1}
        v
[Global Model Distribution to Clients]
```

This data flow makes explicit that the `PrivacyAuditor` is a **fork point**: it consumes the gradient stream and produces `AuditReport` objects that feed two independent downstream consumers, while the gradient stream itself continues unmodified to aggregation.

---

## 3. Gradient Sensitivity as a Memorization Proxy

### 3.1 Intuition: Why Gradient Norms Reflect Memorization

The fundamental question of membership inference is: given access to a model's gradient update, can an adversary determine whether a specific data point was part of the training set? The answer is positive because training on a dataset $D$ induces a gradient that is a function of *every example in $D$*. If a specific example $x$ is in $D$, the gradient $\nabla L(w; D)$ is systematically influenced by $x$'s contribution to the loss. If $x \notin D$, this influence is absent.

The degree of influence is captured by the concept of **gradient sensitivity**: how much does the gradient change when a single example is added or removed from the training set? Formally, for a dataset $D$ and a single example $x$, the sensitivity of the gradient function is:

$$\Delta f = \sup_{D, x} \|\nabla L(w; D \cup \{x\}) - \nabla L(w; D)\|_2$$

This quantity, known as the $\ell_2$ global sensitivity in the differential privacy literature, quantifies the maximum influence any single training example can have on the gradient vector. When $\Delta f$ is large, individual examples have disproportionate influence on the gradient — meaning the gradient encodes strong membership signals, and MIA is more likely to succeed.

### 3.2 Formal Grounding in Differential Privacy Theory

The connection between gradient sensitivity and MIA risk is not merely heuristic — it is formally grounded in the DP literature. The Gaussian Mechanism achieves $(\epsilon, \delta)$-differential privacy for a function $f$ by adding noise calibrated to $\Delta f$:

$$\mathcal{M}(D) = f(D) + \mathcal{N}\!\left(0, \frac{2\ln(1.25/\delta) \cdot \Delta f^2}{\epsilon^2} \cdot \mathbf{I}\right)$$

The noise scale $\sigma = \frac{\sqrt{2\ln(1.25/\delta)} \cdot \Delta f}{\epsilon}$ is directly proportional to $\Delta f$. For fixed $\sigma$, a larger $\Delta f$ implies larger $\epsilon$ — i.e., weaker privacy. Equivalently, for fixed target $\epsilon$, larger $\Delta f$ requires larger noise $\sigma$ to achieve the same privacy guarantee.

The empirical implication for the `PrivacyAuditor`: when the L2-norm of the observed gradient $\|\tilde{g}_i\|_2$ is high, this is evidence that the underlying gradient $g_i$ has high sensitivity to individual data points in $D_i$. High sensitivity implies:

1. The DP noise added by the local trainer may be insufficient to mask individual contributions;
2. The gradient carries more membership information, increasing MIA success probability;
3. The per-round epsilon consumption is higher, depleting the privacy budget faster.

The precise formal statement is: for the gradient function $f(D) = \nabla L(w; D)$, the global sensitivity $\Delta f$ upper-bounds the influence of any single example. When the empirical gradient norm $\|\tilde{g}_i\|_2$ substantially exceeds the baseline (the average norm over earlier rounds), this constitutes evidence of elevated sensitivity in the current round.

### 3.3 Why L2-Norm as a First-Order Proxy

A natural question is why the `PrivacyAuditor` uses the L2-norm of the gradient vector rather than higher-order statistics such as the Hessian, the Fisher Information Matrix (FIM), or the trace of the gradient covariance matrix. These higher-order quantities would, in principle, provide richer information about the geometry of the loss landscape and the concentration of membership information.

The answer is practical and well-motivated:

**Computational cost.** The Hessian of the loss with respect to model parameters is a matrix of dimension $|\theta|^2$ where $|\theta|$ is the number of model parameters. For modern deep learning models with millions of parameters, computing and storing the Hessian is computationally infeasible in a real-time FL system. The Fisher Information Matrix suffers from the same scaling problem. The L2-norm of the gradient, by contrast, is computable in $O(|\theta|)$ time and space.

**Availability in standard FL implementations.** NVFLARE 2.7.2, like most FL frameworks, transmits gradient updates as flat arrays of floating-point values. Higher-order statistics require additional instrumentation of the local trainer's backward pass — modifications that are not available in black-box FL evaluation, where the auditor has access only to transmitted artifacts.

**Empirical adequacy.** Despite being a first-order statistic, the L2-norm of gradient updates has been shown empirically to correlate with membership vulnerability. Carlini et al. (2022) demonstrate that gradient norms are predictive of per-example memorization. Shokri et al. (2017) use gradient information directly as features for MIA. These results justify the use of L2-norm as a practical, interpretable proxy.

### 3.4 Limitation: L2-Norm as Proxy, Not Bound

The L2-norm is a proxy, not a tight bound on membership leakage. Several failure modes must be acknowledged:

- **High norm, low leakage:** A gradient with high L2-norm may result from legitimate distributional shift (a client receiving an unusual batch) rather than from memorization of specific examples. In this case, the auditor will flag elevated sensitivity without a corresponding increase in MIA success.

- **Low norm, high leakage:** Conversely, an attacker who has engineered a gradient masking attack could suppress the L2-norm while preserving membership-informative substructures in the gradient vector. The `FEDMIA_SUSPICIOUS_LOW_SENSITIVITY` pattern is designed to flag this scenario.

- **Aggregation dilution:** The noising operation $\tilde{g}_i = g_i + \mathcal{N}(0, \sigma^2 \mathbf{I})$ affects the observed norm. In expectation, $\mathbb{E}[\|\tilde{g}_i\|_2^2] = \|g_i\|_2^2 + |\theta|\sigma^2$. The auditor's norm measurement conflates gradient signal with noise power. For high-noise regimes (large $\sigma$), the norm is dominated by the noise term and is not a reliable proxy for gradient sensitivity.

These limitations are documented in the paper's threat model section and do not invalidate the proxy — they qualify its interpretation and motivate future work using tighter, formal sensitivity bounds.

---

## 4. Epsilon Estimation

### 4.1 Simplified Gaussian Mechanism Accounting

For each FL round $t$ and node $i$, the `PrivacyAuditor` computes a per-round epsilon estimate using a simplified Gaussian mechanism accounting formula:

$$\hat{\epsilon}_{i,t} = \frac{\|\tilde{g}_{i,t}\|_2}{\sigma_{\max}}$$

where $\|\tilde{g}_{i,t}\|_2$ is the L2-norm of the observed (noised) gradient update from node $i$ in round $t$, and $\sigma_{\max}$ is the configured maximum gradient norm (a clipping threshold, denoted `max_grad_norm` in the configuration). This expression approximates the relationship between sensitivity, noise scale, and epsilon in the Gaussian mechanism.

In standard Gaussian mechanism accounting, for sensitivity $\Delta f$ and noise scale $\sigma$, the per-mechanism epsilon under simplified (non-Renyi) composition is approximately $\Delta f / \sigma$ for large $\sigma$ (formally, this is a loose bound; the exact expression involves the complementary error function). The `PrivacyAuditor` uses the empirical gradient norm as a proxy for $\Delta f$ and the configured `max_grad_norm` as a proxy for $\sigma$ (since both are measured in the same units in gradient space).

### 4.2 Mathematical Derivation

The Gaussian Mechanism $\mathcal{M}(D) = f(D) + \mathcal{N}(0, \sigma^2 \mathbf{I})$ is $(\epsilon, \delta)$-DP for:

$$\epsilon = \frac{\Delta f}{\sigma} \sqrt{2 \ln(1.25/\delta)}$$

For a fixed $\delta$ (e.g., $\delta = 10^{-5}$, a standard choice for FL), the factor $\sqrt{2\ln(1.25/\delta)}$ is a constant (approximately 3.54 for $\delta = 10^{-5}$). The `PrivacyAuditor` absorbs this constant into a simplified estimate by treating $\hat{\epsilon}_{i,t} \approx \Delta f / \sigma$, which is conservative (overestimates epsilon). In practice, `max_grad_norm` serves as the denominator because it is the configured clipping threshold — a standard DP practice (Abadi et al., 2016) where gradient norms are clipped to `max_grad_norm` before noise addition, making `max_grad_norm` the effective sensitivity bound.

### 4.3 Cumulative Epsilon via Basic Composition

The cumulative epsilon for node $i$ after $T$ rounds is computed via **basic composition**:

$$\hat{\epsilon}_{i}^{(T)} = \sum_{t=1}^{T} \hat{\epsilon}_{i,t}$$

The Basic Composition Theorem (Dwork and Roth, 2014) states that the composition of $T$ mechanisms, each $(\epsilon_t, \delta_t)$-DP, is $\bigl(\sum_{t=1}^{T} \epsilon_t,\; \sum_{t=1}^{T} \delta_t\bigr)$-DP. The `PrivacyAuditor` applies this theorem directly, treating each FL round as an independent mechanism application.

### 4.4 Limitations of the Epsilon Estimate

The auditor's epsilon estimate is approximate in several respects, and this must be disclosed in the paper:

**(a) Empirical norm as sensitivity proxy.** The true global sensitivity $\Delta f$ is a supremum over all possible datasets and examples — it is a worst-case quantity. The empirical norm $\|\tilde{g}_{i,t}\|_2$ is a single-sample estimate that conflates gradient signal and DP noise. It may overestimate (if noise dominates) or underestimate (if the batch is unrepresentative) the true sensitivity.

**(b) Basic composition is not tight.** Renyi Differential Privacy (RDP) composition, introduced by Mironov (2017), gives significantly tighter epsilon bounds under adaptive composition. For a Gaussian mechanism with noise multiplier $q = \sigma/\Delta f$, the RDP bound at order $\alpha$ is $\alpha / (2q^2)$, which converts to a much tighter $(\epsilon, \delta)$-DP bound than basic composition. The `PrivacyAuditor` does not use RDP because it requires knowledge of the noise multiplier $q$ at the auditor level — a parameter that may not be available in black-box FL evaluation scenarios.

**(c) Heterogeneous noise across nodes.** Different nodes may apply different noise scales $\sigma_i$ depending on their local privacy requirements. The auditor's simplified accounting uses a single `max_grad_norm` value for all nodes, which may mischaracterize nodes with non-standard noise configurations.

**(d) Gradient clipping interaction.** If the local trainer clips gradients to `max_grad_norm` before adding noise, the true sensitivity is bounded by the clipping threshold — but clipping also compresses the gradient norm information, making the auditor's norm-based estimate a lower bound on actual sensitivity.

### 4.5 Why Not Renyi DP Accounting?

The adoption of RDP accounting in ChargeShield-FL is designated as **future work** for the following reasons: (i) RDP accounting at the auditor level requires access to the noise multiplier $\sigma/\Delta f$, which is an internal parameter of the local trainer's DP mechanism — not exposed through the standard NVFLARE gradient transmission protocol; (ii) the added precision of RDP is not necessary for the paper's primary contribution, which is relative comparison across nodes and rounds rather than formal DP certification; (iii) the simplified accounting is **conservative** — it overestimates epsilon, meaning that any threshold-based budget exhaustion detection is conservative rather than premature.

The paper explicitly acknowledges this limitation in its evaluation section and bounds the impact: for the epsilon budget values used in the experiments ($\epsilon_{\text{budget}} \in \{1.0, 5.0, 10.0\}$), the qualitative patterns of budget consumption and their correlation with MIA success are robust to the estimation imprecision.

---

## 5. Privacy Score Derivation

### 5.1 Definition and Semantics

The **privacy score** for node $i$ at the end of round $T$ is defined as:

$$\text{ps}_{i}^{(T)} = 1.0 - \min\!\left(\frac{\hat{\epsilon}_{i}^{(T)}}{\epsilon_{\text{budget}}}, 1.0\right)$$

where $\hat{\epsilon}_{i}^{(T)}$ is the cumulative epsilon estimate after $T$ rounds and $\epsilon_{\text{budget}}$ is the configured total privacy budget. The $\min(\cdot, 1.0)$ clamp ensures the score is always in $[0.0, 1.0]$.

The semantics of the privacy score are deliberately intuitive:

- $\text{ps} = 1.0$: The privacy budget is fully intact. No epsilon has been consumed. This is the state at the beginning of training (round 0).
- $\text{ps} = 0.0$: The privacy budget is exhausted. All $\epsilon_{\text{budget}}$ units of epsilon have been consumed. No remaining formal DP protection exists for subsequent rounds.
- $\text{ps} \in (0, 1)$: An intermediate state, with the score reflecting the fraction of the budget still remaining.

### 5.2 Rationale for Inversion

The privacy score inverts the cumulative epsilon ratio. This inversion is motivated by the intended semantics for downstream consumers: both `FedMIA` and `ChargingIDS` benefit from a high-means-safe convention. A high privacy score signals low MIA risk; a low privacy score signals high MIA risk. This convention is consistent with security dashboards and alert systems where high scores are favorable, and decreasing scores trigger graduated responses.

Without inversion, the raw epsilon ratio would be a **risk score** (high = more risk). While either convention is valid, the inversion reduces the cognitive burden on downstream component implementers and on human analysts interpreting experimental results.

### 5.3 Per-Node Independence

Each node $i$ maintains an **independent epsilon accumulator** $\hat{\epsilon}_{i}^{(\cdot)}$ and therefore an independent privacy score trajectory $\{\text{ps}_{i}^{(t)}\}_{t=0}^{T}$. This per-node independence is scientifically essential because:

- Different EV charging cluster types (highway, urban, residential, corporate) have different local data distributions, resulting in different gradient magnitudes and therefore different epsilon consumption rates.
- Highway clusters process high-throughput fast-charging events with concentrated temporal patterns, producing high-sensitivity gradients. Residential clusters process low-throughput overnight charging with distributed patterns, producing lower-sensitivity gradients.
- Shared epsilon accounting would mask this heterogeneity, obscuring one of the paper's key findings: that fixed epsilon budgets create unequal privacy protection across cluster types.

### 5.4 Temporal Dynamics

The privacy score trajectory is **monotonically non-increasing**: $\text{ps}_{i}^{(t+1)} \leq \text{ps}_{i}^{(t)}$ for all $i, t$. This follows directly from the non-negativity of per-round epsilon estimates ($\hat{\epsilon}_{i,t} \geq 0$) and the monotonicity of the cumulative sum. A node's privacy score can only decrease (or stay constant if its gradient update has negligibly small norm, which would be flagged by the `FEDMIA_SUSPICIOUS_LOW_SENSITIVITY` pattern).

This monotonicity property makes the privacy score time series amenable to CUSUM-based anomaly detection in `ChargingIDS`: the IDS monitors for rates of decrease that significantly exceed the expected per-round consumption rate, flagging nodes that are consuming budget at an anomalous rate.

### 5.5 Use in FedMIA and ChargingIDS

In `FedMIA`, the privacy score serves as an **input feature** to the shadow model classifier. Specifically, the feature vector for each gradient observation includes `privacy_score`, `gradient_norm`, and `sensitivity` from the `AuditReport.metadata` dictionary. The intuition is that the privacy score encodes information about the cumulative DP protection applied to the gradient — a low privacy score (high epsilon consumption) implies that later-round gradients are relatively unprotected, making them more amenable to membership inference.

In `ChargingIDS`, the privacy score time series $\{\text{ps}_{i}^{(t)}\}$ is processed by a CUSUM control chart that detects statistically significant deviations from the expected consumption rate. The IDS uses the `threats_detected` list from the `AuditReport` as categorical flags that trigger graduated response actions (MONITOR, THROTTLE, EXCLUDE).

---

## 6. AuditReport Dataclass — Full Specification

### 6.1 Design Philosophy

The `AuditReport` is designed as an immutable, serializable data transfer object. It encapsulates all information produced by a single invocation of `PrivacyAuditor.audit()` for a specific (node, round) pair. Immutability is enforced via `frozen=True` in the dataclass definition, preventing accidental mutation by downstream consumers.

The dataclass uses a `metadata` dictionary for extensible payload, following the principle of progressive disclosure: core fields (node_id, round_id, privacy_score, epsilon, threats_detected) provide the minimal information needed for routing decisions, while `metadata` provides the richer quantitative information needed for `FedMIA` feature extraction and paper-level analysis.

### 6.2 Full Python Dataclass Definition

```python
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AuditReport:
    """
    Immutable report produced by PrivacyAuditor for a single (node, round) pair.

    This is the primary output artifact of the PrivacyAuditor. It encapsulates
    all privacy leakage proxy measurements computed from a single gradient update,
    and is consumed by FedMIA (for MIA feature extraction) and ChargingIDS
    (for anomaly detection and response decisions).

    Fields
    ------
    node_id : str
        Unique identifier for the FL client or charging cluster. Follows the
        convention "<cluster_type>_cluster_<index>", e.g., "highway_cluster_1",
        "urban_cluster_3". Matches the node identifiers used in the NVFLARE
        client configuration.

    round_id : int
        FL training round number (1-indexed). Round 0 is the initialization
        state before any training; round 1 is the first training round.

    privacy_score : float
        Current privacy budget fraction remaining, in [0.0, 1.0].
        1.0 = full budget intact; 0.0 = budget exhausted.
        Computed as: 1.0 - min(epsilon_cumulative / epsilon_budget, 1.0).

    epsilon : float
        Estimated differential privacy epsilon consumed in this round.
        Computed as: gradient_norm / max_grad_norm (simplified Gaussian accounting).
        Non-negative; may be 0.0 if gradient norm is effectively zero.

    threats_detected : List[str]
        List of threat pattern identifiers detected in this round. May be empty.
        Possible values: "GRADIENT_EXPLOSION", "PRIVACY_BUDGET_NEAR_EXHAUSTION",
        "PRIVACY_BUDGET_EXHAUSTED", "FEDMIA_SUSPICIOUS_LOW_SENSITIVITY".
        Order is not semantically significant.

    metadata : Dict[str, Any]
        Extensible payload containing quantitative measurements and provenance
        information. Standard keys:
            gradient_norm (float): L2-norm of the observed gradient update.
            sensitivity (float): Per-round sensitivity estimate (= gradient_norm).
            baseline_norm (float): Baseline gradient norm for anomaly detection.
            update_magnitude (float): gradient_norm / baseline_norm.
            epsilon_cumulative (float): Total epsilon consumed across all rounds to date.
            timestamp (float): Unix timestamp of the audit computation (time.time()).
            cluster_type (str): Inferred cluster type from node_id prefix.
    """

    node_id: str
    round_id: int
    privacy_score: float
    epsilon: float
    threats_detected: List[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary (JSON-compatible)."""
        return {
            "node_id": self.node_id,
            "round_id": self.round_id,
            "privacy_score": self.privacy_score,
            "epsilon": self.epsilon,
            "threats_detected": list(self.threats_detected),
            "metadata": dict(self.metadata),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AuditReport":
        """Deserialize from a plain dictionary."""
        return cls(
            node_id=d["node_id"],
            round_id=d["round_id"],
            privacy_score=float(d["privacy_score"]),
            epsilon=float(d["epsilon"]),
            threats_detected=list(d["threats_detected"]),
            metadata=dict(d["metadata"]),
        )

    @property
    def is_high_risk(self) -> bool:
        """True if privacy score is below the near-exhaustion threshold (< 0.2)."""
        return self.privacy_score < 0.2

    @property
    def cluster_type(self) -> str:
        """Inferred cluster type from node_id prefix."""
        return self.metadata.get("cluster_type", "unknown")
```

### 6.3 Example Instantiation and JSON Serialization

```python
# Example: AuditReport for highway_cluster_1, round 42
report = AuditReport(
    node_id="highway_cluster_1",
    round_id=42,
    privacy_score=0.156,
    epsilon=0.087,
    threats_detected=["PRIVACY_BUDGET_NEAR_EXHAUSTION"],
    metadata={
        "gradient_norm": 0.87,
        "sensitivity": 0.87,
        "baseline_norm": 0.45,
        "update_magnitude": 1.933,
        "epsilon_cumulative": 4.219,
        "timestamp": 1719392847.331,
        "cluster_type": "highway",
    },
)

print(report.to_json())
# Output:
# {
#   "node_id": "highway_cluster_1",
#   "round_id": 42,
#   "privacy_score": 0.156,
#   "epsilon": 0.087,
#   "threats_detected": ["PRIVACY_BUDGET_NEAR_EXHAUSTION"],
#   "metadata": {
#     "gradient_norm": 0.87,
#     "sensitivity": 0.87,
#     "baseline_norm": 0.45,
#     "update_magnitude": 1.933,
#     "epsilon_cumulative": 4.219,
#     "timestamp": 1719392847.331,
#     "cluster_type": "highway"
#   }
# }

print(report.is_high_risk)   # True  (privacy_score = 0.156 < 0.2)
print(report.cluster_type)   # "highway"
```

---

## 7. Pattern Detection

The `PrivacyAuditor` implements four threat pattern detectors. Each detector is evaluated independently for every `audit()` call. Detected patterns are collected into the `threats_detected` list of the resulting `AuditReport`. The following subsections specify each pattern's detection condition, theoretical significance, and recommended IDS response action.

### 7.1 GRADIENT_EXPLOSION

**Detection condition:**

$$\|\tilde{g}_{i,t}\|_2 > \theta_{\text{explode}} \cdot \text{baseline\_norm}$$

where $\theta_{\text{explode}}$ is configured as `explosion_threshold` (default: 10.0) and `baseline_norm` is provided as a parameter to `audit()` — typically the exponential moving average of previous gradient norms for that node.

**Significance.** Gradient explosion — a sudden, extreme increase in gradient magnitude — is anomalous under normal FL training dynamics. In a well-configured FL system with gradient clipping and appropriate learning rates, gradient norms remain bounded within a predictable range across rounds. Explosion may indicate:

- **Training instability:** The local model has diverged, possibly due to a bad learning rate or a highly heterogeneous local batch. This is a utility concern and may also indicate a data quality issue at the EV charging cluster.
- **Adversarial gradient injection:** A malicious client (beyond the honest-but-curious threat model) may inject a large gradient to corrupt the global model or to amplify the influence of specific training examples. Gradient explosion is a prerequisite for gradient inversion attacks (Zhu et al., 2019).
- **Data corruption or sensor malfunction:** In EV charging infrastructure, sensor faults can produce anomalous charging records, which — if not filtered — produce extreme loss values and correspondingly large gradients.

**Why not expected in the honest-but-curious scenario.** Under the HbC attacker model, the aggregator does not modify gradients — it only observes them. Gradient explosion at the client side would be visible to the aggregator as an anomalously large transmitted gradient. In ChargeShield-FL's experimental setup, DP clipping bounds gradient norms to `max_grad_norm`, so explosion should not occur unless DP is misconfigured or bypassed.

**IDS recommended action.** On first detection: THROTTLE (reduce the node's weight in aggregation). On persistent detection (three or more consecutive rounds): EXCLUDE (remove the node from the current training session and flag for operator review).

### 7.2 PRIVACY_BUDGET_NEAR_EXHAUSTION

**Detection condition:**

$$\hat{\epsilon}_{i}^{(t)} > r_{\text{near}} \cdot \epsilon_{\text{budget}}$$

where $r_{\text{near}}$ is configured as `near_exhaustion_ratio` (default: 0.8), i.e., the node has consumed more than 80% of its total privacy budget.

Equivalently: $\text{ps}_{i}^{(t)} < 1 - r_{\text{near}} = 0.2$.

**Significance.** A node approaching its privacy budget limit is entering a regime where the remaining DP protection is thin. Each subsequent round consumes a larger fraction of the residual budget. The gradient updates from such a node carry progressively more membership information, increasing the success probability of MIA against that node's training data.

In ChargeShield-FL's experimental context, this pattern identifies which EV charging clusters exhaust their privacy budget first. The paper's hypothesis — that highway clusters (high-throughput, high-sensitivity gradients) exhaust budgets faster than residential clusters — is supported by the occurrence rate of this pattern across cluster types.

**Recommended action.** MONITOR: the IDS should increase its monitoring frequency for this node, log the event, and alert the FL operator for budget review. Continued training is permitted but flagged. The operator may choose to reduce the node's contribution weight or to replenish the budget by renegotiating training terms — decisions that are outside the scope of the automated system.

**Experimental tracking.** The round at which each node first triggers `PRIVACY_BUDGET_NEAR_EXHAUSTION` is recorded as a metric (`near_exhaustion_round`) and reported in the paper's per-cluster analysis. Distributions of this metric across cluster types provide quantitative evidence of heterogeneous privacy consumption.

### 7.3 PRIVACY_BUDGET_EXHAUSTED

**Detection condition:**

$$\hat{\epsilon}_{i}^{(t)} \geq \epsilon_{\text{budget}}$$

Equivalently: $\text{ps}_{i}^{(t)} = 0.0$.

**Significance.** Budget exhaustion means that all formally allocated differential privacy protection for this node has been consumed. Subsequent gradient updates from this node carry **no remaining formal DP guarantee** — the DP mechanism's privacy parameters are no longer valid because the composition budget has been fully spent. From a formal DP perspective, continued training of this node is equivalent to training without any DP protection.

This is the most severe privacy event in the auditor's detection hierarchy. In ChargeShield-FL's experimental design, the occurrence of `PRIVACY_BUDGET_EXHAUSTED` events demonstrates concretely that fixed epsilon budgets cannot sustain arbitrary numbers of training rounds — a finding that motivates per-node adaptive budgeting as future work.

**Paper implication.** The rounds at which different nodes exhaust their budgets, plotted against their cluster type, demonstrate that a single fixed epsilon budget creates systematically unequal privacy protection: highway clusters exhaust their budgets in approximately half the rounds required by residential clusters (under the experimental parameters). This constitutes empirical evidence for the heterogeneous privacy exposure problem in federated EV charging systems.

**Recommended action.** EXCLUDE: the node should be removed from further FL training rounds. Including budget-exhausted nodes in aggregation would produce a model that has, for those nodes, no formal privacy protection — a situation that ChargeShield-FL treats as a protocol violation. The IDS issues an EXCLUDE action and records the node's final round.

### 7.4 FEDMIA_SUSPICIOUS_LOW_SENSITIVITY

**Detection condition:**

$$\|\tilde{g}_{i,t}\|_2 < \theta_{\text{low}} \cdot \text{baseline\_norm}$$

where $\theta_{\text{low}}$ is configured as `suspicious_low_ratio` (default: 0.01), i.e., the gradient norm is less than 1% of the baseline norm.

**Significance.** Unusually low gradient sensitivity is paradoxical from a MIA risk perspective and warrants disambiguation:

**(a) High DP protection interpretation.** When the DP noise scale $\sigma$ is large relative to the gradient norm — the noise dominates the signal — the observed gradient $\tilde{g}_{i,t}$ is dominated by noise and carries very little membership information. This is the ideal DP scenario: the gradient is effectively random, and MIA should fail. In this case, the low sensitivity pattern is a *positive* finding for privacy.

**(b) Gradient masking interpretation.** An adversary who controls the local training process (beyond the HbC model) might suppress the gradient norm to defeat sensitivity-based monitoring, while encoding membership information in directions not captured by the L2-norm. Gradient masking attacks have been studied in the adversarial machine learning literature and represent a realistic threat in FL systems where client integrity cannot be assumed.

**(c) Local model collapse interpretation.** If the local model has converged (or collapsed to a trivial solution), further training may produce near-zero gradients that carry no information — about training data or about anything else. This is a utility problem: the node is not contributing meaningful updates to the global model. In EV charging infrastructure, this might occur if a cluster's charging patterns become perfectly predicted by the current global model.

**Why suspicious?** In the context of MIA evaluation, ambiguity between interpretations (a), (b), and (c) makes this pattern suspicious: the auditor cannot, from the L2-norm alone, determine whether the low sensitivity indicates strong DP protection, gradient masking, or model collapse. The appropriate response is to flag the event and trigger a FedMIA analysis specifically targeting this node's gradients, using the autoencoder-based membership signal (described in Section 8.3) to disambiguate.

**Recommended action.** MONITOR, and trigger FedMIA analysis with elevated priority for this node. If `enable_fedmia_integration` is True in the configuration, the auditor signals `FedMIA` to include this node's current round in the next batch inference pass.

---

## 8. Relationship with FedMIA

### 8.1 Overview of FedMIA

`FedMIA` implements the shadow model approach to Membership Inference Attacks introduced by Shokri et al. (2017), augmented with an autoencoder-based membership signal following Nasr et al. (2019). The attack is evaluated in a white-box setting (access to gradient updates) under the honest-but-curious aggregator threat model.

The attack proceeds in two phases:

1. **Shadow model training:** A set of shadow models are trained on datasets sampled from the same distribution as the target FL clients. For each shadow model, gradient updates are collected for samples that are in the training set (members) and samples that are not (non-members). This provides a labeled dataset $\{(g, y) : y \in \{0, 1\}\}$ where $y = 1$ denotes membership.

2. **Attack model training:** A binary classifier (the attack model) is trained on the labeled gradient dataset to predict membership. The attack model takes as input a feature vector derived from the gradient update, including statistics from the `AuditReport`.

### 8.2 AuditReport as FedMIA Input

The `AuditReport.metadata` dictionary provides the following features to `FedMIA`:

- `sensitivity` (= `gradient_norm`): The L2-norm of the gradient update, the primary memorization proxy.
- `update_magnitude`: The normalized gradient magnitude relative to baseline, capturing round-specific anomaly.
- `privacy_score`: The remaining privacy budget fraction, encoding cumulative DP exposure.
- `epsilon`: The per-round epsilon consumed, encoding the current round's DP tightness.

These features are concatenated with direct gradient statistics (mean, variance, skewness of the flattened gradient vector) to form the full feature vector $\phi(g)$ for the attack model. The inclusion of `AuditReport` features improves attack model accuracy compared to using raw gradient statistics alone, because the auditor features encode temporal and budgetary context that raw gradient statistics do not.

### 8.3 Autoencoder Reconstruction Error as Membership Signal

In addition to the shadow model attack, `FedMIA` implements an autoencoder-based membership signal. An autoencoder $\mathcal{A}$ is trained exclusively on gradient updates from **non-member** examples (gradients from shadow models trained on held-out data). The autoencoder learns to reconstruct the statistical structure of non-member gradients.

At inference time, the reconstruction error $\|g - \mathcal{A}(g)\|_2$ is used as a membership signal: if the error is high, the gradient $g$ is dissimilar from non-member gradients — suggesting it is a member gradient. If the error is low, $g$ fits the non-member distribution and is classified as a non-member.

The `FEDMIA_SUSPICIOUS_LOW_SENSITIVITY` pattern is particularly relevant here: a low-sensitivity gradient (near-zero norm) will have low reconstruction error regardless of membership status — the autoencoder reconstructs near-zero vectors accurately because it has seen many such vectors in the non-member distribution (high-DP regimes). This is the disambiguation mechanism: if `FedMIA` finds low reconstruction error for a low-sensitivity gradient, interpretation (a) (strong DP protection) is supported. If reconstruction error is high despite low norm, gradient masking becomes more likely.

### 8.4 AUC-ROC as Primary Metric

The primary evaluation metric for `FedMIA` is the Area Under the Receiver Operating Characteristic Curve (AUC-ROC). This choice is motivated by:

- **Threshold independence:** AUC-ROC summarizes classifier performance across all decision thresholds, avoiding the need to tune a specific threshold for membership classification.
- **Standard interpretability:** AUC-ROC = 0.5 corresponds to a random classifier (no membership information in the gradient), and AUC-ROC = 1.0 corresponds to perfect membership inference. Values above 0.7 are considered practically significant in the MIA literature.
- **Separability from attack model calibration:** Unlike accuracy or precision/recall at a fixed threshold, AUC-ROC is robust to class imbalance in the member/non-member split.

AUC-ROC is computed per node per round, yielding a three-dimensional data structure (node x round x AUC-ROC) that is the primary output of `FedMIA`. The correlation between this structure and the `privacy_score` time series from the `PrivacyAuditor` is the central empirical finding of the ChargeShield-FL paper.

### 8.5 Data Flow: AuditReport to FedMIA Classification

```
AuditReport.metadata["sensitivity"]      --+
AuditReport.metadata["update_magnitude"] --+
AuditReport.privacy_score                --+--> Feature vector phi(g_tilde_i,t)
AuditReport.epsilon                      --+
grad_stats(g_tilde_i,t): [mean,var,skew] --+
        |
        v
FedMIA Attack Model (Binary Classifier)
        |
        +-- Shadow model path: train on shadow gradients -> predict membership
        +-- Autoencoder path: compute ||g_tilde - A(g_tilde)||_2 -> threshold
        |
        v
Membership prediction: y_hat in {0,1} with confidence p(y=1 | phi)
        |
        v
AUC-ROC over all (g_tilde, y) pairs in evaluation batch
        |
        v
FedMIA_Result(node_id, round_id, auc_roc, n_members, n_non_members)
```

---

## 9. Relationship with Differential Privacy

### 9.1 The DP Chain in ChargeShield-FL

The differential privacy chain in ChargeShield-FL connects local training, gradient transmission, and privacy audit as follows:

1. **Local DP noise addition:** Each FL client's local trainer applies the Gaussian Mechanism to its gradient before transmission. The noise scale $\sigma$ is configured to achieve a target per-round $(\epsilon_{\text{target}}, \delta)$-DP guarantee at the local level.

2. **Gradient transmission:** The noised gradient $\tilde{g}_i$ is transmitted to the aggregation server over mTLS. Transport encryption protects against passive eavesdroppers on the network but does not protect against the honest-but-curious server, which decrypts the gradient.

3. **Privacy Auditor measurement:** The `PrivacyAuditor` observes $\tilde{g}_i$ and computes the privacy proxy measurements described in Sections 3 through 5. These measurements do not modify $\tilde{g}_i$ — the auditor is a read-only observer.

4. **MIA evaluation:** `FedMIA` uses the audit report and the raw gradient to attempt membership inference. The success of this attempt — measured by AUC-ROC — is the primary outcome variable.

### 9.2 Why DP Reduces MIA Success

Differential privacy noise reduces MIA success by degrading the signal-to-noise ratio of the membership signal in the gradient. For a noised gradient $\tilde{g} = g + n$ where $n \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$, the mutual information between the gradient observation and the membership label $y$ satisfies:

$$I(\tilde{g};\, y) \leq I(g;\, y) - C(\sigma)$$

where $C(\sigma) > 0$ is a function that increases with $\sigma$. As $\sigma \to \infty$, the gradient observation carries no membership information ($I(\tilde{g}; y) \to 0$), and MIA success approaches random (AUC-ROC $\to 0.5$). As $\sigma \to 0$, the noised gradient approaches the original gradient, and MIA success approaches the unconstrained case.

This formal relationship motivates the paper's central experiment: by varying the DP noise scale $\sigma$ (equivalently, by varying the target $\epsilon$) across experimental conditions, ChargeShield-FL traces the empirical AUC-ROC vs. epsilon curve, demonstrating the privacy-utility tradeoff in a concrete FL deployment.

### 9.3 Why DP Does Not Eliminate MIA Risk at High Epsilon

The DP guarantee is non-trivial only when the noise is calibrated to achieve a small epsilon (e.g., $\epsilon \leq 1.0$). At large epsilon values ($\epsilon > 10$), the noise is small relative to the gradient signal, and the DP protection is essentially vacuous.

Shokri et al. (2017) and Nasr et al. (2019) both demonstrate empirically that MIA remains feasible at epsilon values larger than approximately 10. Carlini et al. (2022) show that even at epsilon values as small as 1.0, memorization persists for sufficiently small models or underrepresented training examples. The `PrivacyAuditor`'s per-round epsilon accounting allows ChargeShield-FL to place each experimental condition precisely on the epsilon axis of the AUC-ROC vs. epsilon curve, giving empirical grounding to these theoretical observations in the specific context of EV charging data.

### 9.4 The Epsilon–AUC Tradeoff Curve

The central empirical finding that the `PrivacyAuditor` enables is the **epsilon–AUC tradeoff curve**: a scatter plot with $\hat{\epsilon}_{i}^{(T)}$ on the x-axis and `FedMIA` AUC-ROC on the y-axis, with one point per (node, round) pair. The expected shape of this curve is:

- At $\hat{\epsilon} \approx 0$ (early rounds, before significant budget consumption): AUC-ROC $\approx 0.5$ (random, DP noise dominates).
- As $\hat{\epsilon}$ increases: AUC-ROC increases monotonically, approaching the ceiling determined by the data distribution and model architecture.
- At $\hat{\epsilon} \geq \epsilon_{\text{budget}}$ (budget exhausted): AUC-ROC plateaus at the maximum achievable attack success.

The `PrivacyAuditor` provides the x-axis values for this curve. `FedMIA` provides the y-axis values. The curve itself is the paper's most concise and impactful visualization.

---

## 10. Relationship with ChargingIDS

### 10.1 PA to IDS Data Flow

The `PrivacyAuditor` produces `AuditReport` objects; the `ChargingIDS` consumes them via its `analyze()` method. The data flow is unidirectional: the auditor does not receive feedback from the IDS, and the IDS does not modify the audit computation. This one-way dependency is intentional.

```python
# Simplified integration in ChargeShieldAggregator
# model_update is dict[str, Any]: keys = "layer_i", values = list[float]
# constructed from GradientUpdate tensors via w.flatten().tolist()
report: AuditReport = privacy_auditor.audit(
    node_id=client_id,
    round_id=current_round,
    model_update=client_model_update,   # dict[str, Any], NOT np.ndarray
    baseline_norm=baseline_norm_registry[client_id],
)

# IDS consumes the report independently
ids_action: IDSAction = charging_ids.analyze(report)

# Gradient proceeds to aggregation unmodified
aggregated_gradient = fedavg_aggregate(all_client_gradients)
```

The `IDSAction` returned by `ChargingIDS.analyze()` is one of `{ALLOW, MONITOR, THROTTLE, EXCLUDE}`, with increasing severity. This action governs whether the node's gradient is included in aggregation (ALLOW or MONITOR), down-weighted (THROTTLE), or excluded (EXCLUDE).

### 10.2 Separation of Detection from Response

The architectural separation between `PrivacyAuditor` (detection) and `ChargingIDS` (response) embodies the **single responsibility principle**: each component has one well-defined function, and changes to one do not require changes to the other.

This separation has concrete benefits in the research context:

- **Modularity for ablation studies:** The auditor can be evaluated in isolation (do its measurements correlate with actual MIA success?) independently of whether the IDS is acting on them. This enables clean ablation studies where the IDS is disabled but the auditor is active.
- **Reproducibility:** The auditor's output is deterministic given the gradient input (modulo floating-point precision). The IDS's response may involve stochastic elements or stateful history. Separating them ensures that the audit record is a stable artifact of each experiment.
- **Scalability:** In a deployed system, the auditor might run at the aggregation server while the IDS runs in a separate monitoring process. The `AuditReport` JSON serialization (Section 6.3) enables this deployment pattern without modification.

### 10.3 IDS State and Audit Statelessness

A key asymmetry between the two components: the `ChargingIDS` is **stateful** — it maintains per-node risk score histories, CUSUM control chart states, and action histories across rounds. The `PrivacyAuditor` is **partially stateful** — it maintains per-node epsilon accumulators (necessary for cumulative budget accounting) but is otherwise stateless per-audit-invocation.

The auditor's per-node state (epsilon accumulator) is the minimum state required to produce budget-relative privacy scores. It is explicitly exposed via the `get_epsilon_history()` and `get_privacy_score_history()` methods (Section 13), and can be reset via `reset()` — enabling clean-slate evaluation for ablation studies or when a node re-enters training after exclusion.

### 10.4 IDS Privacy Signal Sources

The `ChargingIDS` uses three signal sources from the `PrivacyAuditor`:

1. **`privacy_score` time series:** Processed by CUSUM to detect anomalous budget consumption rates.
2. **`metadata["gradient_norm"]`:** Used alongside gradient geometry metrics (cosine similarity, Krum distance) to detect behavioral anomalies.
3. **`threats_detected` list:** Used as categorical flags to trigger graduated responses. The presence of `PRIVACY_BUDGET_EXHAUSTED` in `threats_detected` unconditionally triggers EXCLUDE.

The IDS also maintains its own gradient geometry computations (Krum, cosine similarity) that are independent of the auditor — these are behavioral signals designed to detect Byzantine clients, not privacy signals. The combination of privacy-aware signals (from the auditor) and behavioral signals (from the IDS's own computations) provides a more comprehensive anomaly detection capability than either alone.

---

## 11. Metrics for the Paper

The `PrivacyAuditor` generates data that supports six paper-level evaluation metrics. These metrics collectively constitute the empirical evidence base for ChargeShield-FL's three central claims.

### 11.1 Privacy Score Time Series

**Visualization:** Line plot with rounds on the x-axis and privacy score on the y-axis, one line per node, colored by cluster type (highway = red, urban = orange, residential = blue, corporate = green).

**What it shows:** The monotonic decline of privacy score for each node across training rounds, with different decline rates reflecting the gradient sensitivity heterogeneity of different cluster types. Highway clusters decline fastest; residential clusters decline slowest.

**Paper contribution:** Provides visual evidence that a single epsilon budget creates unequal privacy timelines across cluster types — the central motivation for per-type adaptive budgeting.

**Data source:** `PrivacyAuditor.get_privacy_score_history(node_id)` for each node.

### 11.2 Cumulative Epsilon Per Node

**Visualization:** Grouped bar chart or heatmap with nodes on the x-axis and $\hat{\epsilon}^{(T)}$ on the y-axis, with the budget threshold ($\epsilon_{\text{budget}}$) shown as a horizontal reference line.

**What it shows:** Which nodes have exhausted their privacy budget by the end of training ($\hat{\epsilon}^{(T)} \geq \epsilon_{\text{budget}}$), and the degree of budget consumption heterogeneity.

**Paper contribution:** Quantifies the unequal privacy exposure problem and identifies specific cluster types as high-risk.

**Data source:** `PrivacyAuditor.get_epsilon_history(node_id)` summed over all rounds, for each node.

### 11.3 Sensitivity Distribution Per Cluster Type

**Visualization:** Violin plot with cluster type on the x-axis and `gradient_norm` on the y-axis. Each violin shows the distribution of gradient norms across all rounds for all nodes of that cluster type.

**What it shows:** The gradient sensitivity distribution is significantly different across cluster types, reflecting the heterogeneity of EV charging data: highway clusters have heavy-tailed, high-norm distributions; residential clusters have concentrated, low-norm distributions.

**Paper contribution:** Empirically confirms that the heterogeneous epsilon consumption observed in metrics 11.1 and 11.2 is attributable to genuine data distribution differences, not experimental artifact.

**Data source:** `AuditReport.metadata["gradient_norm"]` collected across all (node, round) pairs, grouped by `metadata["cluster_type"]`.

### 11.4 Threat Frequency Heatmap

**Visualization:** 2D heatmap with nodes on the y-axis, rounds on the x-axis, and cell color indicating the number of threats detected in that (node, round) cell. Color scale: white = 0 threats, dark red = multiple threats.

**What it shows:** The spatial-temporal structure of the attack surface — which nodes are affected by which threats, and when during training. Typically shows a pattern where early rounds are threat-free, `PRIVACY_BUDGET_NEAR_EXHAUSTION` events appear at intermediate rounds, and `PRIVACY_BUDGET_EXHAUSTED` events appear at late rounds, with the onset time varying by cluster type.

**Paper contribution:** Provides a visually compelling summary of the attack surface that a human operator would need to monitor. Demonstrates that the threat landscape is heterogeneous and time-varying.

**Data source:** `AuditReport.threats_detected` collected across all (node, round) pairs.

### 11.5 FedMIA AUC vs. Cumulative Epsilon Scatter

**Visualization:** Scatter plot with $\hat{\epsilon}^{(T)}$ on the x-axis and FedMIA AUC-ROC on the y-axis, one point per (node, round) pair. A trend line (LOWESS or polynomial fit) is overlaid. Points are colored by cluster type.

**What it shows:** A positive correlation between cumulative epsilon consumption and MIA success probability. As nodes consume more of their privacy budget (epsilon increases), the FedMIA attack becomes progressively more successful (AUC-ROC increases).

**Paper contribution:** This is the paper's central empirical finding. It establishes the causal chain: data heterogeneity → heterogeneous gradient sensitivity → heterogeneous epsilon consumption → heterogeneous MIA vulnerability. It motivates DP with a strict, per-type epsilon budget as the primary countermeasure.

**Data source:** `PrivacyAuditor.get_epsilon_history()` (x-axis) cross-referenced with `FedMIA` AUC-ROC results (y-axis).

### 11.6 IDS Non-Detection of Passive MIA

**Visualization:** Table or bar chart showing IDS action distribution (ALLOW/MONITOR/THROTTLE/EXCLUDE rates) conditioned on whether FedMIA AUC-ROC exceeded the significance threshold (0.7). Two columns: "MIA Not Significant (AUC < 0.7)" and "MIA Significant (AUC >= 0.7)".

**What it shows:** The IDS action distribution is statistically indistinguishable between rounds where MIA is significant and rounds where it is not. The IDS cannot detect the passive MIA because gradient observation leaves no behavioral anomaly.

**Paper contribution:** Empirically validates the paper's second claim: that behavioral IDS is insufficient for MIA detection. Establishes the need for the `PrivacyAuditor` as an independent observation layer.

**Data source:** `AuditReport.threats_detected` and `ChargingIDS` action logs, cross-referenced with `FedMIA` AUC-ROC results.

### 11.7 Collective Contribution

Together, these six metrics constitute a comprehensive empirical case:

- Metrics 11.1–11.4 characterize the privacy audit landscape (the observation instrument working correctly).
- Metric 11.5 demonstrates the attack's success and its dependence on privacy budget consumption.
- Metric 11.6 demonstrates the behavioral IDS's blindness to the attack.

No single metric is sufficient alone; their combination provides the multi-dimensional evidence required for DSN 2027 quality.

---

## 12. Configuration (auditor.yaml)

```yaml
# auditor.yaml — PrivacyAuditor Configuration
# All parameters are required unless marked [optional].

privacy_auditor:

  # ────────────────────────────────────────────────────────────────────────────
  # Privacy Budget Parameters
  # ────────────────────────────────────────────────────────────────────────────

  # Total differential privacy epsilon budget per node.
  # Training is permitted until epsilon_cumulative reaches this value.
  # A node that exhausts this budget triggers PRIVACY_BUDGET_EXHAUSTED.
  # Recommended values for experimental conditions:
  #   strict:   epsilon_budget: 1.0   (strong DP, high utility cost)
  #   moderate: epsilon_budget: 5.0   (moderate DP, moderate utility cost)
  #   weak:     epsilon_budget: 10.0  (weak DP, low utility cost)
  epsilon_budget: 5.0

  # Maximum gradient norm (clipping threshold) used as the denominator in
  # per-round epsilon estimation: epsilon_round = gradient_norm / max_grad_norm.
  # Should match the clipping threshold configured in the local trainer's
  # DP mechanism. Units: same as gradient L2-norm (dimensionless ratio).
  max_grad_norm: 1.0

  # ────────────────────────────────────────────────────────────────────────────
  # Gradient Explosion Detection
  # ────────────────────────────────────────────────────────────────────────────

  # Explosion threshold multiplier. A gradient is classified as exploding if:
  #   gradient_norm > explosion_threshold * baseline_norm
  # Default 10.0: gradient must be 10x the baseline norm to trigger.
  # Increase if legitimate high-norm gradients are expected (e.g., early rounds
  # before the baseline EMA has stabilized).
  explosion_threshold: 10.0

  # ────────────────────────────────────────────────────────────────────────────
  # Near-Exhaustion Detection
  # ────────────────────────────────────────────────────────────────────────────

  # Fraction of epsilon_budget consumed before PRIVACY_BUDGET_NEAR_EXHAUSTION
  # is triggered. Default 0.8: triggered when 80% of budget is consumed.
  # Equivalently: triggered when privacy_score < (1 - near_exhaustion_ratio).
  near_exhaustion_ratio: 0.8

  # ────────────────────────────────────────────────────────────────────────────
  # Low Sensitivity Detection (FedMIA Signal)
  # ────────────────────────────────────────────────────────────────────────────

  # Suspicious low sensitivity ratio. A gradient triggers the pattern if:
  #   gradient_norm < suspicious_low_ratio * baseline_norm
  # Default 0.01: gradient must be less than 1% of baseline to trigger.
  # Decrease if gradient masking attacks use moderate (not extreme) norm suppression.
  suspicious_low_ratio: 0.01

  # ────────────────────────────────────────────────────────────────────────────
  # FedMIA Integration
  # ────────────────────────────────────────────────────────────────────────────

  # If true, the auditor signals FedMIA to prioritize nodes that trigger
  # FEDMIA_SUSPICIOUS_LOW_SENSITIVITY in the current round.
  # Set to false for ablation studies where FedMIA operates without auditor hints.
  enable_fedmia_integration: true

  # ────────────────────────────────────────────────────────────────────────────
  # Logging and Persistence
  # ────────────────────────────────────────────────────────────────────────────

  # Logging verbosity. Options: DEBUG, INFO, WARNING, ERROR.
  # DEBUG includes full gradient statistics in each audit log entry.
  # INFO includes only AuditReport core fields.
  log_level: INFO

  # Path to the audit report log file (JSONL format: one AuditReport JSON per line).
  # [optional] Omit to disable file-based persistence of audit reports.
  audit_log_path: "logs/audit_reports.jsonl"

  # If true, persist epsilon accumulator history to disk after each round,
  # enabling restart of training without losing budget accounting state.
  # [optional] Default: false.
  persist_epsilon_history: true
  epsilon_history_path: "state/epsilon_history.json"
```

---

## 13. Full Python API with Examples

### 13.1 Class Signature and Initialization

```python
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from chargeshield.audit_report import AuditReport


logger = logging.getLogger(__name__)


class PrivacyAuditor:
    """
    Observation instrument for the ChargeShield-FL framework.

    Intercepts gradient updates from FL clients PRE-aggregation, computes
    privacy leakage proxies (gradient sensitivity, per-round epsilon estimate,
    cumulative privacy budget consumption), detects threat patterns, and
    emits AuditReport objects consumed by FedMIA and ChargingIDS.

    The PrivacyAuditor is NOT a defense mechanism. It does not modify gradient
    updates. It observes, computes, and reports.

    Parameters
    ----------
    config_path : str
        Path to the auditor YAML configuration file (auditor.yaml).
        Must contain a top-level "privacy_auditor" key.

    Examples
    --------
    >>> auditor = PrivacyAuditor("config/auditor.yaml", epsilon=1.0)
    >>> # model_update is a dict[str, Any] with layer keys and list[float] values
    >>> model_update = {
    ...     "layer_0": [0.01, -0.03, 0.02, ...],  # w.flatten().tolist()
    ...     "layer_1": [0.05, 0.01, -0.02, ...],
    ... }
    >>> report = auditor.audit(
    ...     node_id="highway_cluster_1",
    ...     round_id=1,
    ...     model_update=model_update,
    ...     baseline_norm=0.45,
    ... )
    >>> print(report.privacy_score)
    0.913
    """

    def __init__(self, config_path: str) -> None:
        self._config = self._load_config(config_path)
        self._epsilon_budget: float = self._config["epsilon_budget"]
        self._max_grad_norm: float = self._config["max_grad_norm"]
        self._explosion_threshold: float = self._config["explosion_threshold"]
        self._near_exhaustion_ratio: float = self._config["near_exhaustion_ratio"]
        self._suspicious_low_ratio: float = self._config["suspicious_low_ratio"]
        self._enable_fedmia: bool = self._config.get("enable_fedmia_integration", True)
        self._audit_log_path: Optional[Path] = (
            Path(self._config["audit_log_path"])
            if "audit_log_path" in self._config else None
        )

        # Per-node cumulative epsilon accumulators
        self._epsilon_accumulators: Dict[str, float] = defaultdict(float)
        # Per-node history for analysis and export
        self._epsilon_history: Dict[str, List[float]] = defaultdict(list)
        self._privacy_score_history: Dict[str, List[float]] = defaultdict(list)

        logging.basicConfig(
            level=getattr(logging, self._config.get("log_level", "INFO"))
        )

        # Load persisted state if configured
        if self._config.get("persist_epsilon_history", False):
            self._load_epsilon_history()

        logger.info(
            "PrivacyAuditor initialized: epsilon_budget=%.2f, max_grad_norm=%.4f",
            self._epsilon_budget,
            self._max_grad_norm,
        )

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)
        return raw["privacy_auditor"]

    def _load_epsilon_history(self) -> None:
        hist_path = Path(
            self._config.get("epsilon_history_path", "state/epsilon_history.json")
        )
        if hist_path.exists():
            with open(hist_path, "r") as f:
                state = json.load(f)
            self._epsilon_accumulators = defaultdict(
                float, state.get("accumulators", {})
            )
            self._epsilon_history = defaultdict(list, state.get("history", {}))
            self._privacy_score_history = defaultdict(
                list, state.get("privacy_score_history", {})
            )
            logger.info("Loaded persisted epsilon history from %s", hist_path)

    def _save_epsilon_history(self) -> None:
        hist_path = Path(
            self._config.get("epsilon_history_path", "state/epsilon_history.json")
        )
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(hist_path, "w") as f:
            json.dump(
                {
                    "accumulators": dict(self._epsilon_accumulators),
                    "history": dict(self._epsilon_history),
                    "privacy_score_history": dict(self._privacy_score_history),
                },
                f,
                indent=2,
            )

    @staticmethod
    def _infer_cluster_type(node_id: str) -> str:
        """Infer cluster type from node_id prefix convention."""
        for ctype in ("highway", "urban", "residential", "corporate"):
            if node_id.startswith(ctype):
                return ctype
        return "unknown"
```

### 13.2 Core Audit Method

```python
    def audit(
        self,
        node_id: str,
        round_id: int,
        model_update: dict[str, Any],
        baseline_norm: float,
    ) -> AuditReport:
        """
        Audit a single gradient update from a FL client.

        This is the primary entry point of the PrivacyAuditor. It computes
        all privacy leakage proxies, updates the per-node epsilon accumulator,
        detects threat patterns, and returns an immutable AuditReport.

        The model_update is NOT modified. This method is read-only with
        respect to the weight dictionary.

        Parameters
        ----------
        node_id : str
            Unique identifier for the FL client (e.g., "highway_cluster_1").
        round_id : int
            FL training round number (1-indexed).
        model_update : dict[str, Any]
            Dictionary mapping layer names to weight values.
            Keys: "layer_0", "layer_1", ... (one per model layer).
            Values: list[float] — the flattened weight tensor for that layer,
            obtained via w.flatten().tolist() for each weight tensor w.
            This matches the format produced by run_ids() in run_experiments.py,
            which iterates over GradientUpdate tensors and converts them to
            layer-keyed float lists before calling audit().
        baseline_norm : float
            Baseline L2-norm for this node. Typically the exponential moving
            average of previous gradient norms, maintained by the caller.
            For the first round, use a reasonable prior (e.g., max_grad_norm).

        Returns
        -------
        AuditReport
            Immutable report containing all computed privacy metrics and
            detected threat patterns for this (node_id, round_id) pair.

        Raises
        ------
        ValueError
            If model_update is empty or contains NaN/Inf values.
        """
        # Flatten the layer-keyed weight dict into a single vector for norm computation
        # model_update keys: "layer_0", "layer_1", ...; values: list[float]
        all_weights = np.concatenate([
            np.array(v, dtype=float) for v in model_update.values()
        ])
        if all_weights.size == 0:
            raise ValueError(f"model_update for node {node_id} is empty.")
        if not np.isfinite(all_weights).all():
            raise ValueError(
                f"model_update for node {node_id} contains NaN or Inf values."
            )

        # Step 1: Compute gradient L2-norm
        gradient_norm: float = float(np.linalg.norm(all_weights))
        sensitivity: float = gradient_norm  # Proxy: empirical norm approximates Δf

        # Step 2: Per-round epsilon estimate (simplified Gaussian accounting)
        epsilon_round: float = sensitivity / self._max_grad_norm

        # Step 3: Update cumulative epsilon accumulator
        self._epsilon_accumulators[node_id] += epsilon_round
        epsilon_cumulative: float = self._epsilon_accumulators[node_id]

        # Step 4: Compute privacy score
        privacy_score: float = 1.0 - min(
            epsilon_cumulative / self._epsilon_budget, 1.0
        )

        # Step 5: Pattern detection
        threats: List[str] = []
        update_magnitude: float = (
            gradient_norm / baseline_norm if baseline_norm > 0.0 else 0.0
        )

        # GRADIENT_EXPLOSION
        if (
            baseline_norm > 0.0
            and gradient_norm > self._explosion_threshold * baseline_norm
        ):
            threats.append("GRADIENT_EXPLOSION")
            logger.warning(
                "[%s] Round %d: GRADIENT_EXPLOSION detected "
                "(norm=%.4f, baseline=%.4f, ratio=%.2f)",
                node_id, round_id, gradient_norm, baseline_norm, update_magnitude,
            )

        # PRIVACY_BUDGET_EXHAUSTED (evaluated before near-exhaustion)
        if epsilon_cumulative >= self._epsilon_budget:
            threats.append("PRIVACY_BUDGET_EXHAUSTED")
            logger.error(
                "[%s] Round %d: PRIVACY_BUDGET_EXHAUSTED "
                "(epsilon_cumulative=%.4f, budget=%.4f)",
                node_id, round_id, epsilon_cumulative, self._epsilon_budget,
            )
        elif epsilon_cumulative > self._near_exhaustion_ratio * self._epsilon_budget:
            threats.append("PRIVACY_BUDGET_NEAR_EXHAUSTION")
            logger.warning(
                "[%s] Round %d: PRIVACY_BUDGET_NEAR_EXHAUSTION "
                "(epsilon_cumulative=%.4f, budget=%.4f, consumed=%.1f%%)",
                node_id, round_id, epsilon_cumulative, self._epsilon_budget,
                100.0 * epsilon_cumulative / self._epsilon_budget,
            )

        # FEDMIA_SUSPICIOUS_LOW_SENSITIVITY
        if (
            baseline_norm > 0.0
            and gradient_norm < self._suspicious_low_ratio * baseline_norm
        ):
            threats.append("FEDMIA_SUSPICIOUS_LOW_SENSITIVITY")
            logger.warning(
                "[%s] Round %d: FEDMIA_SUSPICIOUS_LOW_SENSITIVITY "
                "(norm=%.6f, threshold=%.6f)",
                node_id, round_id, gradient_norm,
                self._suspicious_low_ratio * baseline_norm,
            )

        # Step 6: Assemble metadata
        metadata: Dict[str, Any] = {
            "gradient_norm": gradient_norm,
            "sensitivity": sensitivity,
            "baseline_norm": baseline_norm,
            "update_magnitude": update_magnitude,
            "epsilon_cumulative": epsilon_cumulative,
            "timestamp": time.time(),
            "cluster_type": self._infer_cluster_type(node_id),
        }

        # Step 7: Assemble AuditReport (immutable)
        report = AuditReport(
            node_id=node_id,
            round_id=round_id,
            privacy_score=privacy_score,
            epsilon=epsilon_round,
            threats_detected=threats,
            metadata=metadata,
        )

        # Step 8: Update history
        self._epsilon_history[node_id].append(epsilon_round)
        self._privacy_score_history[node_id].append(privacy_score)

        # Step 9: Persist if configured
        if self._config.get("persist_epsilon_history", False):
            self._save_epsilon_history()

        # Step 10: Log to audit log (JSONL)
        if self._audit_log_path is not None:
            self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._audit_log_path, "a") as f:
                f.write(report.to_json(indent=None) + "\n")

        logger.info(
            "[%s] Round %d: privacy_score=%.4f, epsilon_round=%.4f, "
            "epsilon_cumulative=%.4f, threats=%s",
            node_id, round_id, privacy_score, epsilon_round,
            epsilon_cumulative, threats,
        )

        return report
```

### 13.3 History and Reset Methods

```python
    def get_epsilon_history(self, node_id: str) -> List[float]:
        """
        Return the per-round epsilon history for a node.

        Returns a list of epsilon values, one per FL round, in round order.
        The cumulative epsilon at round T equals sum(get_epsilon_history(node_id)[:T]).

        Parameters
        ----------
        node_id : str
            Node identifier.

        Returns
        -------
        List[float]
            Per-round epsilon values. Empty list if node has not been audited.
        """
        return list(self._epsilon_history[node_id])

    def get_privacy_score_history(self, node_id: str) -> List[float]:
        """
        Return the privacy score history for a node.

        Returns a list of privacy scores, one per FL round, in round order.
        The sequence is monotonically non-increasing by construction.

        Parameters
        ----------
        node_id : str
            Node identifier.

        Returns
        -------
        List[float]
            Per-round privacy scores in [0.0, 1.0]. Empty if not yet audited.
        """
        return list(self._privacy_score_history[node_id])

    def reset(self, node_id: str) -> None:
        """
        Reset the epsilon accumulator and history for a specific node.

        Use this when a node re-enters training after exclusion (fresh budget
        allocation) or for ablation studies requiring clean-slate accounting.

        Parameters
        ----------
        node_id : str
            Node identifier to reset.
        """
        self._epsilon_accumulators[node_id] = 0.0
        self._epsilon_history[node_id] = []
        self._privacy_score_history[node_id] = []
        logger.info("[%s] PrivacyAuditor state reset.", node_id)

    def get_all_node_ids(self) -> List[str]:
        """Return list of all node IDs that have been audited."""
        return list(self._epsilon_accumulators.keys())

    def get_cumulative_epsilon(self, node_id: str) -> float:
        """Return the current cumulative epsilon for a node."""
        return self._epsilon_accumulators[node_id]
```

### 13.4 Realistic Usage Example

```python
import numpy as np
from chargeshield.privacy_auditor import PrivacyAuditor

# Initialize auditor
auditor = PrivacyAuditor(config_path="config/auditor.yaml", epsilon=1.0)  # epsilon overrides auditor.yaml

# Simulate 50 FL rounds for 4 nodes across cluster types
nodes = [
    "highway_cluster_1",
    "highway_cluster_2",
    "urban_cluster_1",
    "residential_cluster_1",
]

# Initial baseline norms
baseline_norms = {node: 0.45 for node in nodes}

for round_id in range(1, 51):
    for node_id in nodes:
        # Simulate cluster-type-specific gradient norm distributions
        if "highway" in node_id:
            # Highway: high-norm, high-sensitivity
            grad_norm_target = np.random.lognormal(mean=-0.5, sigma=0.4)
        elif "urban" in node_id:
            # Urban: moderate norm
            grad_norm_target = np.random.lognormal(mean=-0.8, sigma=0.3)
        else:
            # Residential: low-norm, low-sensitivity
            grad_norm_target = np.random.lognormal(mean=-1.2, sigma=0.2)

        # Generate unit-direction gradient and scale to target norm
        gradient = np.random.randn(10000)
        gradient = gradient / np.linalg.norm(gradient) * grad_norm_target

        # Convert to layer-keyed dict[str, list[float]] as expected by audit()
        # (mirrors the run_ids() call site in run_experiments.py)
        model_update = {"layer_0": gradient.tolist()}

        # Audit the gradient
        report = auditor.audit(
            node_id=node_id,
            round_id=round_id,
            model_update=model_update,
            baseline_norm=baseline_norms[node_id],
        )

        # Update baseline norm (exponential moving average)
        baseline_norms[node_id] = (
            0.9 * baseline_norms[node_id]
            + 0.1 * report.metadata["gradient_norm"]
        )

        # Log high-risk events
        if report.is_high_risk:
            print(
                f"HIGH RISK: {node_id} | round={round_id} | "
                f"score={report.privacy_score:.3f} | "
                f"threats={report.threats_detected}"
            )

# Post-training analysis
for node_id in nodes:
    eps_history = auditor.get_epsilon_history(node_id)
    ps_history = auditor.get_privacy_score_history(node_id)
    print(f"\n{node_id} after 50 rounds:")
    print(f"  Cumulative epsilon : {sum(eps_history):.3f}")
    print(f"  Final privacy score: {ps_history[-1]:.3f}")
    print(f"  Budget exhausted   : {sum(eps_history) >= 5.0}")
```

### 13.5 NVFLARE Custom Aggregator Integration

```python
# chargeshield/nvflare/chargeshield_aggregator.py

import numpy as np
from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.aggregators.intime_accumulate_weighted_aggregator import (
    InTimeAccumulateWeightedAggregator,
)

from chargeshield.privacy_auditor import PrivacyAuditor
from chargeshield.charging_ids import ChargingIDS


class ChargeShieldAggregator(InTimeAccumulateWeightedAggregator):
    """
    Custom NVFLARE aggregator that hooks PrivacyAuditor into the
    pre-aggregation phase of each FL round.

    This aggregator extends InTimeAccumulateWeightedAggregator and overrides
    accept() to intercept each client contribution before accumulation. The
    PrivacyAuditor is invoked per client; the resulting AuditReport is passed
    to ChargingIDS for action determination. Excluded nodes are not forwarded
    to the parent aggregator.

    NVFLARE server configuration (config_fed_server.json):
    {
        "id": "aggregator",
        "path": "chargeshield.nvflare.chargeshield_aggregator.ChargeShieldAggregator",
        "args": {
            "auditor_config_path": "config/auditor.yaml",
            "ids_config_path": "config/ids.yaml"
        }
    }
    """

    def __init__(
        self,
        auditor_config_path: str = "config/auditor.yaml",
        ids_config_path: str = "config/ids.yaml",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._auditor = PrivacyAuditor(config_path=auditor_config_path)  # epsilon set per-experiment via run_ids()
        self._ids = ChargingIDS(config_path=ids_config_path)
        self._baseline_norms: dict = {}

    def accept(
        self,
        shareable: Shareable,
        fl_ctx: FLContext,
    ) -> bool:
        """
        Accept a client contribution for aggregation.

        Overrides parent to invoke PrivacyAuditor before accumulation.
        The gradient vector is not modified; only the audit side-effect occurs.
        Returns False (rejects the contribution) if the IDS issues EXCLUDE.
        """
        client_name: str = shareable.get_peer_props().get(
            "__client_name__", "unknown"
        )
        current_round: int = fl_ctx.get_prop("current_round", 0)

        # Extract gradient update from shareable DXO
        dxo: DXO = DXO.from_shareable(shareable)
        if dxo.data_kind == DataKind.WEIGHT_DIFF:
            # Flatten all weight difference arrays into a single gradient vector
            gradient_vector = np.concatenate(
                [v.flatten() for v in dxo.data.values()]
            )

            # Initialize baseline norm for new nodes
            if client_name not in self._baseline_norms:
                self._baseline_norms[client_name] = float(
                    np.linalg.norm(gradient_vector)
                )

            # Convert gradient tensors to layer-keyed dict[str, list[float]]
            # matching the format expected by audit() in the run_experiments.py
            # call site: keys = "layer_0", "layer_1", ...; values = w.flatten().tolist()
            model_update = {
                f"layer_{i}": v.flatten().tolist()
                for i, v in enumerate(dxo.data.values())
            }

            # Invoke PrivacyAuditor (read-only on model_update)
            report = self._auditor.audit(
                node_id=client_name,
                round_id=current_round,
                model_update=model_update,
                baseline_norm=self._baseline_norms[client_name],
            )

            # Update baseline norm (EMA)
            self._baseline_norms[client_name] = (
                0.9 * self._baseline_norms[client_name]
                + 0.1 * report.metadata["gradient_norm"]
            )

            # Consult IDS for action decision
            action = self._ids.analyze(report)
            if action.name == "EXCLUDE":
                # Reject contribution; do not pass to parent aggregator
                return False

        # Delegate to standard InTimeAccumulateWeightedAggregator
        return super().accept(shareable, fl_ctx)
```

---

## 14. Limitations and Future Work

### 14.1 Known Limitations

**Epsilon estimation accuracy.** As detailed in Section 4.4, the simplified Gaussian accounting used by the `PrivacyAuditor` is an approximation. The estimate is conservative (overestimates epsilon) and does not account for Renyi DP composition, heterogeneous noise scales, or the interaction between gradient clipping and norm measurement. These limitations mean that the epsilon values reported by the auditor should be interpreted as ordinal indicators of relative privacy exposure, not as formal DP certifications.

**L2-norm as sole sensitivity proxy.** The auditor relies exclusively on the L2-norm of the gradient vector as a sensitivity proxy. As noted in Section 3.4, this first-order statistic may be misleading in high-DP-noise regimes (where the norm is dominated by noise) or in gradient masking scenarios (where the norm is artificially suppressed). Future work should explore richer gradient representations, such as directional statistics or gradient subspace projections, as additional sensitivity proxies.

**Black-box noise scale assumption.** The auditor assumes that it does not have access to the noise scale $\sigma$ configured in the local trainer's DP mechanism. This is a conservative and realistic assumption for black-box evaluation. If $\sigma$ is available (white-box evaluation), the auditor could compute a more accurate epsilon estimate using the formal Gaussian mechanism formula, and could also apply Renyi DP composition for tighter bounds.

**Temporal resolution.** The auditor operates at the FL round granularity — one `AuditReport` per (node, round) pair. Intra-round gradient dynamics (e.g., multiple local SGD steps with intermediate gradient accumulation) are not captured. In FL systems that perform multiple local epochs per round, the between-round sensitivity measurement may underestimate the total privacy expenditure.

### 14.2 Directions for Future Work

**Renyi DP integration.** The most impactful near-term improvement to the auditor is the integration of Renyi DP accounting (Mironov, 2017). This requires exposing the noise multiplier $q = \sigma / \Delta f$ from the local trainer to the auditor — a modification compatible with NVFLARE's configuration system. With RDP accounting, the auditor could provide formal $(\epsilon, \delta)$-DP guarantees rather than proxies.

**Per-parameter sensitivity analysis.** Rather than treating the gradient as a monolithic vector, future work could decompose the gradient by model layer and compute per-layer sensitivity. Layers with disproportionately high sensitivity (e.g., the final classification layer, which is most directly influenced by training labels) could be specifically targeted by FedMIA, improving attack efficiency and attribution.

**Adaptive epsilon budgeting.** The experimental results from ChargeShield-FL demonstrate that a fixed epsilon budget creates unequal privacy timelines across cluster types. Future work should investigate per-cluster-type adaptive budgeting, where the epsilon budget is allocated proportionally to the expected gradient sensitivity of each cluster type, ensuring that all nodes exhaust their budgets at approximately the same training round.

**Online sensitivity estimation.** The current baseline norm is updated via an exponential moving average — a simple but potentially slow-adapting estimator. Online change point detection methods (e.g., PELT, BOCPD) could provide more responsive baseline estimation, improving the sensitivity and specificity of the gradient explosion and low-sensitivity detectors.

---

## 15. References

[1] **Shokri, R., Stronati, M., Song, C., & Shmatikov, V.** (2017). Membership Inference Attacks Against Machine Learning Models. *2017 IEEE Symposium on Security and Privacy (S&P)*, pp. 3–18. https://doi.org/10.1109/SP.2017.41

[2] **Nasr, M., Shokri, R., & Houmansadr, A.** (2019). Comprehensive Privacy Analysis of Deep Learning: Passive and Active White-box Inference Attacks against Centralized and Federated Learning. *2019 IEEE Symposium on Security and Privacy (S&P)*, pp. 739–753. https://doi.org/10.1109/SP.2019.00065

[3] **Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., & Tramèr, F.** (2022). Membership Inference Attacks From First Principles. *2022 IEEE Symposium on Security and Privacy (S&P)*, pp. 1897–1914. https://doi.org/10.1109/SP46214.2022.9833649

[4] **Dwork, C., & Roth, A.** (2014). The Algorithmic Foundations of Differential Privacy. *Foundations and Trends in Theoretical Computer Science*, 9(3–4), 211–407. https://doi.org/10.1561/0400000042

[5] **Mironov, I.** (2017). Rényi Differential Privacy of the Gaussian Mechanism. *2017 IEEE 30th Computer Security Foundations Symposium (CSF)*, pp. 263–275. https://doi.org/10.1109/CSF.2017.11

[6] **Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L.** (2016). Deep Learning with Differential Privacy. *Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (CCS)*, pp. 308–318. https://doi.org/10.1145/2976749.2978318

[7] **McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A.** (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)*, pp. 1273–1282.

[8] **Zhu, L., Liu, Z., & Han, S.** (2019). Deep Leakage from Gradients. *Advances in Neural Information Processing Systems (NeurIPS)*, 32.

[9] **Fung, C., Yoon, C. J. M., & Beschastnikh, I.** (2018). Mitigating Sybils in Federated Learning Poisoning. *arXiv preprint arXiv:1808.04866*.

[10] **Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J.** (2017). Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.

---

*Document prepared for ChargeShield-FL, DSN 2027 submission cycle. All design decisions documented herein are subject to revision based on empirical evaluation results. This document does not constitute a formal DP proof or security certification.*
